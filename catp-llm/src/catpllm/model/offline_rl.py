import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
from enum import Enum
from src.config import GlobalToolConfig, DEFAULT_START_TASK_NAME
from src.catpllm.model.prompt import INSTRUCTION_PROMPT, TASK_PROMPT, TOOL_PROMPT2_P1, TOOL_PROMPT2_P2
from src.catpllm.utils.utils import find_sink_nodes_in_plan


INF = 1e5
TOOL_PREDICTION_MODE = 1
DEPENDENCY_PREDICTION_MODE = 2


class TaskType(Enum):
    """
    Types of tasks, used to create action masks.
    """
    Unknown = 0
    ImageInImageOut = 1
    ImageInTextOut = 2
    TextInImageOut = 3
    TextInTextOut = 4
    ImageTextInTextOut = 5
    TextTextInTextOut = 6
    ImageInImageTextOut = 7
    ImageInImageTextTextOut = 8
    ImageInTextTextOut = 9

    @classmethod
    def determine_task_type(cls, task):
        if task in GlobalToolConfig.task_io_dict['in:image-out:image']:
            return cls.ImageInImageOut
        if task in GlobalToolConfig.task_io_dict['in:image-out:text']:
            return cls.ImageInTextOut
        if task in GlobalToolConfig.task_io_dict['in:text-out:image']:
            return cls.TextInImageOut
        if task in GlobalToolConfig.task_io_dict['in:text-out:text']:
            return cls.TextInTextOut
        if task in GlobalToolConfig.task_io_dict['in:image,text-out:text']:
            return cls.ImageTextInTextOut
        if task in GlobalToolConfig.task_io_dict['in:text,text-out:text']:
            return cls.TextTextInTextOut
        if task in GlobalToolConfig.task_io_dict['in:image-out:image,text']:
            return cls.ImageInImageTextOut
        if task in GlobalToolConfig.task_io_dict['in:image-out:image,text,text']:
            return cls.ImageInImageTextTextOut
        if task in GlobalToolConfig.task_io_dict['in:image-out:text,text']:
            return cls.ImageInTextTextOut
        return cls.Unknown


def create_action_mask(action_logits, generated_tool_tokens, device, mode, prev_token_is_sod=False, task_type=TaskType.Unknown, cur_plan=None):
    """
    Create mask for action prediction (only for inference).
    """
    assert len(action_logits.shape) == 1
    mask = torch.zeros(action_logits.shape, dtype=torch.float, device=device)
    mask[0] = -INF  # skip the first token, which is SOP/SOD token

    if mode == TOOL_PREDICTION_MODE:
        # rule 1: filter tool tokens according to the task type.
        valid_tool_tokens = []
        if task_type == TaskType.Unknown:
            valid_tool_tokens = list(GlobalToolConfig.tool_token_vocabulary.values())

        elif task_type == TaskType.ImageInImageOut:
            valid_tool_tokens.extend(GlobalToolConfig.tool_token_io_dict_collection['in:image-out:image'])

        elif task_type == TaskType.TextInImageOut:
            valid_tool_tokens.extend(GlobalToolConfig.tool_token_io_dict_collection['in:text-out:text'])
            valid_tool_tokens.extend(GlobalToolConfig.tool_token_io_dict_collection['in:text-out:image'])
            if len(generated_tool_tokens) > 0:  # only if this is not the first tool token, we can add tools whose input types does not match the task input data.
                valid_tool_tokens.extend(GlobalToolConfig.tool_token_io_dict_collection['in:image-out:image'])

        elif task_type == TaskType.ImageInTextOut:
            valid_tool_tokens.extend(GlobalToolConfig.tool_token_io_dict_collection['in:image-out:image'])
            valid_tool_tokens.extend(GlobalToolConfig.tool_token_io_dict_collection['in:image-out:text'])
            if len(generated_tool_tokens) > 0:  # likewise
                valid_tool_tokens.extend(GlobalToolConfig.tool_token_io_dict_collection['in:text-out:text'])

        elif task_type == TaskType.TextInTextOut:
            valid_tool_tokens.extend(GlobalToolConfig.tool_token_io_dict_collection['in:text-out:text'])

        elif task_type == TaskType.ImageTextInTextOut:
            valid_tool_tokens = list(GlobalToolConfig.tool_token_vocabulary.values())

        elif task_type == TaskType.TextTextInTextOut:
            valid_tool_tokens.extend(GlobalToolConfig.tool_token_io_dict_collection['in:text-out:text'])
            valid_tool_tokens.extend(GlobalToolConfig.tool_token_io_dict_collection['in:text,text-out:text'])

        elif task_type == TaskType.ImageInImageTextOut:
            valid_tool_tokens.extend(GlobalToolConfig.tool_token_io_dict_collection['in:image-out:image'])
            valid_tool_tokens.extend(GlobalToolConfig.tool_token_io_dict_collection['in:image-out:text'])
            if len(generated_tool_tokens) > 0:  # likewise
                valid_tool_tokens.extend(GlobalToolConfig.tool_token_io_dict_collection['in:text-out:text'])

        elif task_type == TaskType.ImageInImageTextTextOut:
            valid_tool_tokens.extend(GlobalToolConfig.tool_token_io_dict_collection['in:image-out:image'])
            valid_tool_tokens.extend(GlobalToolConfig.tool_token_io_dict_collection['in:image-out:text'])
            if len(generated_tool_tokens) > 0:  # likewise
                valid_tool_tokens.extend(GlobalToolConfig.tool_token_io_dict_collection['in:text-out:text'])

        elif task_type == TaskType.ImageInTextTextOut:
            valid_tool_tokens.extend(GlobalToolConfig.tool_token_io_dict_collection['in:image-out:image'])
            valid_tool_tokens.extend(GlobalToolConfig.tool_token_io_dict_collection['in:image-out:text'])
            if len(generated_tool_tokens) > 0:  # likewise
                valid_tool_tokens.extend(GlobalToolConfig.tool_token_io_dict_collection['in:text-out:text'])

        else: 
            valid_tool_tokens = list(GlobalToolConfig.tool_token_vocabulary.values())

        # rule 2: filter already generated tools (disable it if we allow the same tool appears multiple times in the plan)
        for tool_token in generated_tool_tokens:
            if tool_token in valid_tool_tokens:
                valid_tool_tokens.remove(tool_token)
        
        # rule 3: determine whether to add eop token or not
        if len(generated_tool_tokens) > 0:  # only when there are more than one selected tools.
            last_tool_token = generated_tool_tokens[-1]
            if task_type in [TaskType.ImageInImageOut, TaskType.TextInImageOut]:
                if 'image' in GlobalToolConfig.tool_token_io_dict[last_tool_token][-1]:
                    valid_tool_tokens.append(GlobalToolConfig.eop_token)
            elif task_type in [TaskType.ImageInTextOut, TaskType.TextTextInTextOut, TaskType.TextInTextOut, TaskType.ImageTextInTextOut]:
                if 'text' in GlobalToolConfig.tool_token_io_dict[last_tool_token][-1]:
                    valid_tool_tokens.append(GlobalToolConfig.eop_token)
            else:
                assert cur_plan is not None
                sink_nodes = find_sink_nodes_in_plan(cur_plan, is_token=True, return_token=True)
                output_types = [GlobalToolConfig.tool_token_io_dict[node][-1] for node in sink_nodes]
                num_image_outputs, num_text_outputs = 0, 0
                for output_type in output_types:
                    if 'text' in output_type:
                        num_text_outputs += 1
                    elif 'image' in output_type:
                        num_image_outputs += 1

                if task_type == TaskType.ImageInTextTextOut:
                    if num_text_outputs >= 2:
                        valid_tool_tokens.append(GlobalToolConfig.eop_token)
                elif task_type == TaskType.ImageInImageTextOut:
                    if num_image_outputs >= 1 and num_text_outputs >= 1:
                        valid_tool_tokens.append(GlobalToolConfig.eop_token)
                elif task_type == TaskType.ImageInImageTextTextOut:
                    if num_image_outputs >= 1 and num_text_outputs >= 2:
                        valid_tool_tokens.append(GlobalToolConfig.eop_token)
                else:
                    valid_tool_tokens.append(GlobalToolConfig.eop_token)
        
        # perform masking
        valid_indices = [valid_tool_token - GlobalToolConfig.tool_token_start for valid_tool_token in valid_tool_tokens]
        for i in range(1, mask.shape[0]):
            if i not in valid_indices:
                mask[i] = -INF

    if mode == DEPENDENCY_PREDICTION_MODE:
        current_tool_token = generated_tool_tokens[-1]
        current_tool = GlobalToolConfig.tool_token_vocabulary_reverse[current_tool_token]

        # rule 1: when generating dependency tokens, we should only pay attention to those 
        # associated with the previous generated tools, since we cannot generate a dependency 
        # between the current tool and the non-generated tool.
        valid_dependency_tokens = []  # GlobalToolConfig.dependency_token_vocabuary[DEFAULT_START_TASK_NAME]
        for tool_token in generated_tool_tokens[:-1]:
            if tool_token == GlobalToolConfig.sop_token or tool_token == GlobalToolConfig.eop_token:
                continue
            tool = GlobalToolConfig.tool_token_vocabulary_reverse[tool_token]
            # only those tools whose output types match the input type of current tool can be added in valid tokens.
            if GlobalToolConfig.tool_io_dict[tool][1] == GlobalToolConfig.tool_io_dict[current_tool][0]:
                valid_dependency_tokens.append(GlobalToolConfig.dependency_token_vocabulary[tool])
        
        # rule 2: only when the input type of the current tool matches the task input, we can add the special token of "Input of Query"
        if task_type == TaskType.Unknown:
            valid_dependency_tokens.append(GlobalToolConfig.dependency_token_vocabulary[DEFAULT_START_TASK_NAME])
        elif task_type in [TaskType.ImageInImageOut, TaskType.ImageInTextOut, TaskType.ImageTextInTextOut, TaskType.ImageInImageTextOut, 
                           TaskType.ImageInImageTextTextOut, TaskType.ImageInTextTextOut]:
            if 'image' in GlobalToolConfig.tool_io_dict[current_tool][0]:
                valid_dependency_tokens.append(GlobalToolConfig.dependency_token_vocabulary[DEFAULT_START_TASK_NAME])
        elif task_type in [TaskType.TextInImageOut, TaskType.TextInTextOut, TaskType.TextTextInTextOut, TaskType.ImageTextInTextOut]:
            if 'text' in GlobalToolConfig.tool_io_dict[current_tool][0]:
                valid_dependency_tokens.append(GlobalToolConfig.dependency_token_vocabulary[DEFAULT_START_TASK_NAME])
        else:
            valid_dependency_tokens.append(GlobalToolConfig.dependency_token_vocabulary[DEFAULT_START_TASK_NAME])

        # rule 3: before adding a dependency token, the special token eod cannot be activated.
        if not prev_token_is_sod:
            valid_dependency_tokens.append(GlobalToolConfig.eod_token)

        # perform masking
        valid_indices = [valid_dependency_token - GlobalToolConfig.dependency_token_start for valid_dependency_token in valid_dependency_tokens]
        for i in range(1, mask.shape[0]):
            if i not in valid_indices:
                mask[i] = -INF
    return mask


class OfflineRLPolicy(nn.Module):
    """
    We use decision transformer as the offline rl algorithm.
    """
    def __init__(
            self,
            token_encoder,
            tokenizer,
            llm,
            llm_embed_dim,
            num_tool_tokens,
            num_dependency_tokens,
            max_num_tokens,
            max_window_size=2,  # the max size of context window (see decision transformer)
            max_ep_len=100,  # max episode length (= max number of tool and dependency tokens that can be generated)
            device='cuda' if torch.cuda.is_available() else 'cpu',
            device_out = None,
            disable_context_aug=False,
            disable_masking=False,
            **kwargs
    ):
        super().__init__()
        
        if device_out is None:
            device_out = device

        self.token_encoder = token_encoder
        self.tokenizer = tokenizer
        self.llm = llm
        self.llm_vocab = llm.get_input_embeddings()  # the word vocabulary of the LLM, which turns word token into embeddings
        self.llm_embed_dim = llm_embed_dim
        self.num_tool_tokens = num_tool_tokens
        self.num_dependency_tokens = num_dependency_tokens
        self.max_num_tokens = max_num_tokens
        self.max_ep_len = max_ep_len
        self.device = device
        self.device_out = device_out
        self.disable_context_aug = disable_context_aug
        self.disable_masking = disable_masking

        # ========== prompt/token special ==========
        self.instruction_prompt_embed = self.llm_vocab(torch.tensor(self.tokenizer(INSTRUCTION_PROMPT).input_ids, dtype=torch.int64, device=device))
        self.tool_prompt2_p1_embed = self.llm_vocab(torch.tensor(self.tokenizer(TOOL_PROMPT2_P1).input_ids, dtype=torch.int64, device=device))
        self.tool_prompt2_p2_embed = self.llm_vocab(torch.tensor(self.tokenizer(TOOL_PROMPT2_P2).input_ids, dtype=torch.int64, device=device))
        self.all_tool_tokens = torch.tensor(list(GlobalToolConfig.tool_token_vocabulary.values()), dtype=torch.int64, device=device)

        # ========== decision transformer special ==========
        self.embed_timestep = nn.Embedding(max_ep_len + 1, llm_embed_dim).to(device)
        self.embed_return = nn.Linear(1, llm_embed_dim).to(device)
        self.embed_tool_action = nn.Linear(1, llm_embed_dim).to(device)
        self.embed_dep_action = nn.Linear(1, llm_embed_dim).to(device)
        self.embed_ln = nn.LayerNorm(llm_embed_dim).to(device)

        self.tool_head = nn.Linear(llm_embed_dim, max_num_tokens).to(device)
        self.dependency_head = nn.Linear(llm_embed_dim, max_num_tokens).to(device)
        
        # the following are used for efficient auto-regressive prediction
        self.stacked_inputs_dq = deque([torch.zeros((1, 0, llm_embed_dim), device=device)], maxlen=max_window_size - 1)
        self.prompt_cache = None
        self.generated_tool_tokens = []
        
        # the following are used for efficiently saving the modules except llm
        self.modules_except_llm = nn.ModuleList([
            self.token_encoder, self.embed_timestep, self.embed_return, self.embed_tool_action, self.embed_dep_action,
            self.embed_ln, self.tool_head, self.dependency_head
        ])

    def forward(self, states, returns, actions, timesteps, labels, task_info, sample_info,
                sample_size, attention_mask=None, teacher_forcing=True, **kwargs):
        
        """
        Forward function.

        task_info, sample_info: information of task specification and data sample, used to wrap task prompt.
        sample_size: the size of the data sample.
        """
        if teacher_forcing:
            return self._teacher_forcing(states, returns, actions, timesteps, labels, task_info, sample_info,
                                         sample_size, attention_mask, **kwargs)
        else:
            return self._auto_regressive(states, returns, actions, timesteps, labels, task_info, sample_info,
                                         sample_size, attention_mask, **kwargs)

    def _teacher_forcing(self, states, returns, actions, timesteps, labels, task_info, sample_info,
                         sample_size, attention_mask=None,):
        """
        Predict in teacher forcing  (for training).
        """
        assert returns.shape[0] == 1, 'batch size should be 1 to avoid CUDA memory exceed'

        # Step 1: process actions, returns and timesteps first as they are simple
        timesteps = timesteps.to(self.device)  # shape: (1, seq_len)
        returns = returns.to(self.device)  # shape: (1, seq_len, 1)
        actions = actions.to(self.device)  # shape: (1, seq_len, 1)

        time_embeddings = self.embed_timestep(timesteps)  # shape: (1, seq_len, embed_size)
        returns_embeddings = self.embed_return(returns) + time_embeddings  # shape: (1, seq_len, embed_size)

        # 1.1 embed tool and dependency actions
        tool_actions_positions = labels < GlobalToolConfig.dependency_token_start
        action_embeddings = []
        for i in range(actions.shape[1]):
            if tool_actions_positions[0, i]: 
                action_embeddings.append(self.embed_tool_action(actions[:, i:i + 1]))
            else:
                action_embeddings.append(self.embed_dep_action(actions[:, i:i + 1]))
        action_embeddings = torch.cat(action_embeddings, dim=1) + time_embeddings  # shape: (1, seq_len, embed_size)

        # Step 2: process prompt, and turn it into embeddings
        task_prompt = TASK_PROMPT.replace('[Task Specification]', str(task_info))
        task_prompt = task_prompt.replace('[Task Input Attributes]', str(sample_info))
        task_prompt_embed = self.llm_vocab(torch.tensor(self.tokenizer(task_prompt).input_ids, dtype=torch.int64, device=self.device))
            
        tool_embed_pure = self.token_encoder(self.all_tool_tokens, sample_size, fuse_cost=False)
        tool_embed_with_cost = self.token_encoder(self.all_tool_tokens, sample_size, fuse_cost=True)
        prompt_embed = torch.cat([self.instruction_prompt_embed.detach(), self.tool_prompt2_p1_embed.detach(), tool_embed_pure,
                                self.tool_prompt2_p2_embed.detach(), tool_embed_with_cost, task_prompt_embed], dim=0).unsqueeze(0).to(self.device)
            
        # Step 3: process states, turn them into embeddings.
        state_embeddings = []
        for i in range(len(states[0])):
            state = states[0][i]
            state = torch.tensor(state, dtype=torch.int64, device=self.device)
            state_embedding = self.token_encoder(state, sample_size)
            state_embedding = state_embedding.unsqueeze(0) + time_embeddings[0, i]
            state_embeddings.append(state_embedding)
        
        # Step 4: stack returns, states, actions embeddings.
        # this makes the sequence look like (R_1, s_1-1, s_1-2, ..., s_1-n, a_1, R_2, s_2-1, ..., s_2-m, a_2, ...)
        # which works nice in an autoregressive manner since states predict actions
        stacked_inputs = []
        action_embed_positions = np.zeros(returns_embeddings.shape[1], dtype=np.int32)  # record the positions of action embeddings
        for i in range(returns_embeddings.shape[1]):
            stacked_input = torch.cat((returns_embeddings[:, i:i + 1], state_embeddings[i], action_embeddings[:, i:i + 1]), dim=1)
            stacked_inputs.append(stacked_input)
            action_embed_positions[i] = state_embeddings[i].shape[1] + 2 
        action_embed_positions = np.cumsum(action_embed_positions)
        stacked_inputs = torch.cat(stacked_inputs, dim=1)

        # prepend prompt embeddings
        inputs = torch.cat([prompt_embed, stacked_inputs], dim=1)
        action_embed_positions += prompt_embed.shape[1]

        inputs = inputs[:, -self.llm_embed_dim:, :]  # truncate sequence length (should not exceed plm embed size)
        inputs_ln = self.embed_ln(inputs)  # layer normalization
        
        # Step 5: feed stacked embeddings into the plm
        # create attention mask
        if attention_mask is None:
            # 1 if can be attended to, 0 if not
            attention_mask = torch.ones((inputs_ln.shape[0], inputs_ln.shape[1]), dtype=torch.long, device=self.device)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.llm(
            inputs_embeds=inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = transformer_outputs['last_hidden_state']

        # Step 6: predict actions
        # we need to locate the logits corresponding to the state embeddings
        # simply using `action_embed_positions[i] - 2` will do.
        logits_used = logits[:, action_embed_positions - 2]
        
        # use tool head or dependency head for prediction according to the token type
        action_logits, gt = [], []
        for i in range(labels.shape[1]):
            if labels[0, i] < GlobalToolConfig.dependency_token_start:
                # use tool head for prediction
                action_logits.append(self.tool_head(logits_used[:,i:i + 1]))
                gt.append(labels[0, i] - GlobalToolConfig.tool_token_start)
            else:
                # use dependency head for prediction
                action_logits.append(self.dependency_head(logits_used[:,i:i + 1]))
                gt.append(labels[0, i] - GlobalToolConfig.dependency_token_start)
        action_logits = torch.cat(action_logits, dim=1)
        gt = torch.stack(gt).unsqueeze(0)

        return action_logits, gt


    def _auto_regressive(self, cur_state, cur_return, cur_action, cur_timestep, cur_label, task_info, sample_info,
                         sample_size, attention_mask=None, mode=TOOL_PREDICTION_MODE):
        """
        Predict in auto regressive.
        """
        assert mode in [TOOL_PREDICTION_MODE, DEPENDENCY_PREDICTION_MODE]

        # Step 1: stack previous state, action, return features in the dequeue
        prev_stacked_inputs = []
        for i in range(len(self.stacked_inputs_dq)):
            prev_stacked_inputs.append(self.stacked_inputs_dq[i])
        prev_stacked_inputs = torch.cat(prev_stacked_inputs, dim=1)

        # Step 2: process target return and timesteps
        cur_return = torch.as_tensor(cur_return, dtype=torch.float32, device=self.device).reshape(1, 1, 1)
        cur_timestep = torch.as_tensor(cur_timestep, dtype=torch.int64, device=self.device).reshape(1, 1)

        time_embeddings = self.embed_timestep(cur_timestep)
        return_embeddings = self.embed_return(cur_return) + time_embeddings

        # Step 3: process prompt, and turn it into embeddings
        if self.prompt_cache is not None:
            prompt_embed = self.prompt_cache.detach()
        else:
            task_prompt = TASK_PROMPT.replace('[Task Specification]', str(task_info))
            task_prompt = task_prompt.replace('[Task Input Attributes]', str(sample_info))
            task_prompt_embed = self.llm_vocab(torch.tensor(self.tokenizer(task_prompt).input_ids, dtype=torch.int64, device=self.device))
                    
            tool_embed_pure = self.token_encoder(self.all_tool_tokens, sample_size, fuse_cost=False)
            tool_embed_with_cost = self.token_encoder(self.all_tool_tokens, sample_size, fuse_cost=True)
            prompt_embed = torch.cat([self.instruction_prompt_embed.detach(), self.tool_prompt2_p1_embed.detach(), tool_embed_pure,
                                    self.tool_prompt2_p2_embed.detach(), tool_embed_with_cost, task_prompt_embed], dim=0).unsqueeze(0).to(self.device)
                    
            self.prompt_cache = prompt_embed.detach()

        # Step 4: process state
        state = torch.tensor(cur_state, dtype=torch.int64, device=self.device)
        state_embeddings = self.token_encoder(state, sample_size)
        state_embeddings = state_embeddings.unsqueeze(0) + time_embeddings

        # Step 5: stack return, stage and previous embeddings
        stacked_inputs = torch.cat((return_embeddings, state_embeddings), dim=1)  # mind the order
        stacked_inputs = torch.cat((prev_stacked_inputs, stacked_inputs), dim=1)  # mind the order

        # prepend prompt embeddings
        inputs = torch.cat([prompt_embed, stacked_inputs], dim=1)
        inputs = inputs[:, -self.llm_embed_dim:, :]  # truncate sequence length (should not exceed plm embed size)
        inputs_ln = self.embed_ln(inputs)  # layer normalization

        # 1 if can be attended to, 0 if not
        attention_mask = torch.ones((inputs_ln.shape[0], inputs_ln.shape[1]), dtype=torch.long, device=self.device)

        transformer_outputs = self.llm(
            inputs_embeds=inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = transformer_outputs['last_hidden_state']

        # Step 6: predict action
        logits_used = logits[:, -1:]
        if mode == TOOL_PREDICTION_MODE:
            action_logits = self.tool_head(logits_used)
            action_logits = action_logits.reshape(-1)
            truncated_action_logits = action_logits[:self.num_tool_tokens]
            # mask = create_action_mask(truncated_action_logits, self.generated_tool_tokens[:-1], self.device, mode)
            mask = torch.zeros(truncated_action_logits.shape, dtype=torch.float, device=self.device)
            action, _ = self._sample(truncated_action_logits, mask)
            token = action + GlobalToolConfig.tool_token_start
            self.generated_tool_tokens.append(cur_label.item())
        else:
            action_logits = self.dependency_head(logits_used)
            action_logits = action_logits.reshape(-1)
            truncated_action_logits = action_logits[:self.num_dependency_tokens]
            # mask = create_action_mask(truncated_action_logits, self.generated_tool_tokens[:-1], self.device, mode)
            mask = torch.zeros(truncated_action_logits.shape, dtype=torch.float, device=self.device)
            action, _ = self._sample(truncated_action_logits, mask)
            token = action + GlobalToolConfig.dependency_token_start

        # compute action embeddings 
        action_tensor = torch.zeros(1, 1, 1, dtype=torch.float32, device=self.device)
        action_tensor[..., 0] = action / GlobalToolConfig.max_num_tokens
        if token < GlobalToolConfig.dependency_token_start:
            action_embeddings = self.embed_tool_action(action_tensor) + time_embeddings
        else:
            action_embeddings = self.embed_dep_action(action_tensor) + time_embeddings
        
        # stack return, state, action, and cache them
        stacked_inputs = torch.cat((return_embeddings, state_embeddings, action_embeddings), dim=1)  # mind the order
        self.stacked_inputs_dq.append(stacked_inputs)

        return token, action_logits

    def inference(self, state, target_return, timestep, task_id, task_info, sample_info, sample_size, mode, return_logits=False):
        """
        Inference function, used for sampling tool/dependency tokens.

        :param task_info, sample_info: information of task specification and data sample, used to wrap task prompt.
        :param sample_size: the size of the data sample.
        :param mode: prediction mode (tool or dependency prediction).
        :param return_logits: return action logits
        """
        assert mode in [TOOL_PREDICTION_MODE, DEPENDENCY_PREDICTION_MODE]

        # Step 1: stack previous state, action, return features in the dequeue
        prev_stacked_inputs = []
        for i in range(len(self.stacked_inputs_dq)):
            prev_stacked_inputs.append(self.stacked_inputs_dq[i])
        prev_stacked_inputs = torch.cat(prev_stacked_inputs, dim=1)

        # Step 2: process target return and timesteps
        target_return = torch.as_tensor(target_return, dtype=torch.float32, device=self.device).reshape(1, 1, 1)
        timestep = torch.as_tensor(timestep, dtype=torch.int64, device=self.device).reshape(1, 1)

        time_embeddings = self.embed_timestep(timestep)
        return_embeddings = self.embed_return(target_return) + time_embeddings

        # Step 3: process prompt, and turn it into embeddings
        if self.prompt_cache is not None:
            prompt_embed = self.prompt_cache.detach()
        else:
            task_prompt = TASK_PROMPT.replace('[Task Specification]', str(task_info))
            task_prompt = task_prompt.replace('[Task Input Attributes]', str(sample_info))
            task_prompt_embed = self.llm_vocab(torch.tensor(self.tokenizer(task_prompt).input_ids, dtype=torch.int64, device=self.device))
                    
            tool_embed_pure = self.token_encoder(self.all_tool_tokens, sample_size, fuse_cost=False)
            tool_embed_with_cost = self.token_encoder(self.all_tool_tokens, sample_size, fuse_cost=True)
            prompt_embed = torch.cat([self.instruction_prompt_embed.detach(), self.tool_prompt2_p1_embed.detach(), tool_embed_pure,
                                    self.tool_prompt2_p2_embed.detach(), tool_embed_with_cost, task_prompt_embed], dim=0).unsqueeze(0).to(self.device)
                    
            self.prompt_cache = prompt_embed.detach()

        # Step 4: process state
        state = torch.tensor(state, dtype=torch.int64, device=self.device)
        state_embeddings = self.token_encoder(state, sample_size)
        state_embeddings = state_embeddings.unsqueeze(0) + time_embeddings

        # Step 5: stack return, stage and previous embeddings
        stacked_inputs = torch.cat((return_embeddings, state_embeddings), dim=1)  # mind the order
        stacked_inputs = torch.cat((prev_stacked_inputs, stacked_inputs), dim=1)  # mind the order

        # prepend prompt embeddings
        inputs = torch.cat([prompt_embed, stacked_inputs], dim=1)
        inputs = inputs[:, -self.llm_embed_dim:, :]  # truncate sequence length (should not exceed plm embed size)
        inputs_ln = self.embed_ln(inputs)  # layer normalization

        # 1 if can be attended to, 0 if not
        attention_mask = torch.ones((inputs_ln.shape[0], inputs_ln.shape[1]), dtype=torch.long, device=self.device)

        transformer_outputs = self.llm(
            inputs_embeds=inputs_ln,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = transformer_outputs['last_hidden_state']

        # Step 6: predict action
        logits_used = logits[:, -1:]
        task_type = TaskType.determine_task_type(task_id)
        if mode == TOOL_PREDICTION_MODE:
            action_logits = self.tool_head(logits_used)
            action_logits = action_logits.reshape(-1)
            truncated_action_logits = action_logits[:self.num_tool_tokens]
            if not self.disable_masking:
                mask = create_action_mask(truncated_action_logits, self.generated_tool_tokens, self.device, mode, 
                                        task_type=task_type, cur_plan=state.cpu().numpy())
            else:
                mask = torch.zeros(truncated_action_logits.shape, dtype=torch.float, device=self.device)
                mask[0] = -INF  # skip the first token, which is SOP/SOD token
            action, _ = self._sample(truncated_action_logits, mask)
            token = action + GlobalToolConfig.tool_token_start
            self.generated_tool_tokens.append(token)
        else:
            action_logits = self.dependency_head(logits_used)
            action_logits = action_logits.reshape(-1)
            truncated_action_logits = action_logits[:self.num_dependency_tokens]
            prev_token_is_sod = state[-1].item() == GlobalToolConfig.sod_token
            if not self.disable_masking:
                mask = create_action_mask(truncated_action_logits, self.generated_tool_tokens, self.device, mode, 
                                        prev_token_is_sod=prev_token_is_sod, task_type=task_type)
            else:
                mask = torch.zeros(truncated_action_logits.shape, dtype=torch.float, device=self.device)
                mask[0] = -INF  # skip the first token, which is SOP/SOD token
            action, _ = self._sample(truncated_action_logits, mask)
            token = action + GlobalToolConfig.dependency_token_start

        # compute action embeddings 
        action_tensor = torch.zeros(1, 1, 1, dtype=torch.float32, device=self.device)
        action_tensor[..., 0] = action / GlobalToolConfig.max_num_tokens
        if token < GlobalToolConfig.dependency_token_start:
            action_embeddings = self.embed_tool_action(action_tensor) + time_embeddings
        else:
            action_embeddings = self.embed_dep_action(action_tensor) + time_embeddings
        
        # stack return, state, action, and cache them
        stacked_inputs = torch.cat((return_embeddings, state_embeddings, action_embeddings), dim=1)  # mind the order
        self.stacked_inputs_dq.append(stacked_inputs)

        if return_logits:
            return token, action_logits
        return token
    
    def clear_cache(self):
        self.stacked_inputs_dq.clear()
        self.prompt_cache = None
        self.generated_tool_tokens.clear()
        self.stacked_inputs_dq.append(torch.zeros((1, 0, self.llm_embed_dim), device=self.device))

    def _sample(self, logits, mask):
        logits = logits + mask
        pi = F.softmax(logits, 0).detach().cpu().numpy()
        idx = random.choices(np.arange(pi.size), pi)[0]
        lgprob = np.log(pi[idx])
        return idx, lgprob
