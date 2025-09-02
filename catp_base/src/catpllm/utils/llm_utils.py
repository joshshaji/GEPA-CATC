from collections import namedtuple
from transformers import LlamaConfig, LlamaTokenizer, OPTConfig, Qwen2Config, Qwen2Tokenizer, GPT2Tokenizer
from src.catpllm.model.llm import LlamaModel, OPTModel, Qwen2Model

                        
LLMClass = namedtuple("LLMClass", ('config', 'tokenizer', 'model',))


_LLM_CLASSES = {
    "llama2": LLMClass(**{
        "config": LlamaConfig,
        "tokenizer": LlamaTokenizer,
        "model": LlamaModel
    }),
    "opt": LLMClass(**{
        "config": OPTConfig,
        "tokenizer": GPT2Tokenizer,
        "model": OPTModel
    }),
    'qwen2': LLMClass(**{
        "config": Qwen2Config,
        "tokenizer": Qwen2Tokenizer,
        "model": Qwen2Model
    })
}


def get_model_class(llm_type: str):
    if 'llama2' in llm_type or 'tinyllama' in llm_type:
        return _LLM_CLASSES['llama2']
    if 'opt' in llm_type:
        return _LLM_CLASSES['opt']
    if 'gpt2' in llm_type:
        return _LLM_CLASSES['gpt2']
    return None


def load_llm(llm_name, llm_path, specials_to_add=None,  **kwargs):
    r"""A llm loader using a global config.
    It will load the model, tokenizer, and config simulatenously.

    Args:
        llm_name: name of the llm
        llm_path: path of the llm

    Returns:
        :obj:`PreTrainedModel`: The pretrained llm model.
        :obj:`tokenizer`: The llm tokenizer.
        :obj:`llm_config`: The config of the pretrained llm model.
    """
    if "llama" in llm_name:
        specials_to_add = ["<pad>"]
        
    llm_class = get_model_class(llm_type=llm_name)
    llm_config = llm_class.config.from_pretrained(llm_path)
    
    llm = llm_class.model.from_pretrained(llm_path, config=llm_config)
    tokenizer = llm_class.tokenizer.from_pretrained(llm_path) 
    llm, tokenizer = add_special_tokens(
        llm, tokenizer, specials_to_add=specials_to_add
    )
    return llm, tokenizer, llm_config


def add_special_tokens(llm, tokenizer, specials_to_add):
    r"""add the special_tokens to tokenizer if the special token
    is not in the tokenizer.

    Args:
        llm (:obj:`PreTrainedModel`): The pretrained llm to resize embedding
                after adding special tokens.
        tokenizer (:obj:`PreTrainedTokenizer`): The pretrained tokenizer to add special tokens.
        specials_to_add: (:obj:`List[str]`, optional): The special tokens to be added. Defaults to pad token.

    Returns:
        The resized model, The tokenizer with the added special tokens.

    """
    if specials_to_add is None:
        return llm, tokenizer
    for token in specials_to_add:
        if "pad" in token.lower():
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": token})
                llm.resize_token_embeddings(len(tokenizer))
                # print("pad token is None, set to id {}".format(tokenizer.pad_token_id))
    return llm, tokenizer