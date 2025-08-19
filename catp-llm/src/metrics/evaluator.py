import numpy as np
from transformers import AutoModel, AutoFeatureExtractor
import torch
import torch.nn.functional as F
from evaluate import load

from src.config import EVALUATOR_DEVICE_LIST, GlobalPathConfig, GlobalMetricsConfig, log
from src.utils import get_available_device

_evaluator = None


class Evaluator:
    """
    An Evaluator class that initializes and holds the models needed for
    text and image evaluation (BERTScore, ViT-based similarity, etc.).
    """

    _bert_score_evaluator = None
    _vit_score_evaluator = None
    _vit_extractor = None

    def __init__(self):
        log.info("Initializing Evaluator with BERTScore and ViT models...")

        self._bert_score_evaluator = load("bertscore", cache_dir=GlobalPathConfig.hf_cache)

        self._vit_score_evaluator = AutoModel.from_pretrained("nateraw/vit-base-beans", cache_dir=GlobalPathConfig.hf_cache)
        self._vit_extractor = AutoFeatureExtractor.from_pretrained("nateraw/vit-base-beans", cache_dir=GlobalPathConfig.hf_cache)

        device = get_available_device(EVALUATOR_DEVICE_LIST)

        self._vit_score_evaluator.to(device)
        self._vit_score_evaluator.eval()

        log.info("Evaluator loaded successfully.")

    def calculate_image_similarity(self, image1, image2, *, version="F-norm"):
        """
        Calculate similarity between two images using the specified method.

        Args:
            image1: A batch of images (e.g., a list of PIL images or tensors).
            image2: Another batch of images, same size/batch length as image1.
            version: 'cosine' or 'F-norm' to compute similarity or distance-based measure.

        Returns:
            A single float value indicating the similarity (or distance) measure.
            Note that 'F-norm' is effectively a distance, not a similarity.
        """
        # image1 and image2 are assumed to have shape [batch_size, ...]
        # If you have a single image, batch_size = 1
        
        # todo: check this method

        batch_size = len(image1)
        model = self._vit_score_evaluator
        model_device = model.device

        # Convert images into model inputs
        inputs_image1 = self._vit_extractor(images=image1, return_tensors="pt").to(model_device)
        inputs_image2 = self._vit_extractor(images=image2, return_tensors="pt").to(model_device)

        if version == "cosine":
            # Use cosine similarity on pooler_output
            with torch.no_grad():
                feat_image1 = model(**inputs_image1).pooler_output
                feat_image2 = model(**inputs_image2).pooler_output
            similarity = F.cosine_similarity(feat_image1, feat_image2, dim=-1)
            # If batch_size > 1, take mean of all pairwise similarities
            similarity = similarity.mean().item()

        elif version == "F-norm":
            # Compute the last hidden state (or other embedding) and measure F-norm (distance)
            with torch.no_grad():
                emb_image1 = model(inputs_image1.pixel_values)[0].cpu().numpy()  # [batch, seq_len, hidden_dim]
                emb_image2 = model(inputs_image2.pixel_values)[0].cpu().numpy()

            # Calculate F-norm distances for each item in the batch
            distances = []
            for i in range(batch_size):
                # If needed, squeeze to remove extra dims; depends on model's actual output shape
                dist = np.linalg.norm(emb_image1[i] - emb_image2[i], ord='fro')
                distances.append(dist)

            # 'similarity' here is actually the average distance
            similarity = float(np.mean(distances))

        else:
            raise NotImplementedError(f"Image similarity using {version} is not implemented.")
        similarity /= 100  # normalized into [0, 1]
        return similarity

    def calculate_bert_score(self, text1, text2):
        """
        Compute the BERTScore F1 between text1 and text2 using microsoft/deberta-xlarge-mnli.

        Args:
            text1: The predicted text string.
            text2: The reference text string.

        Returns:
            A single float (F1 measure). Typically in the range [0, 1].
        """
        if isinstance(text1, str):
            text1, text2 = [text1], [text2]
        assert isinstance(text1, list) and isinstance(text2, list)
        results = self._bert_score_evaluator.compute(
            predictions=text1,
            references=text2,
            model_type="microsoft/deberta-xlarge-mnli",
            device=get_available_device(EVALUATOR_DEVICE_LIST),
        )
        # 'results["f1"]' is typically a list of floats if there are multiple samples
        # Here, we have a single text pair, so we take results["f1"][0] or .item()
        score = results["f1"][0]
        return float(score)


def get_vit_score(image1, image2):
    """
    A global helper function to compute image similarity using the default Evaluator.
    """
    global _evaluator
    if _evaluator is None:
        _evaluator = Evaluator()
    return _evaluator.calculate_image_similarity(image1, image2)


def get_bert_score(text1, text2):
    """
    A global helper function to compute BERTScore between two text strings.
    """
    global _evaluator
    if _evaluator is None:
        _evaluator = Evaluator()
    return _evaluator.calculate_bert_score(text1, text2)


def calculate_task_score(prediction, ground_truth, sequential=True):
    scores = []
    if sequential:
        if 'image' in ground_truth:
            vit_score = get_vit_score(prediction['image'], ground_truth['image'])
            scores.append(vit_score)
        if 'text' in ground_truth:
            for key in prediction.keys():
                if 'text' in key:
                    bert_score = get_bert_score(prediction[key], ground_truth['text'])
                    scores.append(bert_score)
                    break
    else:
        if 'image' in ground_truth:
            vit_score = get_vit_score(prediction['image'], ground_truth['image'])
            scores.append(vit_score)
        if 'text-object' in ground_truth:
            if 'text-object' in prediction:
                bert_score = get_bert_score(prediction['text-object'], ground_truth['text-object'])
            else:
                bert_score = 0
            scores.append(bert_score)
        if 'text-caption' in ground_truth:
            if 'text-caption' in prediction:
                bert_score = get_bert_score(prediction['text-caption'], ground_truth['text-caption'])
            else:
                bert_score = 0
            scores.append(bert_score)
        if 'text-label' in ground_truth:
            if 'text-label' in prediction:
                bert_score = get_bert_score(prediction['text-label'], ground_truth['text-label'])
            else:
                bert_score = 0
            scores.append(bert_score)
    task_score = np.mean(scores)
    return task_score


def calculate_qop(
        avg_score,
        cost_price,
        alpha=GlobalMetricsConfig.ALPHA,
        min_score=GlobalMetricsConfig.MIN_SCORE,
        max_score=GlobalMetricsConfig.MAX_SCORE,
        min_cost=GlobalMetricsConfig.MIN_COST,
        max_cost=GlobalMetricsConfig.MAX_COST,
):
    """
    Calculate QOP (Quality over Price) metric with configurable alpha,
    based on normalized score and cost.

    Args:
        avg_score: Average score or quality metric.
        cost_price: The total or average cost to achieve that score.
        alpha: The weight factor balancing score and cost in QOP.
        min_score: Bounds used to normalize the score.
        max_score: Bounds used to normalize the score.
        min_cost: Bounds used to normalize the cost.
        max_cost: Bounds used to normalize the cost.

    Returns:
        A float value representing the QOP metric.
    """
    norm_score = (avg_score - min_score) / (max_score - min_score)
    norm_cost = (cost_price - min_cost) / (max_cost - min_cost)
    qop = alpha * norm_score - (1 - alpha) * norm_cost
    return qop
