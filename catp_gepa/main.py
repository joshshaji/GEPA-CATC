import json
import os
import logging
import time
from dotenv import load_dotenv

# Configure DSPy caching before importing dspy to avoid disk permission issues
os.environ.setdefault("DSPY_DISABLE_DISK_CACHE", "1")
os.environ.setdefault("DSPY_CACHE_DIR", os.path.abspath(".dspy_cache"))

import dspy

from catp_gepa.config import load_config, get_dataset
from catp_gepa.dataset import build_valid_plans_examples
from catp_gepa.modules import PlanGenerator
from catp_gepa.metric import metric_qop, metric_qop_feedback

load_dotenv()


def init_dataset():
    cfg = load_config()
    dataset = get_dataset(cfg)
    train_set, val_set, test_set = build_valid_plans_examples(
        dataset,
        train_size=cfg.training_size,
        test_size=cfg.test_size,
        seed=0,
    )
    return train_set, val_set, test_set


def optimize_and_evaluate():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    base_lm = dspy.LM("openai/gpt-4.1-nano", temperature=0, api_key=openai_api_key)
    reflector_lm = dspy.LM("openai/gpt-5", temperature=1.0, api_key=openai_api_key, max_tokens=128000)
    dspy.configure(lm=base_lm)

    train_set, val_set, test_set = init_dataset()
    cfg = load_config()
    train_small = train_set[: cfg.training_size]
    val_small = val_set[: max(1, min(cfg.test_size, len(val_set)))]

    program = PlanGenerator()

    pre_opt_eval = dspy.Evaluate(devset=val_small, metric=metric_qop)
    pre_score = pre_opt_eval(program)
    logger.info("Pre-optimization score: %s", pre_score)

    optimizer = dspy.GEPA(
        metric=metric_qop_feedback,
        auto="heavy",
        reflection_lm=reflector_lm,
        add_format_failure_as_feedback=True,
        log_dir="gepa_logs",
    )
    optimized_program = optimizer.compile(program, trainset=train_small, valset=val_small)

    evaluator = dspy.Evaluate(devset=test_set, metric=metric_qop)
    post_score = evaluator(optimized_program)
    logger.info("Post-optimization score: %s", post_score)


    # save the optimized program to a file
    with open(f"optimized_program_{time.strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        f.write(optimized_program.predict.signature.instructions)

    results = {
        "pre_opt_score": pre_score,
        "post_opt_score": post_score,
        "detailed_results": getattr(optimized_program, "detailed_results", None),
        "optimized_program": optimized_program,
    }
    print(results)
    return results


if __name__ == "__main__":
    results = optimize_and_evaluate()
    print({k: v for k, v in results.items() if k != "optimized_program"})
