import os
import logging
import time
from dotenv import load_dotenv

# Configure DSPy caching before importing dspy to avoid disk permission issues
os.environ.setdefault("DSPY_DISABLE_DISK_CACHE", "1")
os.environ.setdefault("DSPY_CACHE_DIR", os.path.abspath(".dspy_cache"))

import dspy
from catp_gepa.run_state import RunState
from catp_gepa.metric import metric_qop, metric_qop_feedback, vanila_gepa_metric, make_logged_metric

from catp_gepa.config import load_config, get_dataset
from catp_gepa.dataset import build_valid_plans_examples
from catp_gepa.modules import PlanGenerator


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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    logger = logging.getLogger(__name__)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    base_lm = dspy.LM("openai/gpt-4.1-nano", temperature=0, api_key=openai_api_key)
    reflector_lm = dspy.LM("openai/gpt-5", temperature=1.0, api_key=openai_api_key, max_tokens=128000)
    dspy.configure(lm=base_lm)
    # dspy.configure_cache(
    #     enable_disk_cache=False,
    #     enable_memory_cache=True,
    # )

    train_set, val_set, test_set = init_dataset()
    cfg = load_config()
    train_small = train_set[: cfg.training_size]
    val_small = val_set[: max(1, min(cfg.test_size, len(val_set)))]

    program = PlanGenerator()

    # Initialize run-state and graceful handlers
    run_state = RunState(meta={"entry": "optimize_and_evaluate"})
    run_state.install_signal_handlers()
    # Use a unique log directory per run
    run_log_dir = os.path.join("gepa_logs", f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_log_dir, exist_ok=True)

    pre_opt_eval = dspy.Evaluate(devset=val_small, metric=make_logged_metric(metric_qop, run_state, stage_label="pre_eval"))
    try:
        pre_score = pre_opt_eval(program)
    except KeyboardInterrupt:
        path = run_state.dump_once(run_state.default_path("gepa_logs/run_state_keyboard_interrupt"))
        if path:
            logger.error("Run state saved on KeyboardInterrupt (pre_eval) to %s", path)
        raise
    except Exception:
        path = run_state.dump_once(run_state.default_path("gepa_logs/run_state_error_pre_eval"))
        if path:
            logger.exception("Exception during pre-optimization eval. State saved to %s", path)
        else:
            logger.exception("Exception during pre-optimization eval (state already saved)")
        raise
    logger.info("Pre-optimization score: %s", pre_score)
    
    if cfg.use_vanila_gepa:
        optimizer = dspy.GEPA(
            metric=make_logged_metric(vanila_gepa_metric, run_state, stage_label="gepa"),
            auto="heavy",
            reflection_lm=reflector_lm,
            add_format_failure_as_feedback=True,
            log_dir=run_log_dir,
        )
    else:
        optimizer = dspy.GEPA(
            metric=make_logged_metric(metric_qop_feedback, run_state, stage_label="gepa"),
            auto="heavy",
            reflection_lm=reflector_lm,
            add_format_failure_as_feedback=True,
            log_dir=run_log_dir,
        )

    # Create a single optimizer object with the chosen metric (above)
    # and run inside a try/except to dump state on errors
    try:
        optimized_program = optimizer.compile(program, trainset=train_small, valset=val_small)
    except KeyboardInterrupt:
        path = run_state.dump_once(run_state.default_path("gepa_logs/run_state_keyboard_interrupt"))
        if path:
            logger.error("Run state saved on KeyboardInterrupt to %s", path)
        raise
    except Exception as e:
        path = run_state.dump_once(run_state.default_path("gepa_logs/run_state_error"))
        if path:
            logger.exception("Exception during GEPA run. State saved to %s", path)
        else:
            logger.exception("Exception during GEPA run (state already saved)")
        raise

    evaluator = dspy.Evaluate(devset=test_set, metric=make_logged_metric(metric_qop, run_state, stage_label="post_eval"))
    try:
        post_score = evaluator(optimized_program)
    except KeyboardInterrupt:
        path = run_state.dump_once(run_state.default_path("gepa_logs/run_state_keyboard_interrupt"))
        if path:
            logger.error("Run state saved on KeyboardInterrupt (post_eval) to %s", path)
        raise
    except Exception:
        path = run_state.dump_once(run_state.default_path("gepa_logs/run_state_error_post_eval"))
        if path:
            logger.exception("Exception during post-optimization eval. State saved to %s", path)
        else:
            logger.exception("Exception during post-optimization eval (state already saved)")
        raise
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
    # Save final run state on success as well (ensures a visible message)
    run_state.dump_once(run_state.default_path("gepa_logs/run_state_success"))
    print(results)
    return results


if __name__ == "__main__":
    results = optimize_and_evaluate()
    print({k: v for k, v in results.items() if k != "optimized_program"})
