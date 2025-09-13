import os
import logging
import time
from dotenv import load_dotenv

# Configure DSPy caching before importing dspy to avoid disk permission issues
os.environ.setdefault("DSPY_DISABLE_DISK_CACHE", "1")
os.environ.setdefault("DSPY_CACHE_DIR", os.path.abspath(".dspy_cache"))

import dspy
from catp_gepa.run_state import RunState
from catp_gepa.metric import metric_dag_loss, metric_qop, metric_qop_feedback, vanila_gepa_metric, make_logged_metric

from catp_gepa.config import Config, get_metric, load_config, get_dataset
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
    cfg = load_config()
    
    # base_lm = dspy.LM("together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", temperature=0, api_key=together_api_key, api_base="https://api.together.xyz/v1")
    # reflector_lm = dspy.LM("anthropic/claude-sonnet-4-20250514", temperature=1.0, api_key=openai_api_key, max_tokens=64000)
    base_lm, reflector_lm = load_llms(cfg)
    dspy.configure(lm=base_lm)
    # dspy.configure_cache(
    #     enable_disk_cache=False,
    #     enable_memory_cache=True,
    # )

    train_set, val_set, test_set = init_dataset()
    train_small = train_set[: cfg.training_size]
    metric_from_config = get_metric(cfg.metric)
    val_small = val_set[: max(1, min(cfg.test_size, len(val_set)))]

    program = PlanGenerator()

    # Initialize run-state and graceful handlers
    run_state = RunState(meta={"entry": "optimize_and_evaluate"})
    run_state.install_signal_handlers()
    # Use a unique log directory per run
    run_log_dir = os.path.join("gepa_logs", f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_log_dir, exist_ok=True)

    pre_opt_eval = dspy.Evaluate(devset=val_small, metric=make_logged_metric(metric_from_config, run_state, stage_label="pre_eval"))
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
    
    optimizer = dspy.GEPA(
        metric=make_logged_metric(metric_from_config, run_state, stage_label="gepa"),
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

    evaluator = dspy.Evaluate(devset=test_set, metric=make_logged_metric(metric_from_config, run_state, stage_label="post_eval"))
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


def load_llms(cfg: Config):
    base_llm: str = cfg.llm.get("base","")
    reflective_llm: str = cfg.llm.get("reflective")
    api_key, api_base = load_api_key_and_base(base_llm)
    api_key_reflective, api_base_reflective = load_api_key_and_base(reflective_llm)
    
    params_base = {
        "api_key": api_key,
    }
    if api_base:
        params_base["api_base"] = api_base
        
    params_reflective = {
        "api_key": api_key_reflective,
    }
    if api_base:
        params_base["api_base"] = api_base_reflective
    
    base_lm = dspy.LM(base_llm, temperature=0, **params_base)
    reflector_lm = dspy.LM(reflective_llm, temperature=1.0, **params_reflective)
    
    return base_lm, reflector_lm
    
def load_api_key_and_base(llm_model: str):
    api_key, api_base = "", ""
    if llm_model.startswith("together_ai"):
        api_key = os.getenv("TOGETHER_API_KEY")
        api_base = "https://api.together.xyz/v1"
    if llm_model.startswith("openai"):
        api_key = os.getenv("OPENAI_API_KEY")
    if llm_model.startswith("anthropic"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
    
    return api_key, api_base
    

if __name__ == "__main__":
    results = optimize_and_evaluate()
    print({k: v for k, v in results.items() if k != "optimized_program"})
