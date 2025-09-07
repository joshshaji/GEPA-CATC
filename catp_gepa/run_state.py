import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _safe_json(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except Exception:
        try:
            # Convert common non-serializable objects
            if hasattr(obj, "items") and callable(getattr(obj, "items")):
                return {k: _safe_json(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [_safe_json(v) for v in obj]
            # Fall back to string
            return str(obj)
        except Exception:
            return str(obj)


@dataclass
class PredictorCall:
    when: str
    predictor_id: Optional[str] = None
    predictor_name: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    raw_completion_on_failure: Optional[str] = None


@dataclass
class RunEvent:
    when: str
    stage: str
    task_and_sample_id: str
    example_fields: Dict[str, Any] = field(default_factory=dict)
    prediction_summary: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
    feedback: Optional[str] = None
    predictor_calls: List[PredictorCall] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class RunState:
    started_at: str = field(default_factory=_now_iso)
    pid: int = field(default_factory=os.getpid)
    cwd: str = field(default_factory=os.getcwd)
    meta: Dict[str, Any] = field(default_factory=dict)
    events: List[RunEvent] = field(default_factory=list)

    # Internal flag to prevent multiple dumps
    _dumped: bool = field(default=False, init=False, repr=False)
    _dump_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def add_event(self, event: RunEvent) -> None:
        self.events.append(event)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict for the run state.

        Avoids dataclasses.asdict(self) to prevent deep-copying non-serializable
        internals like threading.Lock. Only include public fields.
        """
        # Convert events safely
        safe_events: List[Dict[str, Any]] = []
        for ev in self.events:
            try:
                ev_dict = asdict(ev)
            except Exception:
                # Fallback: shallow manual conversion if asdict fails unexpectedly
                ev_dict = {
                    "when": getattr(ev, "when", None),
                    "stage": getattr(ev, "stage", None),
                    "task_and_sample_id": getattr(ev, "task_and_sample_id", None),
                    "example_fields": getattr(ev, "example_fields", {}),
                    "prediction_summary": getattr(ev, "prediction_summary", {}),
                    "score": getattr(ev, "score", None),
                    "feedback": getattr(ev, "feedback", None),
                    "predictor_calls": [asdict(pc) for pc in getattr(ev, "predictor_calls", []) if pc],
                    "error": getattr(ev, "error", None),
                }
            safe_events.append(_safe_json(ev_dict))

        data = {
            "started_at": self.started_at,
            "pid": self.pid,
            "cwd": self.cwd,
            "meta": _safe_json(self.meta),
            "events": safe_events,
        }
        return data

    def dump(self, path: str) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    def dump_once(self, path: str) -> Optional[str]:
        """Write state to `path` only once.

        Marks the state as dumped only after a successful write, so that a
        failed attempt doesn't permanently suppress future dumps.
        """
        with self._dump_lock:
            if self._dumped:
                return None
            # Attempt dump while holding the lock; set flag after success
            saved_path = self.dump(path)
            self._dumped = True
        try:
            # Always emit a simple stdout line when a dump occurs
            sys.stdout.write(f"Run state saved to {saved_path}\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"Failed to write to stdout: {e}\n")
            sys.stderr.flush()
        return saved_path

    def default_path(self, prefix: str = "gepa_logs/run_state") -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{ts}.json"

    def install_signal_handlers(self, out_prefix: str = "gepa_logs/run_state_interrupt"):
        """Install SIGINT and SIGTERM handlers that write state and exit."""
        original_int = signal.getsignal(signal.SIGINT)
        original_term = signal.getsignal(signal.SIGTERM)

        def _handler(signum, frame):  # noqa: ARG001
            try:
                path = self.default_path(out_prefix)
                self.dump_once(path)
                sys.stderr.write(f"\n[RunState] Saved state on signal {signum} to {path}\n")
                sys.stderr.flush()
            finally:
                # Restore original and re-raise default behavior
                signal.signal(signal.SIGINT, original_int)
                signal.signal(signal.SIGTERM, original_term)
                if signum == signal.SIGINT:
                    raise KeyboardInterrupt()
                else:
                    os._exit(143)  # 128 + SIGTERM

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)


def summarize_prediction(prediction: Any) -> Dict[str, Any]:
    """Extract a minimal, serializable view of a DSPy Prediction-like object."""
    out: Dict[str, Any] = {}
    # Common: plan_json for this repo
    try:
        if isinstance(prediction, dict):
            if "plan_json" in prediction:
                out["plan_json"] = prediction["plan_json"]
        else:
            pj = getattr(prediction, "plan_json", None)
            if pj is not None:
                out["plan_json"] = pj
    except Exception:
        pass

    # Fallback string if empty
    if not out:
        try:
            out["repr"] = str(prediction)
        except Exception:
            out["repr"] = "<unserializable>"
    return _safe_json(out)


def extract_example_fields(example: Any) -> Dict[str, Any]:
    """Extract useful fields from a dspy.Example or dict-like."""
    keys = [
        "task_query",
        "tool_catalog_json_with_description",
        "input_attributes_json",
        "gold_plan_json",
    ]
    data: Dict[str, Any] = {}
    for k in keys:
        try:
            if isinstance(example, dict):
                if k in example:
                    data[k] = example[k]
            else:
                if hasattr(example, k):
                    data[k] = getattr(example, k)
                elif hasattr(example, "__getitem__"):
                    try:
                        data[k] = example[k]
                    except Exception:
                        pass
        except Exception:
            # ignore extraction errors
            pass
    return _safe_json(data)
