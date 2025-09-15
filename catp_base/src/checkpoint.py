import json
import os
import time
from pathlib import Path
from typing import Any, Dict


class JsonCheckpointer:
    def __init__(self, path: str | os.PathLike[str], autosave_steps: int = 0):
        self.path = Path(path)
        self.autosave_steps = max(0, int(autosave_steps))
        self.state: Dict[str, Any] = {}
        self._steps: int = 0

    def load_if_exists(self) -> bool:
        if not self.path.exists():
            return False
        try:
            with self.path.open("r", encoding="utf-8") as f:
                self.state = json.load(f)
            return True
        except Exception:
            return False

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
        tmp_path.replace(self.path)

    def tick(self) -> None:
        if isinstance(self.state.get("started_at"), (int, float)):
            self.state["elapsed_sec"] = float(time.time() - float(self.state["started_at"]))

    def step(self) -> None:
        self._steps += 1
        if self.autosave_steps and (self._steps % self.autosave_steps == 0):
            self.tick()
            self.save()


