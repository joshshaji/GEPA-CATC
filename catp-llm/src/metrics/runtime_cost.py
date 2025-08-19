import os
import time
from threading import Thread, Event

import psutil


class CPUMemoryMonitor(Thread):
    """
    A background thread that periodically checks the RSS memory usage (in bytes) of a specific process
    and records the maximum usage observed during its lifetime.
    """

    pid: int
    interval: float
    max_memory: int
    _stop_event: Event

    def __init__(self, *, pid: int = None, interval: float = 0.1) -> None:
        """
        Args:
            pid: The target process ID to monitor. If None, defaults to the current process (os.getpid()).
            interval: The sampling interval in seconds (default: 0.1).
        """
        super().__init__()
        self.pid = pid or os.getpid()
        self.interval = interval
        self.max_memory = 0
        self._stop_event = Event()

    def run(self) -> None:
        """
        Continuously measure the RSS memory usage of the specified process until stopped.
        """
        process = psutil.Process(self.pid)
        while not self._stop_event.is_set():
            try:
                mem_info = process.memory_info()
                # RSS (Resident Set Size) in bytes
                rss = mem_info.rss
                # Update the max observed memory usage
                if rss > self.max_memory:
                    self.max_memory = rss
            except psutil.NoSuchProcess:
                # If the process no longer exists, exit the loop
                break
            time.sleep(self.interval)

    def stop(self) -> None:
        """
        Signal the monitoring thread to stop and block until it terminates.
        """
        self._stop_event.set()
        self.join()

    def get_max_cpu_memory_allocated(self, unit: str = 'MB') -> float:
        """
        Retrieve the maximum RSS memory usage observed so far.

        Args:
            unit: The unit for the returned memory size. Defaults to 'MB'.

        Returns:
            The maximum RSS memory usage in the specified unit.
        """
        if unit == 'MB':
            return self.max_memory / (1024 ** 2)
        elif unit == 'KB':
            return self.max_memory / 1024
        elif unit == 'GB':
            return self.max_memory / (1024 ** 3)
        else:
            # If the unit is unrecognized, just return bytes
            return float(self.max_memory)
