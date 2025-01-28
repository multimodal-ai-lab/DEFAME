import atexit
import time
from multiprocessing import Queue
from queue import Empty
from typing import Callable
from threading import Thread

import torch
from scipy.optimize import anderson

from defame.common import logger
from defame.helpers.parallelization.task import Task
from defame.helpers.parallelization.worker import execute, Worker, Status


class WorkerPool:
    """Manages a set of workers (sub-processes) executing queued tasks"""

    def __init__(self,
                 n_workers: int,
                 target: Callable,
                 device_assignments: list[int] = None,
                 **kwargs):
        self.target = target
        self.kwargs = kwargs
        self.n_workers = n_workers

        if device_assignments is None:
            device_assignments = self._get_default_device_assignments()
        self.device_assignments = device_assignments

        self._scheduled_tasks = Queue()
        self._results = Queue()
        self._errors = Queue()

        self.tasks: dict[str, Task] = dict()  # task_id: task

        self.n_tasks_received = 0

        atexit.register(self.close)

        self.workers: list[Worker] = []

        Thread(target=self.run, daemon=True).start()

    def run(self):
        """Runs in an own thread, supervises workers, processes their messages, and reports logs."""
        self._run_workers()
        while self.is_running():
            self.process_meta_messages()
            self.report_errors()
            time.sleep(0.1)
        self.close()

    def _get_default_device_assignments(self):
        """Distributes workers evenly across available CUDA devices."""
        n_devices = torch.cuda.device_count()
        return [d % n_devices for d in range(self.n_workers)]

    def _run_workers(self):
        for i in range(self.n_workers):
            device = self.device_assignments[i]
            worker = Worker(
                identifier=i,
                target=self.target,
                kwargs=dict(**self.kwargs,
                            input_queue=self._scheduled_tasks,
                            output_queue=self._results,
                            error_queue=self._errors,
                            worker_id=i,
                            device_id=device)
            )
            self.workers.append(worker)
            logger.info(f"Started worker {i} with PID {worker.pid}.")

    def get_worker(self, worker_id: int) -> Worker:
        return self.workers[worker_id]

    def add_task(self, task: Task):
        self._scheduled_tasks.put(task)
        self.tasks[task.id] = task
        self.n_tasks_received += 1

    def get_result(self, timeout=None):
        if not self.is_running():
            raise Empty
        return self._results.get(timeout=timeout)

    def errors(self):
        while not self._errors.empty():
            yield self._errors.get()

    @property
    def n_tasks_remaining(self) -> int:
        return self._scheduled_tasks.qsize()

    @property
    def n_failed_tasks(self) -> int:
        return len([t for t in self.tasks.values() if t.failed])

    def process_meta_messages(self):
        """Iterates over all waiting meta messages (like a worker reporting that it
        started working at task XY) and updates the tasks accordingly."""
        for worker_id, worker in enumerate(self.workers):
            if worker.is_alive():
                for msg in worker.get_meta_messages():
                    assert msg.get("worker_id") in [None, worker_id]
                    task_id = msg.get("task_id")
                    if task_id is not None:
                        task = self.tasks[task_id]
                        task.assign_worker(self.workers[worker_id])
                        task.apply_changes(msg)
                    if msg.get("status") == Status.FAILED:
                        logger.error(msg.get('status_message'))

    def report_errors(self):
        # Forward error logs
        for error in self.errors():
            logger.error(error)

    def close(self):
        if self.is_running():
            for i, worker in enumerate(self.workers):
                if worker.is_alive():
                    worker.terminate()
                    print(f"Worker {i} has been terminated gracefully.")
            print("Worker pool closed.")

    def is_running(self) -> bool:
        """Returns true if at least one worker is alive."""
        return any(worker.is_alive() for worker in self.workers)

    def is_ready(self) -> bool:
        """Returns true if all workers are alive."""
        return (len(self.workers) == self.n_workers and
                all(worker.is_alive() for worker in self.workers))

    def wait_until_ready(self):
        """Sleeps until all workers are alive."""
        while not self.is_ready():
            time.sleep(0.1)


class FactCheckerPool(WorkerPool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, target=execute, **kwargs)
