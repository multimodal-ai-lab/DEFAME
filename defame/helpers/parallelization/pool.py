import atexit
import time
import traceback
from multiprocessing import Queue
from queue import Empty
from threading import Thread

import torch

from defame.common import logger
from defame.helpers.common import Status
from defame.helpers.parallelization.task import Task
from defame.helpers.parallelization.worker import FactCheckerWorker


class Pool:
    """Manages a set of workers (sub-processes) executing queued tasks."""

    def __init__(self,
                 n_workers: int,
                 device_assignments: list[int] = None,
                 **kwargs):
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

        self.workers: list[FactCheckerWorker] = []

        Thread(target=self.run, daemon=True).start()

        # Setup termination handling
        atexit.register(self.stop)
        self.terminating = False

    def run(self):
        """Runs in an own thread, supervises workers, processes their messages, and reports logs."""
        self._run_workers()
        self.wait_until_ready()
        while self.is_running() and not self.terminating:
            try:
                self.process_messages()
                self.report_errors()
                time.sleep(0.1)
            except Exception:
                logger.error("Error encountered in worker pool main thread:")
                logger.error(traceback.format_exc())

    def _get_default_device_assignments(self):
        """Distributes workers evenly across available CUDA devices."""
        n_devices = torch.cuda.device_count()
        if n_devices == 0:
            return [None] * self.n_workers
        else:
            return [d % n_devices for d in range(self.n_workers)]

    def _run_workers(self):
        for i in range(self.n_workers):
            device = self.device_assignments[i]
            worker = FactCheckerWorker(
                identifier=i,
                kwargs=dict(**self.kwargs,
                            input_queue=self._scheduled_tasks,
                            output_queue=self._results,
                            device_id=device)
            )
            self.workers.append(worker)
            logger.debug(f"Started worker {i} with PID {worker.pid}.")

    def get_worker(self, worker_id: int) -> FactCheckerWorker:
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

    def process_messages(self):
        """Iterates over all waiting meta messages (like a worker reporting that it
        started working at task XY) and updates the tasks accordingly."""
        for worker_id, worker in enumerate(self.workers):
            if worker.is_alive():
                for msg in worker.get_messages():
                    assert msg.get("worker_id") in [None, worker_id]
                    status = msg.get("status")
                    if status == Status.FAILED:
                        logger.error(msg.get('status_message'))
                        # TODO: Move entire error message processing here
                    task_id = msg.get("task_id")
                    if task_id is not None:
                        task = self.tasks[task_id]
                        task.assign_worker(self.workers[worker_id])
                        task.update(msg)

    def report_errors(self):
        # Forward error logs
        for error in self.errors():
            logger.error(error)

    def stop(self):
        self.terminating = True
        self.terminate_all_workers()

    def terminate_all_workers(self):
        if self.is_running():
            for i, worker in enumerate(self.workers):
                if worker.is_alive():
                    worker.terminate()
                    print(f"Terminating worker {i}...")
        else:
            print("All workers terminated already.")

    def is_running(self) -> bool:
        """Returns true if at least one worker is alive."""
        return any(worker.is_alive() for worker in self.workers)

    def is_ready(self) -> bool:
        """Returns true if all workers are initialized and alive."""
        return (len(self.workers) == self.n_workers and
                all(worker.is_alive() for worker in self.workers))

    def wait_until_ready(self):
        """Sleeps until all workers are alive."""
        while not self.is_ready():
            time.sleep(0.1)
