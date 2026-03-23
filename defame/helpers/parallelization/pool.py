import atexit
import time
import traceback
from multiprocessing import Manager, Queue
from queue import Empty
from threading import Thread

import torch

from defame.common import logger
from defame.helpers.parallelization.task import Task
from defame.helpers.parallelization.worker import Worker


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

        self.tasks: dict[str, Task] = dict()  # {task_id: task}
        self._results = Queue()

        # IPC (shared data across workers)
        self.manager = Manager()
        self._scheduled_tasks = self.manager.Queue()
        self._updated_tasks = self.manager.Queue()

        self.n_tasks_received = 0
        self.n_failed_tasks = 0

        self.workers: list[Worker] = []

        # Setup termination handling
        atexit.register(self.stop)
        self.terminating = False

        self._start_workers()
        Thread(target=self.run, daemon=True).start()

    def run(self):
        """Instantiates a multiprocessing manager, then spawns a pool of workers, overseen
        by the manager."""
        while self.is_running() and not self.terminating:
            try:
                self.process_results()
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

    def _start_workers(self):
        for i in range(self.n_workers):
            device = self.device_assignments[i]
            worker = Worker(
                identifier=i,
                kwargs=dict(**self.kwargs,
                            input_queue=self._scheduled_tasks,
                            output_queue=self._updated_tasks,
                            device_id=device)
            )
            self.workers.append(worker)
            logger.debug(f"Started worker {i} with PID {worker.pid}.")

    def get_worker(self, worker_id: int) -> Worker:
        return self.workers[worker_id]

    def add_task(self, task: Task):
        self.tasks[task.id] = task
        self._scheduled_tasks.put(task)
        self.n_tasks_received += 1

    def get_result(self, timeout=None):
        if not self.is_running():
            raise Empty
        return self._results.get(timeout=timeout)

    @property
    def n_tasks_remaining(self) -> int:
        return self._scheduled_tasks.qsize()

    def process_results(self):
        """Reads finished tasks from the output queue and calls their callbacks.
        Propagates errors to the main thread."""
        while not self._updated_tasks.empty():
            task_updated = self._updated_tasks.get()
            task = self.tasks[task_updated.id]
            task.update(task_updated)

            if task.is_done:
                if task.callback:
                    task.callback(task)
                self._results.put(task.result)
            elif task.failed:
                logger.error(f"Task {task.id} failed: {task.status_message}")
                self.n_failed_tasks += 1

    def stop(self):
        self.terminating = True
        self.terminate_all_workers()
        self.manager.shutdown()

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
