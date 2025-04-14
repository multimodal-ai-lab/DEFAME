from typing import Optional, Callable

from pydantic import BaseModel

from defame.common import Content, Claim
from defame.helpers.parallelization.worker import Worker
from defame.helpers.common import Status


class TaskInfo(BaseModel):
    task_id: str
    status: str
    status_message: str


class Task:
    """A unit of work to complete. Can be queued into worker pools
    and transferred to different processes.

        @param payload:
        @param id:
        @param status:
        @param status_message:
        @param callback: Function to be called after completion. Takes
            the completed task itself as an argument.
    """

    def __init__(self,
                 payload: Content | Claim,
                 id: str | int,
                 status: Status = Status.PENDING,
                 status_message: str = "Pending.",
                 callback: Callable = None):
        self.id = str(id)
        self.payload = payload
        self.status = status
        self.status_message = status_message

        self.worker_id: Optional[int] = None
        self.callback = callback

        self.result = None

    def assign_worker(self, worker: Worker):
        self.worker_id = worker.id
        self.status = Status.RUNNING

    def update(self, message: dict):
        """Applies the values from the message to the task."""
        if "status" in message:
            self.status = message["status"]
        if "status_message" in message:
            self.status_message = message["status_message"]
        if "result" in message:
            self.result = message["result"]
            if self.callback:
                self.callback(self)

    @property
    def is_pending(self) -> bool:
        return self.status == Status.PENDING

    @property
    def is_running(self) -> bool:
        return self.status == Status.RUNNING

    @property
    def is_done(self) -> bool:
        return self.status == Status.DONE

    @property
    def failed(self) -> bool:
        return self.status == Status.FAILED

    @property
    def terminated(self) -> bool:
        return self.is_done or self.failed

    @property
    def type(self):
        return "claim" if isinstance(self.payload, Claim) else "content"

    def get_info(self) -> TaskInfo:
        return TaskInfo(task_id=self.id,
                        status=self.status.name,
                        status_message=self.status_message)

    def __getstate__(self):
        """Callbacks can interfere with multithreading/processing."""
        state = self.__dict__.copy()
        del state["callback"]
        return state
