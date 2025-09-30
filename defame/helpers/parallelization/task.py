from typing import Optional, Callable

from pydantic import BaseModel

from defame.common import Content, Claim
from defame.helpers.parallelization.worker import Worker
from defame.helpers.common import TaskState


class TaskInfo(BaseModel):
    task_id: str
    state: str
    message: str


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
                 state: TaskState = TaskState.PENDING,
                 status_message: str = "Pending.",
                 callback: Callable = None):
        self.id = str(id)
        self.payload = payload
        self.status = dict(state=state, message=status_message)

        self.worker_id: Optional[int] = None
        self.callback = callback

        self.result = None

    def assign_worker(self, worker: Worker):
        self.worker_id = worker.id
        self.set_status(TaskState.RUNNING, f"Assigned to worker {self.worker_id}.")

    def set_status(self, state: TaskState = None, message: str = None):
        self.status = dict(state=state or self.status["state"],
                           message=message or self.status["message"])

    @property
    def state(self) -> TaskState:
        return self.status["state"]

    @property
    def status_message(self) -> str:
        return self.status["message"]

    @property
    def is_pending(self) -> bool:
        return self.state == TaskState.PENDING

    @property
    def is_running(self) -> bool:
        return self.state == TaskState.RUNNING

    @property
    def is_done(self) -> bool:
        return self.state == TaskState.DONE

    @property
    def failed(self) -> bool:
        return self.state == TaskState.FAILED

    @property
    def terminated(self) -> bool:
        return self.is_done or self.failed

    @property
    def type(self):
        return "claim" if isinstance(self.payload, Claim) else "content"

    def get_info(self) -> TaskInfo:
        return TaskInfo(task_id=self.id,
                        state=self.state.name,
                        message=self.status_message)

    def update(self, updated: "Task"):
        self.result = updated.result
        self.status = updated.status
        self.worker_id = updated.worker_id

    def __getstate__(self):
        """Callbacks can interfere with multithreading/processing."""
        state = self.__dict__.copy()
        state.pop("callback", None)
        return state
