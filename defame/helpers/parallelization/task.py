from typing import Optional, Callable

from defame.common import Content, Claim
from defame.helpers.parallelization.worker import Status, Worker


class Task:
    def __init__(self,
                 payload: Content | Claim,
                 identifier: str | int,
                 status = Status.PENDING,
                 status_message: str = None):
        self.id = str(identifier)
        self.payload = payload
        self.status = status
        self.status_message = status_message

        # self._get_next_task_update: Optional[Callable] = None
        self.worker_id: Optional[int] = None

        self.result: Optional[dict] = None

    def assign_worker(self, worker: Worker):
        self.worker_id = worker.id
        # self._get_next_task_update = worker.get_next_task_update
        self.status = Status.RUNNING

    # async def update(self, blocking=False) -> Optional[dict]:
    #     """Retrieves all waiting status messages for this task and updates the task's
    #     information accordingly. Returns only the changes."""
    #     message = await self._get_next_update(blocking)
    #     if message:
    #         assert message.get("task_id") in [None, self.id]
    #         self.apply_changes(message)
    #         return message

    def apply_changes(self, message: dict):
        if "status" in message:
            self.status = message["status"]
        if "status_message" in message:
            self.status_message = message["status_message"]
        if "result" in message:
            self.result = message["result"]
            if isinstance(self.payload, Content):
                # Register new claims
                claims_dict = message["result"]
                claims = [c for i, c in sorted(claims_dict.items())]
                self.payload.claims = claims
            elif isinstance(self.payload, Claim):
                result = message["result"]
                self.payload.verdict = result["verdict"]
                self.payload.justification = result["justification"]

    # async def _get_next_update(self, blocking: bool = True) -> Optional[dict]:
    #     """Waits for and returns the next update information for the specified task.
    #     If there are multiple updates available already, merges them into a single update."""
    #     if self.terminated or self.is_pending:
    #         return None
    #
    #     try:
    #         message = self._get_next_task_update(task_id=self.id, blocking=blocking)
    #     except RuntimeError as e:
    #         message = dict(
    #             status=Status.FAILED,
    #             message=str(e)
    #         )
    #
    #     return message

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

    def get_summary(self):
        return dict( # task_id=self.id,
                    status=self.status.value,
                    status_message=self.status_message)
