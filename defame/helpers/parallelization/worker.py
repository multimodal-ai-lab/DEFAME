import traceback
from enum import Enum
from multiprocessing import Queue, Pipe, Process
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Callable

from defame.common import logger, Content, Claim
from defame.fact_checker import FactChecker


class Worker(Process):
    id: int

    def __init__(self, identifier: int, target: Callable, kwargs: dict):
        # task_update_conn_receive, task_update_conn_send = Pipe()
        conn_receive, conn_send = Pipe()
        kwargs = dict(**kwargs, connection=conn_send)

        super().__init__(target=target, kwargs=kwargs)

        self.id = identifier
        # self._task_update_conn: Connection = task_update_conn_receive
        self._connection: Connection = conn_receive
        # self._task_msg_buffer = None

        self.start()

    # def get_next_task_update(self, task_id: str, blocking=False) -> dict:
    #     """Aggregates all waiting messages matching the given task_id into one message (recent
    #     messages update previous messages) and returns it. If the next message does not match the
    #     task_id, an empty dict is returned."""
    #     try:
    #         msg_aggregated = dict()
    #         while self._task_msg_available() or blocking:
    #             msg = self._peek_task_msg(blocking)
    #             if msg["task_id"] == task_id:  # ensure that the message belongs to the requested task
    #                 self._task_msg_buffer = None
    #                 msg_aggregated.update(msg)
    #                 blocking = False  # at least one message was received
    #             else:
    #                 break  # apparently, the task with task_id is already completed
    #         return msg_aggregated
    #
    #     except EOFError:
    #         error_message = f"Worker {self.id} closed the connection unexpectedly - perhaps due to an internal error."
    #         raise RuntimeError(error_message)

    # def _peek_task_msg(self, blocking: bool = False):
    #     if self._task_msg_buffer is None:
    #         if self._task_update_conn.poll() or blocking:
    #             self._task_msg_buffer = self._task_update_conn.recv()
    #     return self._task_msg_buffer

    def task_updates(self) -> dict:
        while self._connection.poll():
            yield self._connection.recv()

    # def _get_from_task_msg_buffer(self):
    #     """Returns the current message in the buffer (if available) and sets the buffer to None."""
    #     msg = self._task_msg_buffer
    #     self._task_msg_buffer = None
    #     return msg

    # def _task_msg_available(self):
    #     return self._task_update_conn.poll() or self._task_msg_buffer is not None

    # def _read_task_update_connection(self):
    #     if self._task_msg_buffer is None:
    #         if self._task_update_conn.poll():
    #             return self._task_update_conn.recv()
    #     else:
    #         return self._get_from_task_msg_buffer()

    def get_messages(self):
        msgs = []
        try:
            while self._connection.poll():
                msgs.append(self._connection.recv())
        except EOFError:
            msgs.append(dict(worker_id=self.id,
                             status=Status.FAILED,
                             status_message=f"Meta connection of worker {self.id} closed unexpectedly."))
        return msgs

    def __getstate__(self):
        return {"id": self.id}


def execute(input_queue: Queue,
            output_queue: Queue,
            error_queue: Queue,
            connection: Connection,
            worker_id: int,
            device_id: int,
            target_dir: str | Path,
            print_log_level: str = "info",
            **kwargs):
    def report(status_message: str, **kwargs):
        task_id = task.id if locals().get("task") else None
        connection.send(dict(worker_id=worker_id,
                             task_id=task_id,
                             status_message=status_message,
                             **kwargs))

    try:
        device = None if device_id is None else f"cuda:{device_id}"

        logger.set_experiment_dir(target_dir)
        logger.set_log_level(print_log_level)
        logger.set_connection(connection)

        # Initialize the fact-checker
        fc = FactChecker(device=device, **kwargs)

    except Exception:
        error_message = f"Worker {worker_id} encountered an error during startup:\n"
        error_message += traceback.format_exc()
        report(error_message, status=Status.FAILED)
        quit(-1)

    # Complete tasks forever
    while True:
        # Fetch the next task and report it
        try:
            task = input_queue.get()
        except Exception:
            error_message = f"Worker {worker_id} encountered an error while reading the task queue:\n"
            error_message += traceback.format_exc()
            report(error_message, status=Status.FAILED)
            continue

        try:
            report("Starting task.", status=Status.RUNNING)
            logger.set_current_fc_id(task.id)

            # Check which type of task this is
            payload = task.payload

            if isinstance(payload, Content):
                # Task is claim extraction
                report("Extracting claims.")
                claims = fc.extract_claims(payload)
                output_queue.put(claims)
                report("Claim extraction from content completed successfully.",
                       status=Status.DONE,
                       result={i: c for i, c in enumerate(claims)})

            elif isinstance(payload, Claim):
                # Task is claim verification
                report("Verifying claim.")
                doc, meta = fc.verify_claim(payload)
                doc.save_to(logger.target_dir)
                output_queue.put((doc, meta))
                report("Claim verification completed successfully.",
                       status=Status.DONE,
                       result=doc.get_result_as_dict())

            else:
                raise ValueError(f"Invalid task type: {type(payload)}")

        except Exception:
            error_message = f"Worker {worker_id} encountered an error while processing task {task.id}:\n"
            error_message += traceback.format_exc()
            report(error_message, status=Status.FAILED)


class Status(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
