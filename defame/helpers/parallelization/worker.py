import traceback
from multiprocessing import Process
from multiprocessing.managers import SyncManager
from pathlib import Path
from queue import Empty
from time import sleep

from defame.common import logger, Content, Claim
from defame.fact_checker import FactChecker
from defame.helpers.common import TaskState


class Worker(Process):
    """Completing tasks in parallel."""
    id: int

    def __init__(self, identifier: int, kwargs: dict):
        super().__init__(target=self.execute, kwargs=kwargs)
        self.id = identifier
        self.running = True
        self.start()

    def execute(self,
                input_queue: SyncManager.Queue,
                output_queue: SyncManager.Queue,
                device_id: int,
                target_dir: str | Path,
                print_log_level: str = "info",
                **kwargs):
        """Main task handling routine."""

        try:
            device = None if device_id is None else f"cuda:{device_id}"

            logger.set_experiment_dir(target_dir)
            logger.set_log_level(print_log_level)

            # Initialize the fact-checker
            fc = FactChecker(device=device, **kwargs)

        except Exception:
            error_message = f"Worker {self.id} encountered an error during startup:\n"
            error_message += traceback.format_exc()
            logger.error(error_message)
            quit(-1)

        # Complete tasks forever
        while self.running:
            # Fetch the next task and report it
            try:
                task = input_queue.get(block=False)
            except Empty:
                sleep(0.1)
                continue

            try:
                task.assign_worker(self)
                output_queue.put(task)

                logger.set_current_fc_id(task.id)

                # Check which type of task this is
                payload = task.payload

                if isinstance(payload, Content):
                    # Task is claim extraction
                    logger.debug("Extracting claims...")
                    task.set_status(message="Extracting claims...")
                    output_queue.put(task)

                    claims = fc.extract_claims(payload)

                    task.result = dict(
                        claims=claims,
                        topic=payload.topic
                    )

                elif isinstance(payload, Claim):
                    # Task is claim verification
                    logger.debug("Verifying claim...")
                    task.set_status(message="Verifying claim...")
                    output_queue.put(task)

                    doc, meta = fc.verify_claim(payload)
                    doc.save_to(logger.target_dir)

                    task.result = (doc, meta)

                else:
                    raise ValueError(f"Invalid task type: {type(payload)}")

                task.set_status(state=TaskState.DONE, message="Finished successfully.")
                output_queue.put(task)
                logger.debug(f"Task {task.id} completed successfully.")

            except Exception:
                error_message = f"Worker {self.id} encountered an error while processing task {task.id}:\n"
                error_message += traceback.format_exc()
                logger.error(error_message)
                task.set_status(state=TaskState.FAILED, message=error_message)
                output_queue.put(task)
