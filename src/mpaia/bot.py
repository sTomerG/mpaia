from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional


from mpaia.assistant import Assistant

if TYPE_CHECKING:
    from mpaia.jobs import Job


class Bot(ABC):
    """
    Abstract base class for bot implementations.

    Args:
        assistant (Assistant): The AI assistant to use for processing messages.

    Attributes:
        assistant (Assistant): The AI assistant used for processing messages.
        allowed_chat_ids (Optional[set[int]]): A set of allowed chat IDs, or None if not set.
        admin_chat_id (int): The chat ID of the admin.
        jobs (dict[str, Job]): A dictionary of scheduled jobs.
    """

    @abstractmethod
    def __init__(self, assistant: Assistant):
        self.assistant = assistant
        self.allowed_chat_ids: Optional[set[int]] = None
        self.admin_chat_id: int = 0
        self.jobs: dict[str, Job] = {}

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the bot application.
        """
        pass

    @abstractmethod
    async def send_message(self, chat_id: int, message: str) -> None:
        """
        Send a message to a specific chat.

        Args:
            chat_id (int): The ID of the chat to send the message to.
            message (str): The message to send.
        """
        pass

    @abstractmethod
    def add_job(self, job: "Job") -> None:
        """
        Add a new job to the scheduler.

        Args:
            job (Job): The job to add.
        """
        pass

    @abstractmethod
    def remove_job(self, job_id: str) -> None:
        """
        Remove a job from the scheduler.

        Args:
            job_id (str): The ID of the job to remove.
        """
        pass

    @abstractmethod
    def list_jobs(self) -> list[str]:
        """
        List all scheduled jobs.

        Returns:
            list[str]: A list of job IDs.
        """
        pass

    @abstractmethod
    async def run(self) -> None:
        """
        Run the bot.
        """
        pass

    @abstractmethod
    async def handle_message(self, *args: Any, **kwargs: Any) -> None:
        """
        Handle incoming messages asynchronously.
        """
        pass

    @abstractmethod
    async def shutdown(self, signal: Optional[Any] = None) -> None:
        """
        Shutdown the bot gracefully.
        """
        pass
