from abc import ABC, abstractmethod
from typing import Any

from apscheduler.triggers.cron import CronTrigger

from mpaia.assistant import Assistant


class Job(ABC):
    """
    Abstract base class for scheduled jobs.

    Attributes:
        cron_expression (str): The cron expression for scheduling the job.
        prompt (str): The prompt to be used for generating messages.
        chat_id (int): The ID of the chat where messages will be sent.
        trigger (CronTrigger): The trigger for scheduling the job.
        assistant (Assistant): The assistant used for processing messages.

    Args:
        cron_expression (str): The cron expression for scheduling the job.
        prompt (str): The prompt to be used for generating messages.
        chat_id (int): The ID of the chat where messages will be sent.
        assistant (Assistant): The assistant used for processing messages.
    """

    def __init__(
        self, cron_expression: str, prompt: str, chat_id: int, assistant: Assistant
    ):
        self.cron_expression = cron_expression
        self.prompt = prompt
        self.chat_id = chat_id
        self.trigger = CronTrigger.from_crontab(cron_expression)
        self.assistant = assistant

    @abstractmethod
    async def execute(self, bot: Any) -> None:
        """
        Execute the job. This method must be implemented by subclasses.

        Args:
            bot (Any): The bot instance used for sending messages.
        """
        pass

    def get_id(self) -> str:
        """
        Generate a unique identifier for the job.

        Returns:
            str: A unique identifier for the job.
        """
        return f"{self.__class__.__name__}_{self.cron_expression}_{self.chat_id}"

    async def generate_message(self) -> str:
        """
        Generate a message using the assistant and the prompt.

        Returns:
            str: The generated message.
        """
        return await self.assistant.process_message(self.prompt)


class TelegramMessageJob(Job):
    """
    A job that sends a message to a Telegram chat at scheduled times.
    """

    async def execute(self, bot: Any) -> None:
        """
        Execute the job by generating and sending a message.

        Args:
            bot (Any): The bot instance used for sending messages.
        """
        message = await self.generate_message()
        await bot.send_scheduled_message(None, self.chat_id, message)

    async def execute_immediately(self, bot: Any) -> None:
        """
        Execute the job immediately without scheduling.

        Args:
            bot (Any): The bot instance used for sending messages.
        """
        message = await self.generate_message()
        await bot.send_message(self.chat_id, message)  # Changed this line


class TestJob(Job):
    """
    A job that immediately sends a test message once.

    Args:
        prompt (str): The prompt to be used for generating messages.
        chat_id (int): The ID of the chat where messages will be sent.
        assistant (Assistant): The assistant used for processing messages.
    """

    def __init__(self, prompt: str, chat_id: int, assistant: Assistant):
        super().__init__(
            "* * * * *", prompt, chat_id, assistant
        )  # Use a dummy cron expression that runs every minute

    async def execute(self, bot: Any) -> None:
        """
        Execute the job by immediately generating and sending a test message.

        Args:
            bot (Any): The bot instance used for sending messages.
        """
        message = f"Test message: {self.generate_message()}"
        await bot.send_scheduled_message(None, self.chat_id, message)
        # Remove the job after execution
        bot.scheduler.remove_job(self.get_id())
