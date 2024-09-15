import asyncio
import os
import signal
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
from loguru import logger
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from mpaia.assistant import Assistant, OpenAIAssistant
from mpaia.bot import Bot
from mpaia.jobs import Job

# Load environment variables from .env file
load_dotenv()


class TelegramBot(Bot):
    """
    A Telegram bot that uses an AI assistant to process messages and manage scheduled jobs.

    Args:
        assistant (Assistant): The AI assistant to use for processing messages.
    """

    def __init__(self, assistant: Assistant):
        super().__init__(assistant)
        self.allowed_chat_ids: Optional[set[int]] = self._get_allowed_chat_ids()
        self.admin_chat_id: int = int(os.getenv("ADMIN_CHAT_ID", "0"))
        self.scheduler: AsyncIOScheduler = AsyncIOScheduler()
        self.scheduler.start()

    def _get_allowed_chat_ids(self) -> Optional[set[int]]:
        """
        Get the set of allowed chat IDs from the environment variable.

        Returns:
            Optional[set[int]]: A set of allowed chat IDs, or None if not set.
        """
        allowed_ids = os.getenv("ALLOWED_CHAT_IDS")
        return (
            set(int(id.strip()) for id in allowed_ids.split(","))
            if allowed_ids
            else None
        )

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the /start command.

        Args:
            update (Update): The update object from Telegram.
            context (ContextTypes.DEFAULT_TYPE): The context object for the handler.
        """
        if (
            self.allowed_chat_ids
            and update.effective_chat.id not in self.allowed_chat_ids
        ):
            await update.message.reply_text(
                "Sorry, you're not authorized to use this bot."
            )
            return
        await update.message.reply_text(
            "Hello! I am your personal AI assistant powered by OpenAI. How can I help you?"
        )

    async def handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle incoming messages asynchronously.
        """
        if (
            self.allowed_chat_ids
            and update.effective_chat.id not in self.allowed_chat_ids
        ):
            return
        response = await self.assistant.process_message(update.message.text)
        await update.message.reply_text(response)

    async def send_log_message(
        self, context: ContextTypes.DEFAULT_TYPE, message: str
    ) -> None:
        """
        Send a log message to the admin chat.

        Args:
            context (ContextTypes.DEFAULT_TYPE): The context object for the handler.
            message (str): The message to send.
        """
        if self.admin_chat_id:
            await context.bot.send_message(chat_id=self.admin_chat_id, text=message)

    async def send_scheduled_message(
        self, context: ContextTypes.DEFAULT_TYPE, chat_id: int, message: str
    ) -> None:
        """
        Send a scheduled message to a specific chat.

        Args:
            context (ContextTypes.DEFAULT_TYPE): The context object for the handler.
            chat_id (int): The ID of the chat to send the message to.
            message (str): The message to send.
        """
        if context and context.bot:
            await context.bot.send_message(chat_id, text=message)
        elif hasattr(self, "application"):
            await self.application.bot.send_message(chat_id, text=message)
        else:
            logger.error("Unable to send message: No bot instance available")

    async def send_message(self, chat_id: int, message: str) -> None:
        """
        Send a message to a specific chat.

        Args:
            chat_id (int): The ID of the chat to send the message to.
            message (str): The message to send.
        """
        if not hasattr(self, "application"):
            await self.initialize()

        await self.application.bot.send_message(chat_id, text=message)

    def add_job(self, job: Job) -> None:
        """
        Add a new job to the scheduler.

        Args:
            job (Job): The job to add.
        """
        job_id = job.get_id()
        if job_id in self.jobs:
            logger.warning(f"Job with ID {job_id} already exists. Replacing it.")
            self.remove_job(job_id)

        self.scheduler.add_job(
            job.execute,
            job.trigger,
            args=[self],
            id=job_id,
        )
        self.jobs[job_id] = job
        logger.info(f"Added job: {job_id}")

    def remove_job(self, job_id: str) -> None:
        """
        Remove a job from the scheduler.

        Args:
            job_id (str): The ID of the job to remove.
        """
        if job_id in self.jobs:
            self.scheduler.remove_job(job_id)
            del self.jobs[job_id]
            logger.info(f"Removed job: {job_id}")
        else:
            logger.warning(f"Job with ID {job_id} not found.")

    def list_jobs(self) -> list[str]:
        """
        List all scheduled jobs.

        Returns:
            list[str]: A list of job IDs.
        """
        return list(self.jobs.keys())

    async def error_handler(
        self, update: object, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle errors that occur during the execution of handlers.

        Args:
            update (object): The update object from Telegram.
            context (ContextTypes.DEFAULT_TYPE): The context object for the handler.
        """
        logger.error(f"Exception while handling an update: {context.error}")
        await self.send_log_message(context, f"Error: {context.error}")

    async def initialize(self) -> None:
        """
        Initialize the bot application.
        """
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is not set")

        self.application = Application.builder().token(token).build()
        await self.application.initialize()
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )
        self.application.add_error_handler(self.error_handler)

        logger.info("Telegram bot initialized")

    async def run(self) -> None:
        """
        Run the bot asynchronously.
        """
        await self.initialize()
        logger.info("Starting Telegram bot")
        await self.application.start()
        await self.application.updater.start_polling()

        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        stop_signal = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_signal.set)

        # Run the bot until stopped
        try:
            await stop_signal.wait()
        finally:
            await self.shutdown()

    async def shutdown(self, signal: Optional[signal.Signals] = None) -> None:
        """
        Shutdown the bot gracefully.
        """
        if signal:
            logger.info(f"Received exit signal {signal.name}...")
        logger.info("Shutting down...")
        await self.application.updater.stop()
        await self.application.stop()
        await self.application.shutdown()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    # Example usage
    assistant = OpenAIAssistant()
    bot = TelegramBot(assistant)
    asyncio.run(bot.run())
