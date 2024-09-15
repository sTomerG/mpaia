import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableSequence
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from loguru import logger


class Assistant(ABC):
    """
    Abstract base class for assistant implementations.
    """

    @abstractmethod
    def process_message(self, message: str) -> str:
        """
        Process a message and return a response.

        Args:
            message (str): The input message to process.

        Returns:
            str: The processed response.
        """
        pass


class SimpleAssistant(Assistant):
    """
    A simple implementation of the Assistant class that echoes the input message.
    """

    def process_message(self, message: str) -> str:
        """
        Process a message by simply echoing it back.

        Args:
            message (str): The input message to process.

        Returns:
            str: A string containing "You said: " followed by the input message.
        """
        return f"You said: {message}"


class OpenAIAssistant(Assistant):
    """
    An implementation of the Assistant class that uses OpenAI's language model.

    Args:
        model_name (str): The name of the OpenAI model to use. Defaults to "gpt-4o-mini".

    Raises:
        ValueError: If the OPENAI_API_KEY environment variable is not set.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        load_dotenv()  # Load environment variables from .env file

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable is not set")
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.llm = ChatOpenAI(temperature=0.7, model_name=model_name)
        self.memory = ChatMessageHistory()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful AI assistant."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        self.chain = RunnableSequence(prompt | self.llm)

    def process_message(self, message: str) -> str:
        """
        Process a message using the OpenAI language model.

        Args:
            message (str): The input message to process.

        Returns:
            str: The AI-generated response or an error message.
        """
        try:
            self.memory.add_user_message(message)
            response = self.chain.invoke(
                {"history": self.memory.messages, "input": message}
            )
            ai_message = str(response.content)  # Explicitly convert to string
            self.memory.add_ai_message(ai_message)
            return ai_message.strip()
        except Exception as e:
            logger.exception(f"An error occurred while processing message: {e}")
            return f"An error occurred: {str(e)}"
