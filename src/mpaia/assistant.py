import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage
from langchain.schema.output import LLMResult
from langchain.schema.runnable import RunnableSequence
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from loguru import logger


class Assistant(ABC):
    """
    Abstract base class for assistant implementations.
    """

    @abstractmethod
    async def process_message(self, message: str) -> str:
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

    async def process_message(self, message: str) -> str:
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
        self.system_message = "You are a helpful personal assistant called Mpaia. Your replies are short and concise."
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_message),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        self.chain = RunnableSequence(prompt | self.llm)

    async def process_message(self, message: str) -> str:
        """
        Process a message using the OpenAI language model.

        Args:
            message (str): The input message to process.

        Returns:
            str: The AI-generated response or an error message.

        Raises:
            ValueError: If the response type is unexpected.
            Exception: If any other error occurs during processing.
        """
        try:
            self.memory.add_user_message(message)

            # Ensure the system message is always included
            history = [("system", self.system_message)] + self.memory.messages

            response: LLMResult = await self.chain.ainvoke(
                {"input": message, "history": history}
            )
            if isinstance(response, AIMessage):
                ai_message = response
            elif isinstance(response, dict) and "generations" in response:
                ai_message = response["generations"][0][0].message
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")

            self.memory.add_ai_message(ai_message)
            return str(ai_message.content)
        except Exception as e:
            logger.exception(f"An error occurred while processing message: {e}")
            return f"An error occurred: {str(e)}"
