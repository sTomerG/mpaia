import os
import re
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage
from langchain.schema.output import LLMResult
from langchain.schema.runnable import RunnableSequence
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from loguru import logger


class Assistant(ABC):
    """
    Abstract base class for assistant implementations.
    """

    @property
    @abstractmethod
    def used_for(self) -> str:
        """
        A property describing what the assistant is used for.
        """
        pass

    @abstractmethod
    async def process_message(self, message: str, memory: Optional[Any] = None) -> str:
        """
        Process a message and return a response.

        Args:
            message (str): The input message to process.
            memory (Optional[Any]): Optional memory object for maintaining conversation context.

        Returns:
            str: The processed response.
        """
        pass


class SimpleAssistant(Assistant):
    """
    A simple implementation of the Assistant class that echoes the input message.
    """

    async def process_message(self, message: str, memory: Optional[Any] = None) -> str:
        """
        Process a message by simply echoing it back.

        Args:
            message (str): The input message to process.
            memory (Optional[Any]): Unused in this implementation, but required for compatibility.

        Returns:
            str: A string containing "You said: " followed by the input message.
        """
        return f"You said: {message}"

    @property
    def used_for(self) -> str:
        """
        A property describing what the assistant is used for.
        """
        return "Simple assistant that echoes the input message."


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

    async def process_message(self, message: str, memory: Optional[Any] = None) -> str:
        """
        Process a message using the OpenAI language model.

        Args:
            message (str): The input message to process.
            memory (Optional[Any]): Optional memory object for maintaining conversation context.

        Returns:
            str: The AI-generated response or an error message.

        Raises:
            ValueError: If the response type is unexpected.
        """
        try:
            self.memory.add_user_message(message)
            memory = memory or self.memory

            # Ensure the system message is always included
            history = [("system", self.system_message)] + memory.messages

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

    @property
    def used_for(self) -> str:
        """
        A property describing what the assistant is used for.
        """
        return "OpenAI assistant that uses OpenAI's language model for non-personal questions"


class DataAssistant(Assistant):
    """
    An implementation of the Assistant class that interacts with a SQL database.

    Args:
        db_connection (Union[str, SQLDatabase]): The database connection. Can be a SQLAlchemy engine, connection string, or SQLDatabase instance.
        model_name (str, optional): The name of the OpenAI model to use. Defaults to "gpt-4o".

    Raises:
        ValueError: If the OPENAI_API_KEY environment variable is not set or if an invalid database connection is provided.
    """

    def __init__(
        self, db_connection: Union[str, SQLDatabase], model_name: str = "gpt-4o"
    ):
        load_dotenv()  # Load environment variables from .env file

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable is not set")
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.db = self._get_database(db_connection)
        self.llm = ChatOpenAI(model_name=model_name)
        self.toolkit = SQLDatabaseToolkit(llm=self.llm, db=self.db, verbose=True)
        self.tools = self.toolkit.get_tools()
        self.memory = ChatMessageHistory()

        system = """You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        You have access to tools for interacting with the database.
        Only use the given tools. Only use the information returned by the tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

        Always try to find the entry with the date closest to, but less than, the given date using `WHERE date <= <given date> ORDER BY date DESC LIMIT 1`.
        If no date is given, find the entry closest to today's date, using `WHERE date <= date('now') ORDER BY date DESC LIMIT 1`.

        Always give the answer confidentely to the question asked. Don't ask follow-up questions.

        You have access to the following tables: {table_names}
        """.format(table_names=self.db.get_usable_table_names())

        self.system_message = SystemMessage(content=system)
        self.agent = create_react_agent(
            self.llm,
            self.tools,
            state_modifier=self.system_message,
        )

    async def process_message(self, message: str, memory: Optional[Any] = None) -> str:
        """
        Process a natural language message by interacting with the SQL database.

        Args:
            message (str): The input message to process.
            memory (Optional[Any]): The chat history memory to use. If None, use the instance's memory.

        Returns:
            str: The processed response from the database interaction.
        """
        memory = memory or self.memory
        memory.add_user_message(message)
        self.memory.add_user_message(message)
        try:
            logger.info(f"Processing input: '{message}'")
            for s in self.agent.stream({"messages": memory.messages}):
                logger.debug(s)
            result = s.get("agent", {}).get("messages", [{}])[0].get("content", "")
            logger.info(f"Result: '{result}'")
            return str(result)
        except Exception as e:
            logger.exception(f"An error occurred while processing message: {e}")
            return f"An error occurred: {str(e)}"

    def _get_database(self, db_connection: Union[str, SQLDatabase]) -> SQLDatabase:
        """
        Helper method to create a SQLDatabase instance from various input types.
        """
        if isinstance(db_connection, SQLDatabase):
            return db_connection
        elif isinstance(db_connection, str):
            if db_connection.startswith("sqlite:///"):
                return SQLDatabase.from_uri(db_connection)
            else:
                return SQLDatabase.from_uri(f"sqlite:///{db_connection}")
        else:
            try:
                return SQLDatabase(db_connection)
            except Exception as e:
                raise ValueError(f"Invalid database connection: {e}")

    @property
    def used_for(self) -> str:
        """
        A property describing what the assistant is used for.
        """
        return f"Data assistant that interacts with a SQL database with the following tables: {self.db.get_usable_table_names()}. Can answer personal questions and questions about now and today."


class MultiAssistant(Assistant):
    """
    A multi-assistant that uses multiple assistants to process messages.
    """

    def __init__(self, assistants: list[Assistant]):
        self.assistants = assistants
        self.memory = ChatMessageHistory()

    async def select_assistant(self, message: str) -> Assistant:
        if len(self.assistants) == 1:
            logger.debug("Only one assistant, returning it")
            return self.assistants[0]
        logger.debug("Selecting assistant based on message")
        """
        Select the appropriate assistant based on the message.
        """
        selection_message = f"""
        You are a helpful assistant that selects the appropriate assistant based on the message.
        You will be given a message and a list of assistants. Only one assistant will be relevant for the message.
        Select the appropriate assistant and return its index.

        Message: {message}
        Assistants: {[(index, assistant.used_for) for index, assistant in enumerate(self.assistants)]}
        """
        self.llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", selection_message),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        self.chain = RunnableSequence(prompt | self.llm)
        response: LLMResult = await self.chain.ainvoke(
            {
                "input": message,
                "history": self.memory.messages,
            }
        )
        if isinstance(response, AIMessage):
            ai_message = response
        elif isinstance(response, dict) and "generations" in response:
            ai_message = response["generations"][0][0].message
        else:
            raise ValueError(f"Unexpected response type: {type(response)}")

        logger.debug(f"Assistant selection response: {ai_message.content}")
        match = re.search(r"\d+", ai_message.content)
        if match:
            index = int(match.group())
            selected_assistant = self.assistants[index]
            logger.info(
                f"Selected assistant: {selected_assistant.__class__.__name__} for message: {message}"
            )
            return selected_assistant
        else:
            raise ValueError("No assistant index found in the response")

    async def process_message(self, message: str, memory: Optional[Any] = None) -> str:
        """
        Process a message using the selected assistant.
        """
        self.memory.add_user_message(message)
        selected_assistant = await self.select_assistant(message)
        ai_message = await selected_assistant.process_message(
            message, memory=self.memory
        )
        self.memory.add_ai_message(ai_message)
        return ai_message

    @property
    def used_for(self) -> str:
        """
        A property describing what the assistant is used for.
        """
        return f"Multi-assistant that chooses the appropriate assistant based on the message. Assistants: {self.assistants}"
