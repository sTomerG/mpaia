# MPAIA (My Personal AI Assistant)

A Python package for your personal AI assistant with Telegram integration and OpenAI capabilities.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sTomerG/mpaia.git
   cd mpaia
   ```

2. Install the package:

   ```bash
   pip install -e .
   ```

## Configuration

1. Create a `.env` file in the root directory with the following content:

   ```env
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   OPENAI_API_KEY=your_openai_api_key
   ALLOWED_CHAT_IDS=comma_separated_list_of_allowed_chat_ids
   ADMIN_CHAT_ID=your_admin_chat_id
   ```

   Replace the placeholders with your actual values.

2. Create a `user_config` directory in the root of the project if it doesn't exist already.

## Usage

1. Create custom jobs by subclassing `TelegramMessageJob` in `user_config/custom_jobs.py`. For example:

   ```python
   from mpaia.jobs import TelegramMessageJob
   from mpaia.assistant import Assistant

   class SalaryJob(TelegramMessageJob):
       def __init__(self, chat_id: int, assistant: Assistant):
           super().__init__(
               "0 8 1 * *",  # Run on the first day of every month at 8 AM
               "Generate a friendly reminder that it's salary day",
               chat_id,
               assistant
           )
           # Add any additional initialization here

       async def execute(self, bot):
           # Optionally implement your custom execution logic here
           await super().execute(bot)
   ```

2. Create a `run_bot.py` file in the `user_config` directory to set up and run your bot:

   ```python
   from dotenv import load_dotenv

   from mpaia.telegram_bot import TelegramBot
   from mpaia.assistant import OpenAIAssistant
   from custom_jobs import SalaryJob

   def run_bot():
       load_dotenv(override=True)
       assistant = OpenAIAssistant()
       bot = TelegramBot(assistant)

       # Add custom jobs
       if bot.allowed_chat_ids:
           for chat_id in bot.allowed_chat_ids:
               bot.add_job(SalaryJob(chat_id, assistant))

       bot.run()

   if __name__ == "__main__":
       run_bot()
   ```

3. Run your bot:

   ```bash
   python user_config/run_bot.py
   ```

## Customization

- To add more custom jobs, create new classes in `user_config/custom_jobs.py` and add them to the bot in `run_bot.py`.
- Modify the `OpenAIAssistant` parameters in `run_bot.py` to change the AI model or its behavior.

## Note

Ensure that your Telegram bot has the necessary permissions to send messages in the allowed chats.

For more detailed information on the MPAIA package and its components, refer to the source code and docstrings in the `src/mpaia` directory.
