import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set tracing flag
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "true")

# Set the LangSmith API Key
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# Set the OpenAI API Key (if it's not already set)
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
