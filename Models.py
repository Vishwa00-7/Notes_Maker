import warnings
# Silence the Python 3.14 pydantic warning to keep your terminal clean
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")



#Imports 

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model


try:
    ling = init_chat_model(model="inclusionai/ling-3.0-flash-vl:free",
                            model_provider="openrouter",  temperature = 0.1)
except Exception as e:
    ling = None


chatgpt = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="groq",
    temperature = 0.1,
    max_tokens = 8000
)

meta = init_chat_model(
    model = "openai/gpt-oss-120b",
    model_provider="groq",
    temperature = 0.1,
    max_tokens = 8000
)

qwen = init_chat_model(
    model = "openai/gpt-oss-120b",
    model_provider="groq",
    temperature = 0.1,
    max_tokens = 8000
)

"""
Curriculum Planner (Master Prompt) ➔ chatgpt (openai/gpt-oss-20b)

Meta-Prompt Generator ➔ meta (openai/gpt-oss-20b)

Content Generator ➔ qwen (qwen/qwen3.8-27b)

"""


