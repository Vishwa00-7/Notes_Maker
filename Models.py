import os
import warnings
from typing import Any, Dict, List, Optional
import requests

# Silence the Python 3.14 pydantic warning to keep your terminal clean
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import BaseMessage


# -----------------------------------------------------------------------------
# Custom OpenRouter Chat Model (Native LangChain Support)
# -----------------------------------------------------------------------------

class OpenRouterChatModel(SimpleChatModel):
    """
    Native LangChain-compatible chat model for OpenRouter.
    Seamlessly supports free models like Space Bunny Alpha without extra external dependencies.
    """
    model_name: str = "stealth/space-bunny-alpha"
    temperature: float = 0.1
    max_tokens: int = 12000
    api_key: Optional[str] = None
    timeout: int = 120
    reasoning_effort: Optional[str] = "medium"

    @property
    def _llm_type(self) -> str:
        return "openrouter-chat"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any
    ) -> str:
        key = self.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPEN_ROUTER_API_KEY")
        if not key:
            raise ValueError(
                "OpenRouter API key not found. Please set OPEN_ROUTER_API_KEY or OPENROUTER_API_KEY in your .env file."
            )

        payload_messages = []
        for m in messages:
            msg_type = getattr(m, "type", "user")
            if msg_type in ("human", "user"):
                role = "user"
            elif msg_type in ("ai", "assistant"):
                role = "assistant"
            elif msg_type in ("system",):
                role = "system"
            else:
                role = "user"
            payload_messages.append({"role": role, "content": str(m.content)})

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/langchain",
            "X-Title": "Notes Maker"
        }
        data = {
            "model": self.model_name,
            "messages": payload_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            data["reasoning"] = {"effort": self.reasoning_effort}
        if stop:
            data["stop"] = stop

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter API error ({response.status_code}): {response.text}")

        res_json = response.json()
        if "error" in res_json:
            error_data = res_json.get("error")
            raise RuntimeError(f"OpenRouter returned error: {error_data}")

        choices = res_json.get("choices", [])
        if not choices:
            raise RuntimeError(f"OpenRouter returned empty choices: {res_json}")

        msg = choices[0].get("message") or {}
        content = msg.get("content")

        # Fallback handling for reasoning models (where content may be None or in reasoning)
        if not content:
            content = msg.get("reasoning")
        if not content:
            content = msg.get("refusal")
        if not content and "reasoning_details" in msg:
            details = msg.get("reasoning_details", [])
            if isinstance(details, list):
                texts = [d.get("text", "") for d in details if isinstance(d, dict) and d.get("text")]
                if texts:
                    content = "\n".join(texts)

        # Strictly enforce non-null string type to guarantee AIMessage Pydantic v2 validation passes
        content_str = str(content or "").strip()
        if not content_str:
            finish_reason = choices[0].get("finish_reason")
            raise RuntimeError(
                f"OpenRouter model '{self.model_name}' returned empty content (finish_reason: {finish_reason}). "
                "Tokens may have been exhausted by reasoning or provider returned blank output. Retrying..."
            )

        return content_str


# -----------------------------------------------------------------------------
# Model Initializations
# -----------------------------------------------------------------------------

# Groq Models
chatgpt = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="groq",
    temperature=0.1,
    max_tokens=8000
)

meta = init_chat_model(
    model="openai/gpt-oss-20b",
    model_provider="groq",
    temperature=0.1,
    max_tokens=8000
)

qwen = init_chat_model(
    model="qwen/qwen3.8-27b",
    model_provider="groq",
    temperature=0.1,
    max_tokens=8000
)

# OpenRouter Models (Free)
space_bunny = OpenRouterChatModel(
    model_name="stealth/space-bunny-alpha",
    temperature=0.1,
    max_tokens=12000,
    reasoning_effort="medium"
)

try:
    ling = OpenRouterChatModel(
        model_name="inclusionai/ling-3.0-flash-vl:free",
        temperature=0.1,
        max_tokens=8000,
        reasoning_effort=None
    )
except Exception:
    ling = None


# -----------------------------------------------------------------------------
# Model Registry & Dynamic Model Selection
# -----------------------------------------------------------------------------

# Dictionary of all models available for user selection
AVAILABLE_MODELS: Dict[str, Any] = {
    "chatgpt": chatgpt,              # openai/gpt-oss-120b (Groq - high capacity reasoning)
    "meta": meta,                    # openai/gpt-oss-20b (Groq - fast lightweight reasoning)
    "qwen": qwen,                    # qwen/qwen3.8-27b (Groq - expressive markdown notes)
    "space_bunny": space_bunny,      # stealth/space-bunny-alpha (OpenRouter - Free 1M context)
}

if ling is not None:
    AVAILABLE_MODELS["ling"] = ling


def get_model(name: Optional[str] = None) -> Any:
    """
    Retrieve an initialized chat model by key name (case-insensitive).
    Falls back to 'chatgpt' if name is empty or not found in AVAILABLE_MODELS.
    """
    if not name:
        return AVAILABLE_MODELS.get("chatgpt", next(iter(AVAILABLE_MODELS.values())))

    clean_name = str(name).strip().lower().replace(" ", "_").replace("-", "_")

    # 1. Exact match
    for key, model_instance in AVAILABLE_MODELS.items():
        if key.lower() == clean_name:
            return model_instance

    # 2. Substring match
    for key, model_instance in AVAILABLE_MODELS.items():
        if key.lower() in clean_name or clean_name in key.lower():
            return model_instance

    # 3. Fallback
    print(f"[WARN] Model '{name}' not found in AVAILABLE_MODELS. Falling back to 'chatgpt'.")
    return AVAILABLE_MODELS.get("chatgpt", next(iter(AVAILABLE_MODELS.values())))




