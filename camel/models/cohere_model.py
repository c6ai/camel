# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import os
import logging
import uuid
from cohere.core.api_error import ApiError
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from cohere.chat import ChatResponse

from camel.configs import COHERE_API_PARAMS
from camel.messages import OpenAIMessage
from camel.models import BaseModelBackend
from camel.types import ChatCompletion, ModelType, ModelPlatformType
from camel.utils import (
    BaseTokenCounter,
    OpenAITokenCounter,
    api_keys_required,
)

try:
    import os

    if os.getenv("AGENTOPS_API_KEY") is not None:
        from agentops import LLMEvent, record
    else:
        raise ImportError
except (ImportError, AttributeError):
    LLMEvent = None

class CohereModel(BaseModelBackend):
    """Cohere API in a unified BaseModelBackend interface."""

    def __init__(
        self,
        model_type: ModelType,
        model_config_dict: Dict[str, Any],
        api_key: Optional[str] = None,
        token_counter: Optional[BaseTokenCounter] = None,
        model_platform: Optional[ModelPlatformType] = None,
    ):
        super().__init__(model_type, model_config_dict, api_key, token_counter=token_counter)
        self._api_key = api_key or os.environ.get("COHERE_API_KEY")
        self.model_platform = model_platform

        print(f"API Key loaded: {'*' * (len(self._api_key) - 4) + self._api_key[-4:] if self._api_key else 'None'}")

        import cohere
        self._client = cohere.Client(api_key=self._api_key)
        self._token_counter: Optional[BaseTokenCounter] = None

    def _to_openai_response(self, response: 'ChatResponse') -> ChatCompletion:
        unique_id = str(uuid.uuid4())
        
        # Safely access nested attributes
        def safe_get(obj, *keys):
            for key in keys:
                if isinstance(obj, dict):
                    obj = obj.get(key)
                else:
                    obj = getattr(obj, key, None)
                if obj is None:
                    return None
            return obj

        obj = ChatCompletion.construct(
            id=unique_id,
            choices=[
                dict(
                    index=0,
                    message={
                        "role": "assistant",
                        "content": response.text,
                    },
                    finish_reason=safe_get(response, 'finish_reason'),
                )
            ],
            created=safe_get(response, 'meta', 'created_at'),
            model=self.model_type.value,
            object="chat.completion",
            usage={
                "prompt_tokens": safe_get(response, 'meta', 'billed_tokens', 'prompt_tokens') or 0,
                "completion_tokens": safe_get(response, 'meta', 'billed_tokens', 'completion_tokens') or 0,
                "total_tokens": (
                    (safe_get(response, 'meta', 'billed_tokens', 'prompt_tokens') or 0) +
                    (safe_get(response, 'meta', 'billed_tokens', 'completion_tokens') or 0)
                ),
            },
        )
        return obj


    def _to_cohere_chatmessage(self, messages: List[OpenAIMessage]) -> List[Dict[str, str]]:
        new_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "user":
                new_messages.append({"role": "User", "message": content})
            elif role == "assistant":
                new_messages.append({"role": "Chatbot", "message": content})
            elif role == "system":
                new_messages.append({"role": "System", "message": content})
            else:
                raise ValueError(f"Unsupported message role: {role}")

        return new_messages

    @property
    def token_counter(self) -> BaseTokenCounter:
        """Initialize the token counter for the model backend.

        Returns:
            BaseTokenCounter: The token counter following the model's
                tokenization style.
        """
        if not self._token_counter:
            self._token_counter = OpenAITokenCounter(
                model=ModelType.GPT_3_5_TURBO
            )
        return self._token_counter

    @api_keys_required("COHERE_API_KEY")
    def run(self, messages: List[OpenAIMessage]) -> ChatCompletion:
        """Runs inference of Cohere chat completion.

        Args:
            messages (List[OpenAIMessage]): Message list with the chat history
                in OpenAI API format.

        Returns:
            ChatCompletion.
        """
        cohere_messages = self._to_cohere_chatmessage(messages)

        # Filter out unsupported parameters
        supported_params = {
            'temperature', 'p', 'k', 'max_tokens', 'prompt_truncation'
        }
        filtered_config = {
            k: v for k, v in self.model_config_dict.items()
            if k in supported_params
        }

        try:
            response = self._client.chat(
                message=cohere_messages[-1]["message"],
                chat_history=cohere_messages[:-1],
                model=self.model_type.value,
                **filtered_config,
            )
        except ApiError as e:
            logging.error(f"Cohere API Error: {e.status_code}")
            logging.error(f"Error body: {e.body}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error when calling Cohere API: {str(e)}")
            raise

        openai_response = self._to_openai_response(response)

        # Add AgentOps LLM Event tracking
        if LLMEvent:
            llm_event = LLMEvent(
                thread_id=openai_response.id,
                prompt=" ".join(
                    [message.get("content") for message in messages]
                ),
                prompt_tokens=openai_response.usage.prompt_tokens,
                completion=openai_response.choices[0].message.content,
                completion_tokens=openai_response.usage.completion_tokens,
                model=self.model_type.value,
            )
            record(llm_event)

        return openai_response

    def check_model_config(self):
        """Check whether the model configuration contains any
        unexpected arguments to Cohere API.

        Raises:
            ValueError: If the model configuration dictionary contains any
                unexpected arguments to Cohere API.
        """
        supported_params = {
            'temperature', 'p', 'k', 'max_tokens', 'prompt_truncation'
        }
        for param in self.model_config_dict:
            if param not in supported_params:
                raise ValueError(
                    f"Unexpected argument `{param}` is "
                    "input into Cohere model backend."
                )
