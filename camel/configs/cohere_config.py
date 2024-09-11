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
from __future__ import annotations

from typing import Optional

from pydantic import field_validator

from camel.configs.base_config import BaseConfig


class CohereConfig(BaseConfig):
    """Defines the parameters for generating chat completions using the
    Cohere API.

    Args:
        temperature (Optional[float]): Controls randomness in the model.
            Values closer to 0 make the model more deterministic, while
            values closer to 1 make it more random. Defaults to None.
        p (Optional[float]): Sets a p% nucleus sampling threshold.
            Defaults to None.
        k (Optional[int]): Limits the number of tokens to sample from on
            each step. Defaults to None.
        max_tokens (Optional[int]): The maximum number of tokens to generate.
            Defaults to None.
        prompt_truncation (Optional[str]): How to truncate the prompt if it
            exceeds the model's context length. Can be 'START', 'END', or
            'AUTO'. Defaults to None.
    """

    temperature: Optional[float] = None
    p: Optional[float] = None
    k: Optional[int] = None
    max_tokens: Optional[int] = None
    prompt_truncation: Optional[str] = None

    @field_validator("prompt_truncation")
    @classmethod
    def validate_prompt_truncation(cls, v):
        if v is not None and v not in ['START', 'END', 'AUTO']:
            raise ValueError("prompt_truncation must be 'START', 'END', or 'AUTO'")
        return v

    def as_dict(self):
        return {k: v for k, v in super().as_dict().items() if v is not None}


COHERE_API_PARAMS = {param for param in CohereConfig().model_fields.keys()}
