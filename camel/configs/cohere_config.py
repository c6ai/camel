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

from camel.configs.base_config import BaseConfig


class CohereConfig(BaseConfig):
    r"""Defines the parameters for generating chat completions using the
    Cohere API.
    Args:
        temperature (Optional[float], optional):
            Controls randomness in the model.
            Values closer to 0 make the model more deterministic, while
            values closer to 1 make it more random. (default: :obj:`None`)
        p (Optional[float], optional): Sets a p% nucleus sampling threshold.
            (default: :obj:`None`)
        k (Optional[int], optional):
            Limits the number of tokens to sample from on
            each step. (default: :obj:`None`)
        max_tokens (Optional[int], optional):
            The maximum number of tokens to generate.
            (default: :obj:`None`)
        prompt_truncation (Optional[str], optional):
            How to truncate the prompt if it
            exceeds the model's context length. Can be 'START', 'END', or
            'AUTO'. (default: :obj:`None`)
    """

    temperature: Optional[float] = None
    p: Optional[float] = None
    k: Optional[int] = None
    max_tokens: Optional[int] = None
    prompt_truncation: Optional[str] = None
    seed: Optional[int] = None

    def as_dict(self):
        return {k: v for k, v in super().as_dict().items() if v is not None}


COHERE_API_PARAMS = {param for param in CohereConfig().model_fields.keys()}
