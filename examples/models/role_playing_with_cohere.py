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

import cohere
from cohere.core.api_error import ApiError
from colorama import Fore

from camel.configs import CohereConfig
from camel.models import ModelFactory
from camel.societies import RolePlaying
from camel.types import ModelPlatformType, ModelType
from camel.utils import print_text_animated


def test_cohere_api(api_key):
    client = cohere.Client(api_key)

    print("Testing Cohere generate endpoint:")
    try:
        response = client.generate(prompt="Hello, world!")
        print("Generate test successful.")
        print(f"Response: {response.generations[0].text}")
    except ApiError as e:
        print(f"Generate test failed. Status code: {e.status_code}")
        print(f"Error body: {e.body}")
    except Exception as e:
        print(f"Generate test failed. Error: {e!s}")

    print("\nTesting Cohere chat endpoint:")
    try:
        response = client.chat(message="Hello", model="command")
        print("Chat test successful.")
        print(f"Response: {response.text}")
    except ApiError as e:
        print(f"Chat test failed. Status code: {e.status_code}")
        print(f"Error body: {e.body}")
    except Exception as e:
        print(f"Chat test failed. Error: {e!s}")


def main(
    model_platform=ModelPlatformType.COHERE,
    model_type=ModelType.COHERE_COMMAND_R,
    chat_turn_limit=10,
) -> None:
    # Test Cohere API
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        print("COHERE_API_KEY not found in environment variables.")
        return
    print(f"API Key: {'*' * (len(api_key) - 4) + api_key[-4:]}")
    test_cohere_api(api_key)
    task_prompt = (
        "Assume now is 2024 in the Gregorian calendar, "
        "estimate the current age of University of Oxford "
        "and then add 10 more years to this age."
    )

    model_config = CohereConfig(temperature=0.2)

    # Create model for both assistant and user
    model = ModelFactory.create(
        model_platform=model_platform,
        model_type=model_type,
        model_config_dict=model_config.as_dict(),
    )

    # Set up role playing session
    role_play_session = RolePlaying(
        assistant_role_name="Searcher",
        user_role_name="Professor",
        assistant_agent_kwargs=dict(model=model),
        user_agent_kwargs=dict(model=model),
        task_prompt=task_prompt,
        with_task_specify=False,
    )

    print(
        Fore.GREEN
        + f"AI Assistant sys message:\n{role_play_session.assistant_sys_msg}\n"
    )
    print(
        Fore.BLUE + f"AI User sys message:\n{role_play_session.user_sys_msg}\n"
    )

    print(Fore.YELLOW + f"Original task prompt:\n{task_prompt}\n")
    print(
        Fore.CYAN
        + "Specified task prompt:"
        + f"\n{role_play_session.specified_task_prompt}\n"
    )
    print(Fore.RED + f"Final task prompt:\n{role_play_session.task_prompt}\n")

    n = 0
    input_msg = role_play_session.init_chat()
    while n < chat_turn_limit:
        n += 1
        assistant_response, user_response = role_play_session.step(input_msg)

        if assistant_response.terminated:
            print(
                Fore.GREEN
                + (
                    "AI Assistant terminated. Reason: "
                    f"{assistant_response.info['termination_reasons']}."
                )
            )
            break
        if user_response.terminated:
            print(
                Fore.GREEN
                + (
                    "AI User terminated. "
                    f"Reason: {user_response.info['termination_reasons']}."
                )
            )
            break

        # Print output from the user
        print_text_animated(
            Fore.BLUE + f"AI User:\n\n{user_response.msg.content}\n"
        )

        # Print output from the assistant
        print_text_animated(Fore.GREEN + "AI Assistant:")
        print_text_animated(f"{assistant_response.msg.content}\n")

        if "CAMEL_TASK_DONE" in user_response.msg.content:
            break

        input_msg = assistant_response.msg


if __name__ == "__main__":
    main()
