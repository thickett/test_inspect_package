from inspect_ai.model import (
    ModelAPI,
    modelapi,
    ChatMessage,
    GenerateConfig,
    ModelOutput,
)
from inspect_ai.tool import ToolChoice, ToolInfo
from typing import Any


@modelapi(name="dummy_endpoint")
class CustomModelAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any
    ) -> None:
        super().__init__(model_name)

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # Simulate response by generating random text
        dummy_response = "This is a dummy response for testing purposes."
        return ModelOutput(text=dummy_response, success=True)
