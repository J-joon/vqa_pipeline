from typing import Protocol, runtime_checkable
from vqa_pipeline.image import ImageProvider

@runtime_checkable
class VLM(Protocol):
    def question(self, images: list[ImageProvider], prompts: dict[str, str]) -> Result[dict[str, str], str]: ...
