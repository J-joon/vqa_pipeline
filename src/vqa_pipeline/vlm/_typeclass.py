from typing import Protocol, runtime_checkable, TypeVar
from vqa_pipeline.image import ImageProvider

T_Image = TypeVar("T_Image", bound=ImageProvider)

@runtime_checkable
class VLM(Protocol):
    def question(self, images: list[T_Image], prompts: dict[str, str]) -> Result[dict[str, str], str]: ...
