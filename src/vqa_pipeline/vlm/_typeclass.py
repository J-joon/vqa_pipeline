from typing import Protocol, runtime_checkable, TypeVar
from vqa_pipeline.image import ImageProvider
from static_error_handler import *

T_Image = TypeVar("T_Image", bound=ImageProvider)

@runtime_checkable
class VLM(Protocol):
    def question(self, images: list[T_Image], prompts: tuple[tuple[str, str],...]) -> Result[tuple[tuple[str, str], ...], str]: ...
