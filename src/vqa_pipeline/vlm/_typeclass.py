from typing import Protocol, runtime_checkable, TypeVar, TypeAlias
from vqa_pipeline.image import ImageProvider
from vqa_pipeline import Box
from static_error_handler import Ok, Err, Result

T_Image = TypeVar("T_Image", bound=ImageProvider)

@runtime_checkable
class VLM(Protocol):
    def question(self, images: list[T_Image], prompts: tuple[tuple[str, str],...]) -> Result[tuple[tuple[str, str], ...], str]: ...

@runtime_checkable
class BBoxProvider(Protocol):
    def query(self, image: T_Image, query: str) -> Result[tuple[Box, ...], str]: ...
