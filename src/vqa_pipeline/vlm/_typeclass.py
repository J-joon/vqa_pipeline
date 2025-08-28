from typing import Protocol, runtime_checkable
from vqa_pipeline.image import t_image

@runtime_checkable
class VLM(Protocol):
    def question(self, images: list[t_image], prompt: str)->str: ...
