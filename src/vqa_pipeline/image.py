from typing import TypeAlias
from PIL.Image import Image
from typing import Protocol, runtime_checkable

t_image: TypeAlias = Image

@runtime_checkable
class ImageProvider(Protocol):
    image: t_image
