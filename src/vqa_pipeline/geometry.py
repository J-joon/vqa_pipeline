from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Point:
    x: int
    y: int
    label: Optional[str] = None

@dataclass(frozen=True)
class Box:
    minimum: Point
    maximum: Point
    label: Optional[str] = None

    @property
    def width(self) -> float:
        return self.maximum.x - self.minimum.x

    @property
    def height(self) -> float:
        return self.maximum.y - self.minimum.y

@dataclass(frozen=True)
class Mask:
    mask: list[list[bool]]
    label: Optional[str] = None
