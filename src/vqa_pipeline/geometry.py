from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from numpy.typing import NDArray

@dataclass(frozen=True)
class Point:
    x: int
    y: int
    label: Optional[str] = None

    def to_list(self) -> list[int]:
        return [self.x, self.y]

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

    # return value must have shape (4,)
    def to_numpy(self) -> NDArray[np.float32]:
        return np.array([self.minimum.x, self.minimum.y, self.maximum.x, self.maximum.y], dtype=np.float32)

    @staticmethod
    def from_list(box: tuple[float, float, float, float], label: Optional[str]) -> Box:
        return Box(
                minimum = Point(
                    x = box[0],
                    y = box[1],
                ),
                maximum = Point(
                    x = box[2],
                    y = box[3],
                    ),
                label = label,
                )

@dataclass(frozen=True)
class Mask:
    mask: list[list[bool]]
    label: Optional[str] = None
