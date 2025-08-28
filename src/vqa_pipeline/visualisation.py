from dataclasses import dataclass, field
from typing import NewType, Optional, Callable, TypeAlias, Protocol
from static_error_handler import *
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .geometry import Point, Box, Mask

@dataclass(frozen=True)
class Recipe:
    boxes: Optional[list[Box]] = None
    points: Optional[list[Point]] = None
    masks: Optional[list[Mask]] = None

class RecipeProvider(Protocol):
    def __call__(self, episode_index: int, frame_index: int) -> Recipe: ...

def draw_box(image: Image.Image, box: Box, color: str = "red", width: int = 2, font_size: int = 14) -> Image.Image:
    draw = ImageDraw.Draw(image)

    # Draw rectangle
    draw.rectangle(
        [box.minimum.x, box.minimum.y, box.maximum.x, box.maximum.y],
        outline=color,
        width=width
    )

    # Draw label if exists
    if box.label:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        # ✅ Use textbbox instead of textsize
        bbox = draw.textbbox((0, 0), box.label, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        text_x, text_y = box.minimum.x, box.minimum.y - text_h - 2

        # Background for readability
        draw.rectangle(
            [text_x, text_y, text_x + text_w, text_y + text_h],
            fill=color
        )
        draw.text((text_x, text_y), box.label, fill="white", font=font)

    return image

def draw_point(image: Image.Image, point: Point, color: str = "blue", radius: int = 3) -> Image.Image:
    """
    Draw a point (as a filled circle) on a PIL image.

    :param image: PIL Image to draw on
    :param point: Point object
    :param color: Color of the point
    :param radius: Radius of the circle
    :return: Image with the point drawn
    """
    draw = ImageDraw.Draw(image)
    left_up = (point.x - radius, point.y - radius)
    right_down = (point.x + radius, point.y + radius)
    draw.ellipse([left_up, right_down], fill=color)
    if point.label is not None:
        # 폰트 설정 (기본 폰트 사용)
        try:
            font = ImageFont.load_default()
        except:
            font = None
        draw.text((point.x + radius + 2, point.y - radius), point.label, fill=color, font=font)
    return image


def draw_mask(image: Image.Image, mask: Mask, color: str = "green", alpha: int = 100) -> Image.Image:
    """
    Overlay a semi-transparent mask on a PIL image.

    :param image: PIL Image to draw on
    :param mask: Mask object containing a 2D boolean list
    :param color: Color of the mask
    :param alpha: Opacity (0-255)
    :return: Image with the mask applied
    """
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    height = len(mask.mask)
    width = len(mask.mask[0]) if height > 0 else 0

    min_x, min_y = width, height
    max_x, max_y= 0, 0

    for y in range(height):
        for x in range(width):
            if mask.mask[y][x]:
                overlay_draw.point((x, y), fill=(0, 255, 0, alpha) if color == "green" else color)
                min_x, min_y = min(min_x, x), min(min_y, y)
                max_x, max_y = max(max_x, x), max(max_y, y)
    result = Image.alpha_composite(image.convert("RGBA"), overlay)
    if mask.label:
        draw = ImageDraw.Draw(result)
        try:
            font = ImageFont.load_default()
        except:
            font = None
        draw.text((min_x, min_y - 10), mask.label, fill=color, font=font)

    # Merge overlay with original image
    return result

def draw_recipe(image: Image.Image, recipe: Recipe)->Image.Image:
    if recipe.boxes is not None:
        for box in recipe.boxes:
            image = draw_box(image, box)
    if recipe.points is not None:
        for point in recipe.points:
            image = draw_point(image, point)
    if recipe.masks is not None:
        for mask in recipe.masks:
            image = draw_mask(image, mask)
    return image

def pil_images_to_video(images: list[Image.Image], output_path: Path, fps=30) -> Result[None, str]:
    if not images:
        return Err("Image list is empty.")

    # Convert first image to get dimensions
    first_frame = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
    height, width, _ = first_frame.shape

    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 'mp4v' for .mp4
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img in images:
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()
    return Ok(None)
