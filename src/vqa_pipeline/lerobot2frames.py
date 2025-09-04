from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torchvision.transforms.functional import to_pil_image
import tyro

@runtime_checkable
class Config(Protocol):
    repo_id: str
    episode_index: int
    output_dir: Path
    image_column: str

@dataclass(frozen=True)
class ConfigImpl(Config):
    repo_id: str = tyro.MISSING
    episode_index: int = tyro.MISSING
    output_dir: Path = tyro.MISSING
    image_column: str = tyro.MISSING

def main(config: Config):
    dataset = LeRobotDataset(config.repo_id, episodes=[config.episode_index])
    for frame in dataset:
        frame_index = frame["frame_index"].item()
        image = to_pil_image(frame[config.image_column])
        output_path = config.output_dir / f"{frame_index:05d}.jpeg"
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(output_path, "JPEG")


def entrypoint():
    _CONFIGS = {
            "aloha_sim_insertion_scripted": (
                "aloha sim insertion scripted",
                ConfigImpl(
                    repo_id = "J-joon/sim_insertion_scripted",
                    episode_index = tyro.MISSING,
                    output_dir = tyro.MISSING,
                    image_column = "observation.images.top",
                    )
                ),
            "aloha_sim_transfer_cube_scripted": (
                "aloha sim cube transfer scripted",
                ConfigImpl(
                    repo_id = "J-joon/sim_transfer_cube_scripted",
                    episode_index = tyro.MISSING,
                    output_dir = tyro.MISSING,
                    image_column = "observation.images.top",
                    )
                ),
            "libero": (
                "libero",
                ConfigImpl(
                    repo_id = "physical-intelligence/libero",
                    episode_index = tyro.MISSING,
                    output_dir = tyro.MISSING,
                    image_column = "image",
                )
                ),
            "custom": (
                "custom",
                ConfigImpl()
                ),
            }
    config = tyro.extras.overridable_config_cli(_CONFIGS)
    main(config)

if __name__=="__main__":
    entrypoint()
