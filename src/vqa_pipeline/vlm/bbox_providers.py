from __future__ import annotations
from ._typeclass import BBoxProvider, T_Image
from vqa_pipeline.image import ImageProvider
from vqa_pipeline import Box
from dataclasses import dataclass
from typing import Literal, Any, TypeAlias
from static_error_handler import Ok, Err, Result
from functools import cache
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
)
import torch
T_Model: TypeAlias = Any
T_Processor: TypeAlias = Any
@dataclass(frozen=True)
class GroundingDino(BBoxProvider):
    model_id: str = "IDEA-Research/grounding-dino-base"
    device: Literal["cuda", "cpu"] = "cuda"
    threshold: float = 0.3
    text_threshold: float = 0.4

    @property
    @cache
    def model(self, ) -> Result[T_Model, str]:
        try:
            model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)
            return Ok(model)
        except Exception as e:
            return Err(str(e))

    @property
    @cache
    def processor(self, ) -> Result[T_Processor, str]:
        try:
            processor = AutoProcessor.from_pretrained(self.model_id)
            return Ok(processor)
        except Exception as e:
            return Err(str(e))

    def query(self, image: T_Image, query: str) -> Result[tuple[Box, ...], str]:
        device = self.device
        image = image.image
        def run(processor: T_Processor) -> Result[tuple[Box, ...], str]:
            try:
                try:
                    inputs = processor(
                        images=image,
                        text=query,
                        return_tensors="pt"
                        ).to(device)
                except Exception as e:
                    return Err(f"processor: {e}")
                def run_model(model: T_Model) -> Result[tuple[Box, ...], str]:
                    try:
                        with torch.no_grad():
                            outputs = model(**inputs)
                    except Exception as e:
                        return Err(f"fail during inference of Grounding Dino: {e}")
                    try:
                        results = processor.post_process_grounded_object_detection(
                            outputs,
                            inputs.input_ids,
                            threshold=self.threshold,
                            text_threshold=self.text_threshold,
                            target_sizes=[image.size[::-1]],
                        )
                    except Exception as e:
                        return Err(f"fail during processing result: {e}")
                    try:
                        result = results[0]
                        boxes = ( Box.from_list(box, label) for box, label in zip(result["boxes"], result["labels"]) )
                        return self.model.and_then(run_model)
                    except Exception as e:
                        return Err(f"run_model: {e}")
                return self.model.and_then(run_model)
            except Exception as e:
                return Err(f"run: {e}")
        return self.processor.and_then(run)
