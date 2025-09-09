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
            model = AutoProcessor.from_pretrained(self.model_id)
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
        def run(processor: T_Processor) -> Result[tuple[Box, ...], str]:
            try:
                inputs = processor(
                    images=[image.image],
                    text=query,
                    return_tensors="pt"
                    ).to(device)
                def run_model(model: T_Model) -> Result[tuple[Box, ...], str]:
                    try:
                        with torch.no_grad():
                            outputs = model(**inputs)
                        results = processor.post_process_grounded_object_detection(
                            outputs,
                            inputs.input_ids,
                            threshold=self.threshold,
                            text_threshold=self.text_threshold,
                            target_sizes=[image.image.size[::-1]],
                        )
                        result = results[0]
                        boxes = ( Box.from_list(box, label) for box, label in zip(result["boxes"], result["labels"]) )
                        return self.model.and_then(run_model)
                    except Exception as e:
                        return Err(f"run_model: {e}")
                return self.model.and_then(run_model)
            except Exception as e:
                return Err(f"run: {e}")
        return self.processor.and_then(run)
