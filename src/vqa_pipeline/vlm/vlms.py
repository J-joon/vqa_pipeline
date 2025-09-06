from __future__ import annotations
from ._typeclass import VLM, T_Image
from vqa_pipeline.image import ImageProvider
from dataclasses import dataclass
from typing import Protocol, Optional, runtime_checkable, TypeVar
from static_error_handler import *
import torchvision.transforms as T
import torch
from PIL import Image
from torchvision.transforms.functional import InterpolationMode, to_pil_image
from transformers import AutoModel, AutoTokenizer, AutoConfig

import math
from static_error_handler import *
from accelerate import init_empty_weights, dispatch_model

#=== InternVL3 ===
@runtime_checkable
class LabelProvider(Protocol):
    frame_index: int
    camera_type: str

@runtime_checkable
class ImageLabelProvider(ImageProvider, LabelProvider, Protocol):
    pass

@dataclass
class InternVL3(VLM):
    model: object
    tokenizer: object
    generation_config: object
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    @staticmethod
    def build_transform(input_size):
        MEAN, STD = InternVL3.IMAGENET_MEAN, InternVL3.IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform
    
    @staticmethod
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    @staticmethod
    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = InternVL3.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    @staticmethod
    def load_image(pil_image, input_size=448, max_num=12):
        transform = InternVL3.build_transform(input_size=input_size)
        images = InternVL3.dynamic_preprocess(pil_image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    @staticmethod
    def split_model(model_path):
        device_map = {}
        world_size = torch.cuda.device_count()
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        num_layers = config.llm_config.num_hidden_layers
        # Since the first GPU will be used for ViT, treat it as half a GPU.
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        num_layers_per_gpu = [num_layers_per_gpu] * world_size
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1
        device_map['vision_model'] = 0
        device_map['mlp1'] = 0
        device_map['language_model.model.tok_embeddings'] = 0
        device_map['language_model.model.embed_tokens'] = 0
        device_map['language_model.output'] = 0
        device_map['language_model.model.norm'] = 0
        device_map['language_model.model.rotary_emb'] = 0
        device_map['language_model.lm_head'] = 0
        device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

        return device_map

    @staticmethod
    def create(path: str = 'OpenGVLab/InternVL3-14B') -> Result[VLM, str]:
        try:
            device_map = InternVL3.split_model(path)
            model = AutoModel.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=False,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
                device_map=device_map).eval()
            # If you set `load_in_8bit=True`, you will need two 80GB GPUs.
            # If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
            generation_config = dict(max_new_tokens=4096, do_sample=True)
            return Ok(InternVL3(model = model, tokenizer=tokenizer, generation_config = generation_config))
        except Exception as e:
            return Err(error=e)

    def question(self, images: list[ImageLabelProvider], prompts: tuple[tuple[str, str],...]) -> Result[tuple[tuple[str, str],...], str]:
        ls_pixel_values = [ InternVL3.load_image(image.image, max_num=12).to(torch.bfloat16).cuda() for image in images ]
        prefix = ''.join([ f"Frame-{image.frame_index}_{image.camera_type}" for image in images ])
        pixel_values = torch.cat(ls_pixel_values, dim=0)
        num_patches_list = [ pixel_values.size(0) for pixel_values in ls_pixel_values ]
        history = None
        frame_information = list()
        try:
            for key, prompt in prompts:
                question = prefix + prompt
                response, history = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config, num_patches_list=num_patches_list, history=history, return_history=True)
                frame_information.append((key, response))
            return Ok(tuple(frame_information))
        except Exception as e:
            return Err(e)
