import os
from typing import Dict
import json
import torch
from PIL import Image

from lmms_engine.datasets.collator import LLaVACollator, VisionCollator, PackingCollator
from lmms_engine.datasets.iterable.multimodal_iterable_dataset import (
    MultiModalIterableDataset,
)
from lmms_engine.mapping_func import register_dataset
from lmms_engine.utils.train_utils import TrainUtilities


@register_dataset("vision_iterable")
class VisionSFTIterableDataset(MultiModalIterableDataset):
    def load_from_csv(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        """Load from CSV data directly without intermediate transformation."""
        images_list = []
        videos = []
        kwargs = {}

        # Build messages directly from CSV data
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": data["video"]}},
                    {"type": "text", "text": data["prompt"]},
                ],
            }
        ]

        # Process video content directly
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    images_list.append(content["image_url"]["url"])
                elif content["type"] == "video_url":
                    frames, sample_fps = self.load_videos(
                        content["video_url"]["url"],
                        data_folder=data_folder,
                        fps=self.config.fps,
                    )
                    videos.append(frames)
                    kwargs["fps"] = sample_fps

        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        if data_folder is not None:
            images = [Image.open(os.path.join(data_folder, image)) for image in images_list]
        else:
            images = [Image.open(image) for image in images_list]
        if len(images) == 0:
            images = None
        if len(videos) == 0:
            videos = None
        inputs = self.processor.process(images=images, hf_messages=hf_messages, videos=videos, **kwargs)
        return inputs

    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        images_list = []
        videos = []
        pil_image_flag = False
        kwargs = {}
        
        
        if "conversations" in data:
            # LLava format
            # {
            #   "image": "path/to/image.png", or list of "path/to/image.png"
            #   # "image": <PIL.Image.Image>, or list of <PIL.Image.Image>
            #   # "video": "path/to/video.mp4", or list of "path/to/video.mp4"
            #   "conversations": [
            #       {
            #           "from": "human",
            #           "value": "<image> 这张图里有什么？"
            #       },
            #       {
            #           "from": "gpt",
            #           "value": "这是一只狗在草地上玩耍。"
            #       }
            # }
            convs = data["conversations"]
            if isinstance(convs, str):
                if data_folder and not os.path.isabs(convs):
                    convs = os.path.join(data_folder, convs)
                with open(convs, 'r') as f:
                    convs = json.load(f)
            images = data.get("image", None)
            image_paths = None
            if images is not None:
                image_paths = []
                if not isinstance(images, list):
                    images = [images]
                if isinstance(images[0], str):
                    image_paths = images
                elif isinstance(images[0], Image.Image):
                    pil_image_flag = True
                    image_paths = [''] * len(images) # 占位符
            video_paths = data.get("video", None)
            messages = TrainUtilities.convert_llava_to_open(convs, image_paths=image_paths, video_paths=video_paths)
        elif "caption" in data:
            # LLava caption format
            # {
            #   "image": "path/to/image.png",
            #   # "image": <PIL.Image.Image>,
            #   "caption": "这是一只狗在草地上玩耍。",
            # }
            if isinstance(data["image"], Image.Image):
                pil_image_flag = True
            messages = TrainUtilities.convert_llava_image_caption_to_open(data["caption"], data.get("image", None))
        elif "messages" in data:
            messages = data["messages"]
        else:
            raise ValueError("JSON data must contain either 'conversations' or 'messages' field.")

        for message in messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    images_list.append(content["image_url"]["url"])
                elif content["type"] == "video_url":
                    # Loading videos with fps
                    frames, sample_fps = self.load_videos(
                        content["video_url"]["url"],
                        data_folder=data_folder,
                        fps=self.config.fps,
                    )
                    videos.append(frames)
                    # Update kwargs
                    kwargs["fps"] = sample_fps

        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        images = []
        if pil_image_flag:
            images = data["image"]
            if not isinstance(images, list):
                images = [images]
        else:
            if data_folder is not None:
                images = [Image.open(os.path.join(data_folder, image)) for image in images_list]
            else:
                images = [Image.open(image) for image in images_list]
        if len(images) == 0:
            images = None
        if len(videos) == 0:
            videos = None
        inputs = self.processor.process(images=images, hf_messages=hf_messages, videos=videos, **kwargs)
        return inputs

    def load_from_hf(self, data) -> Dict[str, torch.Tensor]:
        messages = data["messages"]
        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        if isinstance(data["image"], list):
            images = data["image"]
        else:
            images = [data["image"]]
        inputs = self.processor.process(images=images, hf_messages=hf_messages)
        return inputs

    def get_collator(self):
        if self.processor_config.processor_type == "llava":
            return LLaVACollator(self.processor)
        elif self.config.dataset_type == "spb2_vl_iterable" and self.config.packing:
            return PackingCollator(self.processor)
        else:
            return VisionCollator(self.processor)
