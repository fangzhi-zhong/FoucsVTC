import os
from typing import Dict, Tuple

from io import BytesIO
import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import fetch_video
import json

from lmms_engine.datasets.collator import VisionCollator
from lmms_engine.datasets.iterable.vision_iterable_dataset import (
    VisionSFTIterableDataset,
)
from lmms_engine.mapping_func import register_dataset
from lmms_engine.utils.train_utils import TrainUtilities


@register_dataset("qwen3_vl_iterable")
class Qwen3VLIterableDataset(VisionSFTIterableDataset):

    def _get_user_prompt_suffix(self):
        """Return the configured suffix, optionally sampled per example.

        The hook is intentionally checked dynamically so the Qwen3.5 subclass
        can add random grounding prompts without changing the Qwen3-VL data
        behavior.
        """
        if (
            getattr(self.config, "random_grounding_prompt", False)
            and hasattr(self, "_random_grounding_prompt")
        ):
            return self._random_grounding_prompt()
        return self.config.user_prompt_suffix

    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        images_list = []
        videos = []
        pil_image_flag = False
        dict_image_flag = False
        kwargs = {}
        
        
        if "conversations" in data:
            # LLava format
            # {
            #   "image": "path/to/image.png", or list of "path/to/image.png"
            #   # "image": <PIL.Image.Image>, or list of <PIL.Image.Image>
            #   image = {
                #     "bytes": b"...PNG binary...",
                #     "path": "xxx.png"   # 可能有，也可能没有
                # }
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
            
            # 如果convs是字符串，则认为是json文件路径，需要读取json文件
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
                # print(f"images_len:{len(images)}")
                # print(f"type of image:{type(images[0])}")
                if isinstance(images[0], str):
                    image_paths = images
                elif isinstance(images[0], Image.Image):
                    pil_image_flag = True
                    image_paths = [''] * len(images) # 占位符
                elif isinstance(images[0], dict) and "bytes" in images[0]:
                    dict_image_flag = True
                    image_paths = [''] * len(images)
                else:
                    raise ValueError("Unsupported image format in data.")


            video_paths = data.get("video", None)
            # if image_paths is not None:
            #     print(f"image_paths_size:{len(image_paths)}")
            # else:
            #     print("no image_paths")
            messages = TrainUtilities.convert_llava_to_open(convs, image_paths=image_paths, video_paths=video_paths)
            # print(f"openai:{messages}")
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
                    frames, video_metadata, sample_fps = self.load_videos(
                        content["video_url"]["url"],
                        data_folder=data_folder,
                        fps=self.config.fps,
                    )
                    videos.append(frames)
                    # Update kwargs
                    kwargs["fps"] = sample_fps
                    kwargs["video_metadata"] = video_metadata
                    kwargs["do_sample_frames"] = False


        hf_messages = TrainUtilities.convert_open_to_hf(
            messages,
            user_prompt_suffix=self._get_user_prompt_suffix(),
        )
        images = []

        if dict_image_flag:
            images_dict_list = data["image"]
            if not isinstance(images_dict_list, list):
                images_dict_list = [images_dict_list]
            for image in images_dict_list:
                images.append(Image.open(BytesIO(image["bytes"])))
        elif pil_image_flag:
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
        # print("images:", images)
        # print("hf_messages:", hf_messages)
        inputs = self.processor.process(
            images=images,
            hf_messages=hf_messages,
            videos=videos,
            add_system_prompt=self.config.add_system_prompt,
            think_mode=self.config.think_mode,
            **kwargs,
        )
        return inputs

    def load_videos(self, video_path: str, data_folder=None, fps: int = 1):
        assert (
            self.config.video_backend == "qwen_vl_utils"
        ), "Qwen3VLIterableDataset only supports qwen_vl_utils backend"
        frames, video_metadata, sample_fps = self.load_video_qwen_vl_utils(video_path, fps)
        return frames, video_metadata, sample_fps

    def load_video_qwen_vl_utils(
        self,
        video_path: str,
        fps: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Load video using Qwen VL utils.

        Args:
            video_path: Path to video file
            fps: Target frames per second

        Returns:
            Tuple of (video frames, video metadata, sample fps)
        """
        video_dict = {
            "type": "video",
            "video": f"file://{video_path}",
            "min_frames": 1,
            "max_pixels": self.config.video_max_pixels,
            "max_frames": self.config.video_max_frames,
            "min_pixels": self.config.video_min_pixels,
        }

        if self.config.video_sampling_strategy == "frame_num":
            n_frames = self.config.frame_num
            video_dict["nframes"] = n_frames
            video_inputs, sample_fps = fetch_video(video_dict, return_video_sample_fps=True, return_video_metadata=True)
            frames, video_metadata = video_inputs
            frames = frames.numpy()
            return frames, video_metadata, sample_fps
        elif self.config.video_sampling_strategy == "fps":
            video_dict["fps"] = fps
            video_inputs, sample_fps = fetch_video(video_dict, return_video_sample_fps=True, return_video_metadata=True)
            frames, video_metadata = video_inputs
            frames = frames.numpy()
            return frames, video_metadata, sample_fps
        else:
            raise ValueError(f"Invalid video sampling strategy: {self.config.video_sampling_strategy}")
