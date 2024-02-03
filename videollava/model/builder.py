#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil
import numpy as np
import decord

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo, CenterCropVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample

#from videollava.model import *
from videollava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, \
    DEFAULT_VIDEO_PATCH_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN
from videollava.model.lvm.vision_llama import LongVisionLlamaForCausalLM


def load_pretrained_model(model_path, model_base, model_name, device_map="auto", device="cuda", **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}
    kwargs['torch_dtype'] = torch.float16

    model = LongVisionLlamaForCausalLM.from_pretrained(model_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    transform = Compose(
        [
            Lambda(lambda x: x.float() / 127.5 - 1),
            ShortSideScale(size=256),
            CenterCropVideo(256),
        ]
    )
    def video_processor(video_paths, n_frames):
        decord.bridge.set_bridge('torch')
        videos = []
        for video_path in video_paths:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            duration = len(vr)
            if duration <= n_frames:
                frame_id_list = list(range(duration))
            else:
                frame_id_list = np.linspace(0, duration - 1, n_frames, dtype=int)
            video_data = vr.get_batch(frame_id_list)
            video_data = video_data.permute(3, 0, 1, 2)
            video_data = transform(video_data) # THWC -> CTHW
            video_data = video_data.permute(1, 0, 2, 3) # CTHW -> TCHW)
            videos.append(video_data)
        return dict(pixel_values=videos)
    
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, video_processor, context_len
