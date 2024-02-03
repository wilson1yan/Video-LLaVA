from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM,\
    LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from llamabpt.pyt.vqgan import VQGANModel

VISION_START = torch.tensor([529, 4924, 29958], dtype=torch.long)
VISION_END = torch.tensor([1533, 4924, 29958], dtype=torch.long)
EOF_TOKEN_INDEX = 8192
EOV_TOKEN_INDEX = 8193
VIDEO_TOKEN_INDEX = -200
IGNORE_INDEX = -100


class LongVisionMetaModel:
    def __init__(self, config):
        super(LongVisionMetaModel, self).__init__(config)
        self.vqgan = VQGANModel()
        self.vision_embed_tokens = nn.Embedding(config.num_vision_codes, config.hidden_size)

    def encode_videos(self, videos):
        video_features = []
        for video in videos:
            video_codes = self.model.vqgan.get_code(video)
            video_feat = self.model.vision_embed_tokens(video_codes)
            video_features.append(video_feat)
        return video_feataures


class LongVisionMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def encode_videos(self, videos):
        video_features = []
        for video in videos:
            device = video.device
            video_codes = self.model.vqgan.get_code(video).flatten(start_dim=1).cpu().numpy() # T(HW)
            new_video_codes = []
            for i, vc in enumerate(video_codes):
                vc = vc.tolist()
                new_video_codes.extend(vc)
                if i == len(video_codes) - 1:
                    new_video_codes.append(EOV_TOKEN)
                else:
                    new_video_codes.append(EOF_TOKEN)
            new_video_codes = torch.LongTensor(np.array(new_video_codes, dtype=np.int32)).to(device)
            video_feat = self.model.vision_embed_tokens(new_video_codes)
            video_features.append(video_feat)
        return video_features

    def prepare_inputs_for_multimodal(
        self, input_ids, attention_mask, past_key_values, videos
    ):
        if videos is None or input_ids.shape[1] == 1:
            if past_key_values is not None and videos is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None 

        video_features = self.encode_videos(videos)

        device = self.get_model().embed_tokens.weight.device
        new_input_embeds = []
        cur_video_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            assert (cur_input_ids == VIDEO_TOKEN_INDEX).sum() > 0
            video_token_indices = torch.where(cur_input_ids == VIDEO_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            while video_token_indices.numel() > 0:
                cur_video_features = video_features[cur_video_idx]
                video_token_start = video_token_indices[0]
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:video_token_start]))
                cur_new_input_embeds.append(self.get_model().embed_tokens(VISION_START.to(device)))
                cur_new_input_embeds.append(cur_video_features)
                cur_new_input_embeds.append(self.get_model().embed_tokens(VISION_END.to(device)))
                cur_video_idx += 1
                cur_input_ids = cur_input_ids[video_token_start+1:]
                video_token_indices = torch.where(cur_input_ids == VIDEO_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)

        return None, attention_mask, past_key_values, new_input_embeds


class LongVisionConfig(LlamaConfig):
    model_type = "lvm"
    num_vision_codes = 8448


class LongVisionLlamaModel(LongVisionMetaModel, LlamaModel):
    config_class = LongVisionConfig

    def __init__(self, config: LlamaConfig):
        super(LongVisionLlamaModel, self).__init__(config)


class LongVisionLlamaForCausalLM(LlamaForCausalLM, LongVisionMetaForCausalLM):
    config_class = LongVisionConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LongVisionLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vision_head = nn.Linear(config.hidden_size, config.num_vision_codes, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        videos: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        assert labels is None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds = self.prepare_inputs_for_multimodal(input_ids, attention_mask, past_key_values, videos)
        attention_mask = None # TODO

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        # vision_logits = self.vision_head(hidden_states) # TODO

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "videos": kwargs.get("videos", None),
            }
        )
        return model_inputs

AutoConfig.register("lvm", LongVisionConfig)
AutoModelForCausalLM.register(LongVisionConfig, LongVisionLlamaForCausalLM)

