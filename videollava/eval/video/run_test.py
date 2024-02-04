import math
import os
import argparse
import json
import requests
import pandas as pd
import io

import torch
import transformers
from tqdm import tqdm
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.constants import DEFAULT_VIDEO_TOKEN, VIDEO_TOKEN_INDEX
from videollava.mm_utils import get_model_name_from_path, tokenizer_video_token, KeywordsStoppingCriteria
from videollava.model.builder import load_pretrained_model
from videollava.model.language_model.llava_llama import LlavaLlamaForCausalLM


URL = "https://docs.google.com/spreadsheets/d/1AIwpV-VLAJ4tHKtcQXMdfTYuqwvSGn1MUSIjYC5ll0E/export?format=csv&gid={gid}"


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--cache_dir', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)

    return parser.parse_args()

def get_model_output(model, video_processor, tokenizer, video, qs, args):
    #if model.config.mm_use_im_start_end:
    #    qs = DEFAULT_X_START_TOKEN['VIDEO'] + ''.join([DEFAULT_IMAGE_TOKEN]*8) + DEFAULT_X_END_TOKEN['VIDEO'] + '\n' + qs
    #else:
    #qs = ''.join([DEFAULT_IMAGE_TOKEN]*8) + '\n' + qs
    #qs = ''.join(DEFAULT_VIDEO_TOKEN) + '\n' + qs
    qs = f"{qs}\n{DEFAULT_VIDEO_TOKEN}"

    conv_mode = "lvm"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    #print(prompt)


    n_frames = 50
    video_tensor = video_processor([video], n_frames)['pixel_values'][0].half().to(args.device)
    input_ids = tokenizer_video_token(prompt, tokenizer, VIDEO_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            videos=[video_tensor],
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print(outputs)
    return outputs


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    model = model.to(args.device)


    gid = "496919303"
    videos_folder = "videos"
    res = requests.get(url=URL.format(gid=gid))
    assert res.status_code == 200, res.status_code
    df = pd.read_csv(io.BytesIO(res.content))

    prompts = []
    for _, row in df.iterrows():
        n_questions = row['Number of Questions']
        yt_id = row['Video'].split('?v=')[-1]
        video_path = os.path.join(videos_folder, gid, f"{yt_id}.mp4")
        for i in range(n_questions):
            question = row[f"Question {i + 1}"]
            prompts.append({'id': yt_id, 'video_path': video_path, 'question': question})
            print(prompts[-1])
    print(f"Found {len(prompts)} prompts")
     
    index = 0
    for prompt in tqdm(prompts):
        print(prompt)
        output = get_model_output(model, processor, tokenizer, prompt['video_path'], prompt['question'], args)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
