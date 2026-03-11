import os
import re
import sys
import glob
import json
import random
import logging
import torch
import datasets
import transformers
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from tqdm import tqdm

from datasets import Dataset, DatasetDict
from transformers import AutoProcessor, set_seed, Qwen2VLForConditionalGeneration
from trl import SFTTrainer, ScriptArguments, TrlParser, SFTConfig, ModelConfig, get_peft_config, get_quantization_config
from qwen_vl_utils import process_vision_info

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MixedSFTScriptArguments(ScriptArguments):
    train_data_path: str = field(default="./Charades/charades_annotation/train.json")
    video_folder: str = field(default="./Charades/Charades_v1", metadata={"help": "原始视频文件夹"})
    video_image_dir: str = field(default="./Charades/Charades_v1_grid", metadata={"help": "TGrid预处理后的网格图文件夹"})
    mixing_ratio: float = field(default=0.5, metadata={"help": "TGrid 样本所占的比例"})

def parse_grid_config(path: str):
    """从路径名中解析 TGrid 配置，例如 'g43_f0.5_s3'"""
    folder_name = os.path.basename(path)
    result = {"grid_size": (4, 4), "stride": 1} # 默认值
    g_match = re.search(r"g(\d)(\d)", folder_name)
    if g_match: result["grid_size"] = (int(g_match.group(1)), int(g_match.group(2)))
    s_match = re.search(r"s(\d+)", folder_name)
    if s_match: result["stride"] = int(s_match.group(1))
    return result

def load_mixed_dataset(args):
    with open(args.train_data_path, "r") as f:
        data = json.load(f)
    
    examples = []
    grid_config = parse_grid_config(args.video_image_dir)
    
    for video_id, video_data in tqdm(data.items(), desc="Loading Data"):
        for timestamps, sentence in zip(video_data["timestamps"], video_data["sentences"]):
            example = {
                "video_id": video_id,
                "problem": sentence.strip().lower().rstrip("."),
                "solution": (timestamps[0], timestamps[1]),
                "video_path": os.path.join(args.video_folder, f"{video_id}.mp4"),
                "video_image_path": os.path.join(args.video_image_dir, video_id),
                "grid_config": grid_config,
                "duration": video_data["duration"]
            }
            # 过滤掉不存在的路径
            if os.path.exists(example["video_path"]) or os.path.exists(example["video_image_path"]):
                examples.append(example)

    random.shuffle(examples)
    return Dataset.from_list(examples)

QUESTION_TEMPLATE = "During which frames can we see [EVENT]? Answer strictly in the format of 'from x to y'."

def convert_to_messages(example, use_grid: bool):
    """核心转换逻辑：决定样本是 Base 还是 TGrid 表现形式"""
    content = []
    
    if use_grid and os.path.exists(example["video_image_path"]):
        # --- TGrid 模式 ---
        image_paths = sorted(glob.glob(os.path.join(example["video_image_path"], "*.png")))
        cols, rows = example["grid_config"]["grid_size"]
        stride = example["grid_config"]["stride"]
        frames_per_grid = cols * rows
        
        last_frame = 0
        for idx, img_path in enumerate(image_paths):
            start = idx * stride
            end = start + frames_per_grid - 1
            m = re.search(r"partial_(\d+)", img_path)
            if m: end = int(m.group(1))
            last_frame = max(last_frame, end)
            
            content.append({"type": "text", "text": f"Frame {start} to Frame {end}"})
            content.append({"type": "image", "image": img_path})
        
        content.append({"type": "text", "text": f"This video has {last_frame} frames. " + QUESTION_TEMPLATE.replace("[EVENT]", example["problem"])})
        print(content)
    else:
        # --- Base 模式 ---
        content = [
            {"type": "video", "video": example["video_path"], "fps": 1.0},
            {"type": "text", "text": QUESTION_TEMPLATE.replace("[EVENT]", example["problem"])}
        ]

    st, ed = example["solution"]
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": f"from {round(st)} to {round(ed)}"}
    ]

def collate_fn(examples, processor, mixing_ratio):
    batch_messages = []
    for ex in examples:
        # 按照设定比例决定当前样本采用哪种模式
        use_grid = random.random() < mixing_ratio
        batch_messages.append(convert_to_messages(ex, use_grid))

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in batch_messages
    ]
    
    image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages, return_video_kwargs=True)
    
    batch = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        fps=video_kwargs.get("fps"),
        return_tensors="pt",
        padding=True,
    )

    # 屏蔽 vision tokens 和 pad tokens 的梯度
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    for token in [processor.image_token, processor.video_token]:
        token_id = processor.tokenizer.convert_tokens_to_ids(token)
        labels[labels == token_id] = -100
    batch["labels"] = labels
    
    return batch

def main():
    parser = TrlParser((MixedSFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    set_seed(training_args.seed)
    
    # 1. 加载模型与 Processor
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    
    # 检查是否使用了 DeepSpeed ZeRO-3
    # is_deepspeed_zero3 = (
    #     training_args.deepspeed and 
    #     "zero_stage": 3 in open(training_args.deepspeed).read()
    # ) if training_args.deepspeed else False
    is_deepspeed_zero3 = True
    # 如果是 ZeRO-3，device_map 必须为 None
    device_map = None if is_deepspeed_zero3 else "auto"
    # 明确告诉 Trainer 不要寻找 'text' 列，跳过预处理
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True, 
    }

    # 确保不移除 collate_fn 需要的列
    training_args.remove_unused_columns = False
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        attn_implementation=model_args.attn_implementation,
        device_map=device_map, # 修改这里
        trust_remote_code=True,
    )
    
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     model_args.model_name_or_path,
    #     torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
    #     attn_implementation=model_args.attn_implementation,
    #     device_map="auto"
    # )

    # 2. 加载数据
    train_dataset = load_mixed_dataset(script_args)

    # 3. 训练配置
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda x: collate_fn(x, processor, script_args.mixing_ratio),
        peft_config=get_peft_config(model_args),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

if __name__ == "__main__":
    main()