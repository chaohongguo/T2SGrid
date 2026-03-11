import json
import os
import re
import peft
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)


def load_expanded_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    expanded_items = []
    for k, v in data.items():
        timestamps = v["timestamps"]
        sentences = v["sentences"]
        for t, s in zip(timestamps, sentences):
            expanded_items.append((k, t, s))  # 记录视频ID、时间戳、句子
    return expanded_items


def load_jsonl_data(jsonl_path):
    with open(jsonl_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search("[ABCDEFG]", s):
        return ""

    matches = re.search(r"[ABCDEFG]", s)
    if matches is None:
        return ""
    return matches[0]


def load_qwenvl_model(args):
    """
    Load Qwen VL models (Qwen2-VL, Qwen2.5-VL, Qwen3-VL) with optional LoRA fine-tuning.
    Supports loading from a model path that already has LoRA merged.

    Args:
        args: argparse.Namespace with attributes:
            - model_name: str, one of ["qwen2vl", "qwen2.5vl", "qwen3vl", "llava_ov_15"]
            - model_path: Optional[str], path to model (can include merged LoRA)
            - lora_path: Optional[str], path to LoRA weights if not merged

    Returns:
        model: Qwen model (torch.nn.Module)
        processor: corresponding AutoProcessor
    """
    model_name = args.model_name.lower()
    print(f"Load model {model_name}")

    # 优先使用用户指定的 model_path
    base_model_path = getattr(args, "model_path", None)
    lora_path = getattr(args, "lora_path", None)

    if not base_model_path:
        # 如果没指定 model_path，根据模型名选择默认路径
        if model_name == "qwen2vl":
            base_model_path = "Qwen/Qwen2-VL-7B-Instruct"
            model_class = Qwen2VLForConditionalGeneration
            device_map = "auto"
        elif model_name == "qwen2.5vl":
            from transformers import Qwen2_5_VLForConditionalGeneration

            base_model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
            model_class = Qwen2_5_VLForConditionalGeneration
            device_map = "auto"
        elif model_name == "qwen3vl":
            from transformers import Qwen3VLForConditionalGeneration

            base_model_path = (
                "Qwen/Qwen3-VL-8B-Instruct"
            )
            model_class = Qwen3VLForConditionalGeneration
            device_map = "cuda:0"
        elif model_name == "llava_ov_15":
            base_model_path = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
            model_class = AutoModelForCausalLM
            device_map = "cuda:0"

        else:
            raise ValueError(f"Unknown model name: {args.model_name}")
    else:
        # 如果指定了 model_path，根据模型名选择 class
        if model_name == "qwen2vl":
            model_class = Qwen2VLForConditionalGeneration
            device_map = "auto"
        elif model_name == "qwen2.5vl":
            from transformers import Qwen2_5_VLForConditionalGeneration

            model_class = Qwen2_5_VLForConditionalGeneration
            device_map = "cuda:0"
        elif model_name == "qwen3vl":
            from transformers import Qwen3VLForConditionalGeneration

            model_class = Qwen3VLForConditionalGeneration
            device_map = "cuda:0"
        elif model_name == "llava_ov_15":
            model_class = AutoModelForCausalLM
            device_map = "cuda:0"
        else:
            raise ValueError(f"Unknown model name: {args.model_name}")

    print(f"Loading model from: {base_model_path}")
    model = model_class.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    # 仅当 model_path 没有包含 LoRA 时，再加载 lora_path
    if lora_path:
        print(f"Loading LoRA weights from: {lora_path}")
        model = peft.PeftModel.from_pretrained(model, lora_path)

    processor = AutoProcessor.from_pretrained(
        base_model_path, use_fast=True, trust_remote_code=True
    )
    return model, processor


def get_prompt(prompt_type):
    if prompt_type == "numpro":
        prompt = """During which frames can we see {}? Answer strictly in the format of 'from x to y'."""
    elif prompt_type == "hd":
        prompt = "Please find the highlight contents in the video described by the query {}. Determine the highlight frames and its saliency score on a scale from 1 to 5. If the video content more related to the query, the saliency score should be higher. The output format should be like: 'The highlight frames are in the 80, 81, 82, 83, 84, 85, 86, 87, 88, 89 frames. Their saliency scores are 1.3, 1.5, 2.6, 3.0, 2.9, 4.0, 3.7, 3.2, 2.1, 2.3'."
    elif prompt_type == "mo":
        prompt = (
            "To accurately pinpoint the event '{}' in the video,"
            " determine the precise timeperiod of the event."
            " Provide the start and end times in the format: 'From x to y'."
        )
    elif prompt_type == "qa":
        prompt = """Answer the question: "[QUESTION]" according to the content of the video. Select the answer from: [OPTION].only provide your answer within the <answer> </answer> tags, output the corresponding letter of the option. No other text."""
    return prompt


def parser_timestamps(output_text):
    patterns = [
        # from frame 12 to frame 34   / from 12 to 34 / from 12s to 34s
        r"\bfrom\s*(?:frame[s]?\s*)?(\d+(?:\.\d+)?)(?:s)?\s*to\s*(?:frame[s]?\s*)?(\d+(?:\.\d+)?)(?:s)?\b",
        # frame 12 to frame 34  / frames 12 to 34 / frame 12 to 34
        r"\b(?:frame[s]?\s*)(\d+(?:\.\d+)?)(?:s)?\s*(?:to|and)\s*(?:frame[s]?\s*)?(\d+(?:\.\d+)?)(?:s)?\b",
        # between frames 12 and 34  / between frames 12-34
        r"\bbetween\s*frames?\s*(\d+(?:\.\d+)?)(?:s)?\s*(?:and|-)\s*(\d+(?:\.\d+)?)(?:s)?\b",
        # 12-34 or 12 - 34
        r"\b(\d+(?:\.\d+)?)(?:s)?\s*-\s*(\d+(?:\.\d+)?)(?:s)?\b",
        # 12 to 34  (general fallback)
        r"\b(\d+(?:\.\d+)?)(?:s)?\s*(?:to|and)\s*(\d+(?:\.\d+)?)(?:s)?\b",
    ]

    for p in patterns:
        m = re.search(p, output_text, re.IGNORECASE)
        if m:
            start = int(float(m.group(1)))
            end = int(float(m.group(2)))
            # 保证顺序正确
            start, end = min(start, end), max(start, end)
            return [start, end]

    return [0, 0]


def parse_video_grid_dir(s: str):
    result = {}

    # grid g43 -> (4, 3)
    g_match = re.search(r"g(\d)(\d)", s)
    if g_match:
        result["grid_size"] = (int(g_match.group(1)), int(g_match.group(2)))

    # fps f0.5 -> 0.5
    f_match = re.search(r"f([\d.]+)", s)
    if f_match:
        result["fps"] = float(f_match.group(1))

    # s3 -> 3
    s_match = re.search(r"s(\d+)", s)
    if s_match:
        result["stride"] = int(s_match.group(1))

    # r1 -> 1
    r_match = re.search(r"r(\d+)", s)
    if r_match:
        result["r"] = int(r_match.group(1))

    return result


def compute_iou(span, solution):
    a, b = span
    c, d = solution

    inter = max(0.0, min(b, d) - max(a, c))
    union = max(b, d) - min(a, c)

    return inter / union if union > 0 else 0.0


def save_metrics(metrics, metric_path):
    os.makedirs(os.path.dirname(metric_path), exist_ok=True)
    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"✅ Metrics saved to: {metric_path}")
