import glob
import os
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import sklearn
import json
import argparse
import torch

from tqdm import tqdm
from qwen_vl_utils import process_vision_info

from eval.util import (
    load_expanded_data,
    load_qwenvl_model,
    get_prompt,
    parser_timestamps,
    parse_video_grid_dir,
    compute_iou,
    save_metrics,
)
from data.data_config import DATASETS


def build_messages(video_grid_dir, vid, prompt, grid_size, stride):
    image_dir = f"{video_grid_dir}/{vid}"
    # print(image_dir)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))

    contents = []
    cols, rows = grid_size
    frames_per_grid = cols * rows
    for idx, url in enumerate(image_paths):
        start = idx * stride
        end = start + frames_per_grid - 1
        m = re.search(r"partial_(\d+)", url)
        if m:
            end = int(m.group(1))

        if frames_per_grid == 1:
            assert start == end
            contents.append({"type": "text", "text": f"<Frame {start}>"})
        else:
            contents.append({"type": "text", "text": f"<Frame {start} to Frame {end}>"})

        contents.append({"type": "image", "image": url})

    # print(contents)
    contents.append(
        {"type": "text", "text": f"This video has {end} frames in total." + prompt}
    )
    messages = [{"role": "user", "content": contents}]
    return messages


def get_finall_answer_single(
    video_grid_dir, vid, sentence, model, processor, args, preprocess_config
):

    model_type = args.model_name
    print(model.__class__)
    prompt = get_prompt(args.prompt_type).format(sentence)
    messages = build_messages(
        video_grid_dir,
        vid,
        prompt,
        grid_size=preprocess_config["grid_size"],
        stride=preprocess_config["stride"],
    )

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if model_type == "qwen3vl":
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages],
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True,
        )
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None
    else:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        video_metadatas = None

    if "qwen" in model_type:
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            video_metadata=video_metadatas,
            **video_kwargs,
        ).to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        second_per_grid_ts = inputs.pop("second_per_grid_ts", None)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return parser_timestamps(output_text), output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen2vl")
    parser.add_argument("--dataset", type=str, default="charades")
    parser.add_argument("--splits", type=str, default="test")
    parser.add_argument("--prompt_type", type=str, default="numpro")
    parser.add_argument(
        "--video_grid_dir", type=str, default="g43_f1.0_s6_r1_l0_ts336x336"
    )
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--sub", type=str, default=None)
    args = parser.parse_args()

    # 1. 读取json文件
    data_json_path = DATASETS[args.dataset]["splits"][args.splits]["annotation_file"]
    data = load_expanded_data(data_json_path)

    # 2. 构建模型
    model, processor = load_qwenvl_model(args)

    # 3. 获得result_apth 和metric_path
    result_dir = DATASETS[args.dataset]["splits"][args.splits]["result_dir"]
    if args.lora_path:
        lora_suffix = args.lora_path.split("/")[-1]
        result_path = f"{result_dir}/{args.model_name}/{args.video_grid_dir}_{lora_suffix}{args.sub}.jsonl"
        metric_path = f"{result_dir}/{args.model_name}/{args.video_grid_dir}_{lora_suffix}{args.sub}_metric.jsonl"
    else:
        # fmt:off
        lora_suffix = None
        result_path = f"{result_dir}/{args.model_name}/{args.video_grid_dir}.jsonl"
        metric_path = f"{result_dir}/{args.model_name}/{args.video_grid_dir}_metric.jsonl"
        # fmt:on

    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    # 4. 读取已完成样本（支持断点续跑）
    done_set = set()
    if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
        with open(result_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                done_set.add((item["vid"], item["query"]))
        print(f"🔁 Loaded {len(done_set)} completed samples, will skip them.")

    ious = []
    recall_03 = recall_05 = recall_07 = 0
    preprocess_config = parse_video_grid_dir(args.video_grid_dir)
    video_grid_dir = DATASETS[args.dataset]["splits"][args.splits]["video_image_dir"]
    video_grid_dir = f"{video_grid_dir}/{args.video_grid_dir}"
    video_grid_dir = args.video_grid_dir
    # print(video_grid_dir)

    # 5. 执行推理
    with open(result_path, "a", encoding="utf-8") as fout:
        pbar = tqdm(data, desc="Evaluating", dynamic_ncols=True)

        for k, gt_s_e, s in pbar:
            # _video_path = f"{video_path}/{k}.mp4"
            s = s.rstrip(".")
            if (k, s) in done_set:  # 跳过已完成的样本
                continue
            ans, ans_txt = get_finall_answer_single(
                video_grid_dir,
                k,
                s,
                model,
                processor,
                args=args,
                preprocess_config=preprocess_config,
            )
            iou = compute_iou(ans, gt_s_e)
            ious.append(iou)

            recall_03 += iou >= 0.3
            recall_05 += iou >= 0.5
            recall_07 += iou >= 0.7

            miou = sum(ious) / len(ious)
            r03 = recall_03 / len(ious)
            r05 = recall_05 / len(ious)
            r07 = recall_07 / len(ious)

            # fmt:off
            pbar.set_postfix({
                "mIoU": f"{miou:.4f}", "R@0.3": f"{r03:.4f}",
                "R@0.5": f"{r05:.4f}", "R@0.7": f"{r07:.4f}"
            })
            # fmt:on

            pred_s_e = ans
            new_item = {
                "vid": k,
                "query": s,
                "ans_txt": ans_txt,
                "gt": gt_s_e,
                "pred": pred_s_e,
                "iou": compute_iou(gt_s_e, pred_s_e),
            }
            fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")
            fout.flush()

    # 6.计算并保存最终指标
    metrics = {
        "R@0.3": recall_03 / len(ious),
        "R@0.5": recall_05 / len(ious),
        "R@0.7": recall_07 / len(ious),
        "mIoU": sum(ious) / len(ious),
    }

    save_metrics(metrics, metric_path)

    print(
        f"📊 Final Results:\n"
        f"  mIoU:  {metrics['mIoU']:.4f}\n"
        f"  R@0.3: {metrics['R@0.3']:.4f}\n"
        f"  R@0.5: {metrics['R@0.5']:.4f}\n"
        f"  R@0.7: {metrics['R@0.7']:.4f}"
    )
