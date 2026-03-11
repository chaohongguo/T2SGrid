import os
import math
import json
import argparse
from PIL import Image, ImageDraw, ImageFont
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from data.data_config import *


def frames_to_grid(
    frames,
    timestamps,
    grid_cols=3,
    grid_rows=3,
    thumb_size=None,
    thumb_ratio=None,
    add_visual_num=False,
    line_width=0,  # 分割线宽度
    line_color=(255, 128, 128),
):
    """
    line_width: 分割线宽度（像素）
    line_color: 分割线颜色
    """
    assert len(frames) <= grid_cols * grid_rows
    orig_w, orig_h = frames[0].size

    if thumb_size is not (None, None):
        w, h = thumb_size
    elif thumb_ratio is not None and thumb_ratio != 1:
        w, h = int(orig_w * thumb_ratio), int(orig_h * thumb_ratio)
    else:
        w, h = orig_w, orig_h

    grid_w = grid_cols * w
    grid_h = grid_rows * h
    grid_img = Image.new("RGB", (grid_w, grid_h))

    for idx, (img, sec) in enumerate(zip(frames, timestamps)):
        if (w, h) != (orig_w, orig_h):
            img = img.resize((w, h), Image.LANCZOS)

        col = idx % grid_cols
        row = idx // grid_cols

        x = col * w
        y = row * h
        grid_img.paste(img, (x, y))
    if line_width > 0:
        # === 绘制分割线 ===
        draw = ImageDraw.Draw(grid_img)
        for c in range(1, grid_cols):
            x = c * w - line_width // 2
            draw.line([(x, 0), (x, grid_h)], fill=line_color, width=line_width)
        for r in range(1, grid_rows):
            y = r * h - line_width // 2
            draw.line([(0, y), (grid_w, y)], fill=line_color, width=line_width)

    return grid_img


def get_grid_size(duration, fps=1, target_ratio=None):
    """
    计算 grid 大小
    - duration: 视频时长（秒）
    - fps: 帧率（默认 1，即每秒 1 帧）
    - target_ratio: 目标宽高比 (w/h)，如 16/9。如果为 None，则最小浪费优先
    """
    num_frames = int(duration * fps) + 2  # 包含首尾帧

    best_cols, best_rows = None, None
    best_score = float("inf")

    for cols in range(5, num_frames + 1):
        rows = math.ceil(num_frames / cols)
        total_cells = cols * rows
        waste = total_cells - num_frames

        if target_ratio is None:
            # 优先浪费少
            score = waste
        else:
            # 同时考虑浪费和比例
            actual_ratio = cols / rows
            ratio_diff = abs(actual_ratio - target_ratio)
            score = waste + ratio_diff * num_frames * 0.2
            # 0.2 是权重，可以调节浪费 vs 比例的重要性

        if score < best_score:
            best_score = score
            best_cols, best_rows = cols, rows

    return (best_cols, best_rows)


def video_to_grids(
    video_path,
    out_dir,
    grid_size=(3, 3),
    fps=1.0,
    thumb_size=(320, 180),
    thumb_ratio=1,
    stride=3,
    add_visual_num=False,
    line_width=2,
):
    os.makedirs(out_dir, exist_ok=True)

    clip = VideoFileClip(video_path)
    duration = clip.duration
    if grid_size is None:
        grid_size = get_grid_size(duration)

    # print("Grid size:", grid_size)

    cols, rows = grid_size
    frames_per_grid = cols * rows

    if stride is None:
        stride = frames_per_grid  # 默认和窗口大小一样，不重叠

    collected = []
    timestamps = []
    grid_count = 0

    step = 1.0 / fps
    frames = []
    t = 0.0
    while t < duration:
        frame = clip.get_frame(t)
        pil_img = Image.fromarray(frame)
        frames.append((pil_img, t))
        t += step

    # 用滑动窗口取 frame
    i = 0
    while i + frames_per_grid <= len(frames):
        collected = [f for f, _ in frames[i : i + frames_per_grid]]
        timestamps = [ts for _, ts in frames[i : i + frames_per_grid]]
        grid_img = frames_to_grid(
            collected,
            timestamps,
            cols,
            rows,
            add_visual_num=add_visual_num,
            thumb_size=thumb_size,
            thumb_ratio=thumb_ratio,
            line_width=line_width,
        )
        out_path = os.path.join(out_dir, f"grid_{grid_count:05d}.png")
        grid_img.save(out_path, quality=95)
        # print(f"保存: {out_path}")
        grid_count += 1
        i += stride  # 滑动窗口步长

    # 保存最后不足一页的 grid
    if i < len(frames):
        collected = [f for f, _ in frames[i:]]
        timestamps = [ts for _, ts in frames[i:]]
        missing = frames_per_grid - len(collected)
        w, h = collected[0].size
        for _ in range(missing):
            collected.append(Image.new("RGB", (w, h), color=(0, 0, 0)))
            timestamps.append(-1)
        grid_img = frames_to_grid(
            collected,
            timestamps,
            cols,
            rows,
            thumb_size=thumb_size,
            thumb_ratio=thumb_ratio,
            line_width=line_width,
        )
        out_path = os.path.join(
            out_dir, f"grid_{grid_count:05d}_partial_{int(duration)}.png"
        )
        grid_img.save(out_path, quality=95)
        # print(f"保存残余: {out_path}")

    clip.close()
    # print("完成")


def get_vid_list(data_path, dataset):
    if dataset == "charades" or dataset == "anet" or dataset == "didemo":
        with open(data_path, "r") as f:
            data = json.load(f)
            return data.keys()
    elif dataset == "qvhighlights":
        with open(data_path, "r") as f:
            data = [json.loads(line)["vid"] for line in f]
            return data
    elif dataset == "videomme" or dataset == "mvbench":
        with open(data_path, "r") as f:
            data = [json.loads(line)["videoID"] for line in f]
            return data
    elif dataset == "internvid":
        # with open(data_path, "r") as f:
        #     data = [json.loads(line)["vid"] for line in f]
        #     return data    
        with open(data_path, "r") as f:
            data = json.load(f)
            return data.keys()  
    return None


def process_video(vid, video_dir, out_dir, args):
    """单个视频的处理逻辑"""
    for ext in [".mp4", ".mkv", ".avi", ".webm"]:
        video_path = os.path.join(video_dir, f"{vid}{ext}")
        if os.path.exists(video_path):
            break
    else:
        return f"[ERROR] No video found for {vid} in {video_dir}"

    _out_dir = os.path.join(out_dir, vid)
    video_to_grids(
        video_path,
        _out_dir,
        grid_size=tuple(args.grid_size),
        fps=args.fps,
        thumb_size=args.thumb_size,
        thumb_ratio=args.thumb_ratio,
        stride=args.stride,
        add_visual_num=args.add_visual_num,
        line_width=args.line_width,
    )
    return f"[OK] {vid}"


def process_all_videos(video_dir, vid_list, out_dir, args, num_workers=8):
    """并行处理所有视频"""
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_video, vid, video_dir, out_dir, args): vid
            for vid in vid_list
        }

        for f in tqdm(
            as_completed(futures), total=len(futures), desc="Processing videos"
        ):
            results.append(f.result())

    print("\n".join(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid_size",
        type=int,
        nargs=2,
        metavar=("cols", "rows"),
        default=(3, 3),
        help="",
    )
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument(
        "--thumb_size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=(None, None),
        help="Thumbnail size as (width height)",
    )
    parser.add_argument("--line_width", type=int, default=0)
    parser.add_argument("--add_visual_num", action="store_true")
    parser.add_argument("--dataset", type=str, default="charades")
    parser.add_argument("--split", type=str, default="small")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--thumb_ratio", type=float, default=1)

    args = parser.parse_args()

    data_path = DATASETS[args.dataset]["splits"][args.split]["annotation_file"]
    vid_list = get_vid_list(data_path, dataset=args.dataset)
    video_dir = DATASETS[args.dataset]["video_path"]
    out_dir = DATASETS[args.dataset]["splits"][args.split]["video_image_dir"]

    # 假设 args.thumb_size 是 (W, H) 或 (None, None)
    if args.thumb_size[0] is not None and args.thumb_size[1] is not None:
        thumb_suffix = f"_ts{args.thumb_size[0]}x{args.thumb_size[1]}"
    else:
        thumb_suffix = ""

    out_dir = (
        f"{out_dir}/"
        f"g{args.grid_size[0]}{args.grid_size[1]}"
        f"_f{args.fps}"
        f"_s{args.stride}"
        f"_r{args.thumb_ratio}"
        f"_l{args.line_width}"
        f"{thumb_suffix}"
    )
    process_all_videos(video_dir, vid_list, out_dir, args, num_workers=80)
