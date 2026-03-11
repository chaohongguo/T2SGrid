import glob
import logging
import os
import re
import sys

import datasets
from dataclasses import dataclass, field
from typing import Optional
from peft import get_peft_model
import torch
import transformers
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
import trl
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from tqdm import tqdm
import json
import random

from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SFTScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'iou', 'format'.
    """

    train_data_path: str = field(
        default="./Charades/charades_annotation/train.json",
        metadata={"help": "Path to the training data JSON file."},
    )
    eval_data_path: str = field(
        default="./Charades/charades_annotation/val.json",
        metadata={"help": "Path to the evaluation data JSON file."},
    )

    video_image_dir: str = field(
        default="./Charades/Charades_v1",  # Replace with your actual video folder path
        metadata={"help": "Path to the folder containing video files."},
    )
    preprocessed_data_path: Optional[str] = (
        field(  # Add preprocessed_data_path argument
            default="",
            metadata={
                "help": "Path to the preprocessed dataset directory. If provided, load preprocessed data instead of raw videos."
            },
        )
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(
        default=False, metadata={"help": "Whether to overwrite the Hub revision."}
    )
    push_to_hub_revision: bool = field(
        default=False, metadata={"help": "Whether to push to a Hub revision/branch."}
    )

def parse_string(s: str):
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


def load_json_dataset(
    train_data_path, eval_data_path, video_image_dir, preprocessed_data_path=None
):  # Modified to accept preprocessed_data_path
    def create_dataset_from_json(file_path, split_name):
        with open(file_path, "r") as f:
            data = json.load(f)
        examples = []

        for video_id, video_data in tqdm(data.items()):
            for sentence_id, (timestamps, sentence) in enumerate(
                zip(video_data["timestamps"], video_data["sentences"])
            ):
                sentence = sentence.strip().lower()
                if sentence.endswith("."):
                    sentence = sentence[:-1]
                video_filename_base = video_id
                video_path = None
                # for ext in ['mp4', 'mkv', 'webm']:
                # ext = "mp4"
                # candidate_path = os.path.join(
                #     video_folder, f"{video_filename_base}.{ext}"
                # )
                video_image_path = f"{video_image_dir}/{video_id}"
                # if os.path.isfile(candidate_path):
                # video_path = candidate_path
                # break
                # if video_path is None:
                #     print(f"Warning: Video file not found for ID: {video_id}")
                #     continue
                preprocess_config = parse_string(os.path.basename(video_image_dir))
                example = {
                    "problem": sentence,
                    # "solution": (timestamps[0] / video_data['duration'], timestamps[1] / video_data['duration']),
                    # "solution": (timestamps[0], timestamps[1]),
                    "solution": (timestamps[0], timestamps[1]),
                    "video_image_path": video_image_path,
                    "preprocess_config": preprocess_config,
                    "durations": video_data["duration"],
                    "preprocessed_path": "",  # Initialize preprocessed_path as None
                }
                example["preprocessed_path"] = video_image_path
                # if preprocessed_data_path != "": # If preprocessed data path is provided, construct the path
                #     example["preprocessed_path"] = os.path.join(preprocessed_data_path, split_name, f"{video_id}_{sentence_id}")
                examples.append(example)

        random.shuffle(examples)
        print(len(examples))
        print(examples[:5])
        dataset = Dataset.from_list(examples)

        def __getitem__(
            self, idx
        ):  # Define getitem within the scope where dataset is available
            example = dataset[idx]

            # return example
            data_to_return = {
                k: v for k, v in example.items()
            }  # Create a copy to avoid modifying original dataset

            if example["preprocessed_path"] != "":  # Check if preprocessed path exists
                # try:
                # import pdb; pdb.set_trace()
                # import pdb;pdb.set_trace()
                messages = convert_example(dataset[idx[0]])["messages"]
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    [messages], return_video_kwargs=True
                )
                fps_inputs = video_kwargs["fps"]
                data_to_return["video_inputs"] = [video_inputs]
                data_to_return["image_inputs"] = [image_inputs]
                data_to_return["video_kwargs"] = [video_kwargs]
                data_to_return["use_preprocessed"] = [
                    True
                ]  # Flag to indicate preprocessed data is used
            # except Exception as e:
            #     print(f"Warning: Error loading preprocessed data from {example['preprocessed_path'][0]}, falling back to video_path. Error: {e}")
            #     data_to_return["use_preprocessed"] = [False] # Fallback to video_path if loading fails
            else:
                data_to_return["use_preprocessed"] = [
                    False
                ]  #  No preprocessed data to use or path invalid

            return data_to_return

        dataset.__getitem__ = __getitem__.__get__(
            dataset, Dataset
        )  # Bind getitem to the dataset

        return dataset

    train_dataset = create_dataset_from_json(train_data_path, "train")
    # eval_dataset = create_dataset_from_json(eval_data_path, "eval")
    return DatasetDict({"train": train_dataset, "eval": None})


processor = None


QUESTION_TEMPLATE = """During which frames can we see [EVENT]? Answer strictly in the format of 'from x to y'."""

def convert_example(example):
    
    image_paths = sorted(glob.glob(os.path.join(example["video_image_path"], "*.png")))
    grid_size = example["preprocess_config"]["grid_size"]
    stride = example["preprocess_config"]["stride"]
    
    contents = []
    cols, rows = grid_size
    frames_per_grid = cols * rows

    for idx, url in enumerate(image_paths):
        start = idx * stride
        end = start + frames_per_grid - 1

        # 如果文件名里带有 partial_xxx，就用这个数字覆盖 end
        m = re.search(r"partial_(\d+)", url)
        if m:
            end = int(m.group(1))

        contents.append({"type": "text", "text": f"Frame {start} to Frame {end}"})
        contents.append({"type": "image", "image": url})
        

    contents.append({"type": "text", "text": "This video has {end} frames in total."+QUESTION_TEMPLATE.replace("[EVENT]", example["problem"])})
    messages = [{"role": "user", "content": contents}]
    

    st, ed = example["solution"]
    answer_text = f"from {round(st)} to {round(ed)}"
    messages.append(
        {
            "role": "assistant",
            "content": answer_text,
        }
    )
    # print(messages)
    # exit(0)
    example["messages"] = messages

    return example


def collate_fn(examples):
    texts = [
        processor.apply_chat_template(
            convert_example(example)["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        for example in examples
    ]

    # video_inputs = [x["video_inputs"] for x in examples]
    # fps_inputs = [x["video_kwargs"]["fps"] for x in examples]
    # video_inputs = video_inputs[0]
    # fps_inputs = fps_inputs[0]
    
    image_inputs = [x["image_inputs"] for x in examples]
    
    video_inputs = None
    fps_inputs = None
    # print(texts)
    # print(image_inputs)
    # exit(0)
    batch = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        fps=fps_inputs,
        return_tensors="pt",
        padding=True,
    )
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    # video_token_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
    # labels[labels == video_token_id] = -100
    iamge_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == iamge_token_id] = -100
    batch["labels"] = labels

    return batch


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")
    print(model_args)
    print(script_args)
    print(training_args)
    # print("", model_args)
    
    # exit(0)
    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################

    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # import pdb; pdb.set_trace()
    dataset = load_json_dataset(
        script_args.train_data_path,
        script_args.eval_data_path,
        script_args.video_image_dir,
        script_args.preprocessed_data_path,  # Pass preprocessed_data_path
    )

    ################
    # Load tokenizer
    ################
    global processor
    if "vl" in model_args.model_name_or_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
        )
        logger.info("Using AutoProcessor for vision-language model.")
    else:
        processor = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            use_fast=True,
        )
        logger.info("Using AutoTokenizer for text-only model.")
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif (
        hasattr(processor.tokenizer, "pad_token")
        and processor.tokenizer.pad_token is None
    ):
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_sliding_window=True,
    )
    # training_args.model_init_kwargs = model_kwargs
    # from transformers import Qwen2_5_VLForConditionalGeneration

    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     model_args.model_name_or_path,
    #     # torch_dtype=torch.bfloat16,
    #     **model_kwargs,
    # )
    from transformers import Qwen2VLForConditionalGeneration

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        # torch_dtype=torch.bfloat16,
        **model_kwargs,
    )
     
    # peft_config = get_peft_config(model_args)
    # # 如果想冻结原模型，只训练 LoRA，可以设置 freeze_model=True
    # # peft_config.freeze_model = True  # 全部冻结除 LoRA 外的参数
    # model = get_peft_model(model, peft_config)
    
    # # 冻结 ViT
    # for name, param in model.named_parameters():
    #     if 'visual' in name:
    #         param.requires_grad = False
    #     elif 'lora' in name:
    #         param.requires_grad = True    
 
    ############################
    # Initialize the SFT Trainer
    ############################
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    training_args.remove_unused_columns = False
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=None,
        # eval_dataset=dataset["test"] if training_args.eval_strategy != "no" else None,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args),
    )
    
    # <插入一些检查那些参数被微调了>
    # === 打印可训练参数 ===
    trainable_params = []
    frozen_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)

    print(f"\n✅ Trainable layers: {len(trainable_params)}")
    print(f"❄️ Frozen layers: {len(frozen_params)}")
    print("---- Trainable Parameter Names ----")
    for n in trainable_params:
        print(n)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total/1e6:.2f} M")
    print(f"Trainable parameters: {trainable/1e6:.2f} M")
    print(f"Trainable ratio: {100 * trainable / total:.2f}%\n")
    # === 检查代码结束 === 


    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["R1-V"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    #############
    # push to hub
    #############

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)
        processor.push_to_hub(training_args.hub_model_id)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
