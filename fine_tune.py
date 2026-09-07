import os
import sys
import gc
from datetime import datetime

import torch

from sklearn.model_selection import train_test_split
from datasets import Dataset


# ============================================================
# Logging
# ============================================================

log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(
    log_dir,
    f"fine_tune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)


class Tee:
    """Write console output to both the terminal and a log file.

    Implements the common text-stream methods expected by
    libraries such as tqdm and Transformers.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        # Return True if the actual terminal stream is a TTY.
        return any(
            getattr(stream, "isatty", lambda: False)()
            for stream in self.streams
        )

    def fileno(self):
        # Libraries may need the terminal's file descriptor.
        for stream in self.streams:
            if hasattr(stream, "fileno"):
                try:
                    return stream.fileno()
                except (OSError, ValueError):
                    pass
        raise OSError("Tee has no valid file descriptor")

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self.streams[0], "errors", "strict")

    def writable(self):
        return True

    def readable(self):
        return False

    def seekable(self):
        return False


log_handle = open(log_file, "a", encoding="utf-8")

# Save stdout and stderr while still displaying them in the terminal.
sys.stdout = Tee(sys.__stdout__, log_handle)
sys.stderr = Tee(sys.__stderr__, log_handle)

print("=" * 60)
print(f"Log file: {log_file}")
print("=" * 60)

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)

from peft import (
    LoraConfig,
)

from trl import (
    SFTConfig,
    SFTTrainer,
)

from utils import (
    format_data,
    format_dataset,
)


# ============================================================
# 1. Device check
# ============================================================
    

if not torch.xpu.is_available():
    raise RuntimeError(
        "Intel XPU is not available."
    )

device = torch.device("xpu")


def print_xpu_memory(label):
    """Print XPU memory statistics without forcing synchronization."""
    try:
        free_b, total_b = torch.xpu.mem_get_info()
        allocated_b = torch.xpu.memory_allocated()
        reserved_b = torch.xpu.memory_reserved()
        print(
            f"[XPU MEMORY] {label}: "
            f"free={free_b / 1024**3:.2f} GiB, "
            f"allocated={allocated_b / 1024**3:.2f} GiB, "
            f"reserved={reserved_b / 1024**3:.2f} GiB, "
            f"total={total_b / 1024**3:.2f} GiB"
        )
    except Exception as exc:
        print(f"[XPU MEMORY] Could not read memory stats: {exc}")


def cleanup_xpu_memory():
    gc.collect()
    try:
        torch.xpu.empty_cache()
    except Exception:
        pass

print("-" * 60)
print("PyTorch:", torch.__version__)
print("XPU available:", torch.xpu.is_available())
print("XPU device:", torch.xpu.get_device_name(0))
print_xpu_memory("after device initialization")
print("-" * 60)


# ============================================================
# 2. System prompt
# ============================================================

system_message = """
You are an expert document analysis and high-precision OCR
(Optical Character Recognition) assistant.

Your sole task is to visually scan the provided image and
extract all readable text with high accuracy.

Rules:

1. Verbatim Extraction
Transcribe visible text exactly as it appears.
Do not correct spelling mistakes, grammar errors, or typos.

2. Layout Preservation
Preserve line breaks and paragraphs whenever possible.

3. Tables and Structure
Represent tables as Markdown tables when appropriate.

4. Unreadable Text
If a character or word is completely unreadable, output
[illegible]. Do not guess.

5. Zero Commentary
Output only the extracted text.
Do not provide explanations or commentary.
"""


# ============================================================
# 3. Dataset
# ============================================================

dataset_dir = os.path.join(
    os.getcwd(),
    "dataset_folder",
    "data",
)

df = format_dataset(dataset_dir)

print("Total samples:", len(df))


# ============================================================
# 4. Train / validation / test split
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.25,
    random_state=42,
)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# ============================================================
# 5. Format for TRL VLM training
# ============================================================

# TRL SFTTrainer requires a Hugging Face Dataset (or IterableDataset),
# not a plain Python list. Build the formatted rows first, then convert
# each split to datasets.Dataset.
train_rows = [
    format_data(
        sample,
        system_message,
    )
    for sample in train_df.to_dict("records")
]

eval_rows = [
    format_data(
        sample,
        system_message,
    )
    for sample in val_df.to_dict("records")
]

test_rows = [
    format_data(
        sample,
        system_message,
    )
    for sample in test_df.to_dict("records")
]

train_dataset = Dataset.from_list(train_rows)
eval_dataset = Dataset.from_list(eval_rows)
test_dataset = Dataset.from_list(test_rows)

print("Dataset types:")
print("  train_dataset:", type(train_dataset))
print("  eval_dataset:", type(eval_dataset))
print("  test_dataset:", type(test_dataset))


# ============================================================
# 6. Verify one sample BEFORE loading the model
# ============================================================

# print("-" * 60)

# print("Image type:")
# print(type(train_dataset[0]["images"][0]))

# print("\nLabel:")
# print(
#     train_dataset[0]["messages"][-1]["content"][0]["text"]
# )

# print("-" * 60)


# ============================================================
# 7. A750 8-GB memory configuration
# ============================================================


# ============================================================
# 8. Load Qwen2-VL
# ============================================================

model_id = "Qwen/Qwen2-VL-2B-Instruct"

# Intel XPU / Arc A750:
# Do NOT use BitsAndBytes 4-bit quantization here.
# The previous 4-bit path invoked Intel Triton and failed while
# loading spirv_utils.cp313-win_amd64.pyd on Python 3.13.
# We load the model in BF16 directly on the XPU instead.

# Keep the model in BF16, but aggressively reduce the two biggest
# sources of training memory on an 8-GB card: vision tokens and
# language-sequence length. Qwen2-VL explicitly supports controlling
# image resolution through min_pixels/max_pixels.

cleanup_xpu_memory()

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

# Move the complete model once, after loading, instead of using a
# device_map that can create additional placement/copy peaks.
model = model.to(device)

# OCR fine-tuning does not need to update the large vision encoder.
# Freezing it removes its activations from the autograd graph and is a
# major memory saving on an 8-GB GPU.
if hasattr(model, "visual"):
    for parameter in model.visual.parameters():
        parameter.requires_grad = False
    model.visual.eval()

processor = Qwen2VLProcessor.from_pretrained(
    model_id,
    min_pixels=256 * 28 * 28,
    max_pixels=512 * 28 * 28,
)

print(
    "Model loading mode: BF16 on Intel XPU; "
    "BitsAndBytes 4-bit disabled; vision encoder frozen"
)
print(
    "Image pixel limit:",
    f"{256 * 28 * 28}..{512 * 28 * 28}"
)
print_xpu_memory("after model load")


# ============================================================
# 9. Prepare model for LoRA training
# ============================================================

model.config.use_cache = False


# ============================================================
# 10. LoRA
# ============================================================

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",

    target_modules=[
        "q_proj",
        "v_proj",
    ],

    task_type="CAUSAL_LM",
)


# ============================================================
# 11. Training configuration
# ============================================================

training_args = SFTConfig(
    output_dir="qwen2-2b-ocr-xpu",

    num_train_epochs=3,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": False
    },

    # IMPORTANT FOR QWEN2-VL
    max_length=None,

    optim="adamw_torch",
    learning_rate=2e-4,

    logging_steps=10,
    eval_steps=100,
    eval_strategy="steps",

    save_strategy="steps",
    save_steps=100,

    bf16=True,
    max_grad_norm=0.3,

    push_to_hub=False,
    report_to="none",
)


# ============================================================
# 12. Trainer
# ============================================================

trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    peft_config=peft_config,

    processing_class=processor,
)


# ============================================================
# 13. Train
# ============================================================

print("-" * 60)
print("Starting training...")
print_xpu_memory("immediately before trainer.train()")
print("-" * 60)

try:
    trainer.train()
except torch.OutOfMemoryError:
    print("=" * 60)
    print("XPU OUT OF MEMORY")
    print_xpu_memory("at OOM")
    print("The A750 profile is already using batch=1, frozen vision, low image pixels,")
    print("and a 512-token cap. Reduce max_pixels to 256*28*28 or max_length to 384")
    print("if this still OOMs on your particular images/Transformers version.")
    print("=" * 60)
    raise


print_xpu_memory("after training")
cleanup_xpu_memory()

# ============================================================
# 14. Save adapter
# ============================================================

trainer.save_model(
    training_args.output_dir
)

print("-" * 60)
print("Training finished.")
print(
    f"Model saved to: {training_args.output_dir}"
)
print(
    f"Training log saved to: {log_file}"
)
print("-" * 60)

# Flush buffered output before the script exits.
log_handle.flush()
log_handle.close()
