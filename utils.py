import os
from pathlib import Path

import pandas as pd
from PIL import Image


def format_data(sample, system_message=None):
    return {
        "images": [sample["image"]],

        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_message,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract all readable text from this image "
                            "exactly as it appears."
                        ),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": sample["label"],
                    }
                ],
            },
        ],
    }


def format_dataset(dataset_path: str):
    dataset_path = Path(dataset_path)

    image_dir = dataset_path / "images"
    label_dir = dataset_path / "text"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset folder does not exist: {dataset_path}"
        )

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image folder does not exist: {image_dir}"
        )

    if not label_dir.exists():
        raise FileNotFoundError(
            f"Text folder does not exist: {label_dir}"
        )

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
    }

    samples = []

    for image_path in sorted(image_dir.iterdir()):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in image_extensions:
            continue

        # Find corresponding text file.
        label_path = label_dir / f"txt_{image_path.stem.split("_")[-1]}"

        if not label_path.exists():
            print(
                f"WARNING: Missing label for {label_path.split("/")[-1]}"
            )
            continue

        # Load image.
        image = Image.open(image_path).convert("RGB")

        # Read OCR ground truth.
        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:
            label = f.read()

        samples.append(
            {
                "image": image,
                "label": label,
            }
        )

    if len(samples) == 0:
        raise RuntimeError(
            "No valid image/label pairs were found."
        )

    print(f"Loaded {len(samples)} image/label pairs.")

    return pd.DataFrame(samples)


def generate_text_from_sample(
    model,
    processor,
    sample,
    max_new_tokens=1024,
    device="xpu",
):
    from qwen_vl_utils import process_vision_info

    messages = sample["messages"][:2]

    text_input = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )

    model_inputs = {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in model_inputs.items()
    }

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(
            model_inputs["input_ids"],
            generated_ids,
        )
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0]
