# Install the required libraries first:
# pip install optimum[openvino] transformers

from optimum.intel import OVModelForVisualCausalLM
from transformers import AutoProcessor
from PIL import Image

# Load the Intel-optimized quantized model
model_id = "Qwen/Qwen2-VL-2B"
model = OVModelForVisualCausalLM.from_pretrained(model_id, device="GPU")
processor = AutoProcessor.from_pretrained(model_id)

# Load your image
image_name = 'page.png'
image = Image.open(image_name)

if image:
    print(f"{image_name} loaded successfully")  
else:
    raise FileNotFoundError(f"Image not found at: {image_name.resolve()}")

prompt = "<|im_start|>user\n<|image_pad|>\nConvert the text in this image to clean Markdown.<|im_end|>\n<|im_start|>assistant\n"


# Process and run inference on your Arc A750
inputs = processor(text=prompt, images=image, return_tensors="pt")
generated_ids = model.generate(**inputs, max_new_tokens=100)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_text)
