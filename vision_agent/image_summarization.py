from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
import os
import logging
import torch
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from image_check import is_image_valid



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_image_summarizer():
    """Set up the image summarization model."""
    model_id = "microsoft/Phi-3.5-vision-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        _attn_implementation='flash_attention_2'
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=16)
    return model, processor


@torch.inference_mode()
def summarize_image(args):
    """Summarize a single image."""
    image_path, model, processor = args
    
    if not is_image_valid(image_path):
        return None
    
    start_time = time.time()
    
    image = Image.open(image_path)
    messages = [
        {"role": "user", "content": """<|image_1|>\You are material science expert. 
        Explain the important conclusion briefly so we can draw from the image with proper analysis in respect of Material Science.\
        Make proper analysis of the image to check all edges. \
        If the image does not represent any things related to material science you don't generate anything from that image, just give ''
        Don't give any apologetic answer or question."""},
    ]

    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")

    generation_args = {
        "max_new_tokens": 1500,
        "temperature": 0.7,
        "do_sample": True,
    }

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    return f"Summary for the image \n{response}\n"




def main():
    total_start_time = time.time()
    
    image_dir = "your input directory"
    output_dir = "your output directory"
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Setting up the image summarizer...")
    setup_start_time = time.time()
    model, processor = setup_image_summarizer()
    setup_end_time = time.time()
    setup_time = setup_end_time - setup_start_time
    logger.info(f"Image summarizer setup completed in {setup_time:.2f} seconds")
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    logger.info("Starting image summarization...")
    summarization_start_time = time.time()
    summaries = []
    skipped_images = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(summarize_image, (os.path.join(image_dir, img), model, processor)) for img in image_files]
        for future in tqdm(as_completed(futures), total=len(image_files), desc="Summarizing images"):
            try:
                summary = future.result()
                if summary:
                    summaries.append(summary)
                else:
                    skipped_images += 1
            except Exception as e:
                logger.error(f"Error summarizing image: {str(e)}")
                skipped_images += 1
    summarization_end_time = time.time()
    summarization_time = summarization_end_time - summarization_start_time
    logger.info(f"Image summarization completed in {summarization_time:.2f} seconds")
    
    output_file = os.path.join(output_dir, "image_summaries_speed1.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(summaries))
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    logger.info(f"Image summarization completed. Summaries saved to {output_file}")
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    logger.info(f"Breakdown:")
    logger.info(f"  - Setup time: {setup_time:.2f} seconds")
    logger.info(f"  - Summarization time: {summarization_time:.2f} seconds")
    logger.info(f"  - Images processed: {len(image_files) - skipped_images}")
    logger.info(f"  - Images skipped: {skipped_images}")
    if len(image_files) - skipped_images > 0:
        logger.info(f"  - Average time per processed image: {summarization_time / (len(image_files) - skipped_images):.2f} seconds")

if __name__ == "__main__":
    main()


