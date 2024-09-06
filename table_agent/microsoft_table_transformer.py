
import os
from pdf2image import convert_from_path
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import torch
import time
from PIL import Image
from pdf_2_img import extract_images_from_pdfs



def detect_and_crop_table(image_path, image_processor, model):
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.97, target_sizes=target_sizes)[0]
    
    if len(results["boxes"]) > 0:
        box = [round(coord) for coord in results["boxes"][0].tolist()]
        return image.crop(box)
    return None

def process_images_and_cleanup(image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
    
    start_time = time.time()
    total_images = 0
    tables_detected = 0

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(image_folder, filename)
            print(f"Processing {filename}...")
            
            cropped_table = detect_and_crop_table(input_path, image_processor, model)
            
            if cropped_table is not None:
                output_path = os.path.join(output_folder, f"table_{filename}")
                cropped_table.save(output_path)
                print(f"Table detected and saved as {output_path}")
                tables_detected += 1
            else:
                print(f"No table detected in {filename}")
            
            # Delete the original image
            os.remove(input_path)
            print(f"Deleted original image: {filename}")
            
            total_images += 1

    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nTotal images processed: {total_images}")
    print(f"Tables detected: {tables_detected}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time/total_images:.2f} seconds")

