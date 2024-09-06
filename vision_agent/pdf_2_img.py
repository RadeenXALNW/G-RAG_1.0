import os
from pdf2image import convert_from_path
import time 

def extract_images_from_pdfs(pdf_folder, image_folder):
    os.makedirs(image_folder, exist_ok=True)
    total_start_time = time.time()
    total_pdfs = 0
    total_pages = 0

    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, filename)
            start_time = time.time()
            
            images = convert_from_path(pdf_path, output_folder=image_folder, fmt="png", dpi=150, thread_count=3, size=(500,500))
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            total_pdfs += 1
            total_pages += len(images)
            
            print(f"Processed {filename}: {len(images)} pages in {processing_time:.2f} seconds")

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print(f"\nTotal PDFs processed: {total_pdfs}")
    print(f"Total pages extracted: {total_pages}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per PDF: {total_time/total_pdfs:.2f} seconds")
    print(f"Average time per page: {total_time/total_pages:.2f} seconds")
    print("All PDFs have been converted to images.")