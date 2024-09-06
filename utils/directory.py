import os
from typing import List, Tuple

def setup_directories() -> Tuple[str, str, str]:
    """Set up necessary directories for the pipeline."""
    base_dir = "/teamspace/studios/this_studio/NeurIPS_Materials"
    pdf_dir = os.path.join(base_dir, "data1", "documents_pdf")
    image_dir = os.path.join(base_dir, "table_data1")
    output_dir = os.path.join(base_dir, "pipeline_output")
    
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    return pdf_dir, image_dir, output_dir

def get_pdf_files(directory: str) -> List[str]:
    """Get a list of PDF files in the given directory."""
    return [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
