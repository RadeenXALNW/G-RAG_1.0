import os
import logging
from typing import List, Tuple
import pandas as pd
from PIL import Image
import re
import torch
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.core import Document, PropertyGraphIndex, Settings
from llama_index.extractors.relik.base import RelikPathExtractor
from llama_index.llms.groq import Groq
from llama_index.embeddings.jinaai import JinaEmbedding
from transformers import AutoModelForCausalLM, AutoProcessor
from utils.directory import setup_directories,get_pdf_files
# Import functions from other modules
from utils.document_parse import document2text_extraction
from table_agent.microsoft_table_transformer import extract_images_from_pdfs, process_images_and_cleanup
from vision_agent.image_extract_from_pdf import extract_images
from utils.library_init import get_library_name
import time
from llmware.configs import LLMWareConfig, MilvusConfig
from vision_agent.image_summarization import setup_image_summarizer,summarize_image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from utils.txt_reader import read_text_file
from llm_assistant.groq_assistant import process_chunk
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import fitz
from entity_link.relik_rag import split_text,create_documents,chunk_text


# API keys and credentials (replace with your actual keys)

NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD=""
NEO4J_URL=""
GROQ_API_KEY = ""
JINA_API_KEY = ""





def main():
    total_start_time = time.time()
    
    logger.info("Starting the pipeline process")
    
    pdf_dir, image_dir, output_dir = setup_directories()
    pdf_files = get_pdf_files(pdf_dir)
    print(pdf_files)
    if not pdf_files:
        logger.error("No PDF files found in the specified directory.")
        return

    LLMWareConfig().set_active_db("sqlite")
    MilvusConfig().set_config("lite", True)
    # Turn off debug mode
    LLMWareConfig().set_config("debug_mode", 0)
    # Step 1: PDF text extraction
    logger.info("Step 1: Extracting text from PDFs")
    library_name = get_library_name()
    logger.info(f"Using library name: {library_name}")
    text_df = document2text_extraction(library_name, pdf_files,pdf_dir)
    
    # Step 2: Table image extraction
    logger.info("Step 2: Extracting table images from PDFs")
    extract_images_from_pdfs(pdf_dir, image_dir)
    process_images_and_cleanup(image_dir, image_dir)
    
    # Step 3: Figure image extraction
    logger.info("Step 3: Extracting figure images from PDFs")
    for single_pdf in pdf_files:
        single_pdf=os.path.join(pdf_dir,single_pdf)
        pdf = fitz.open(single_pdf)
        for page in range(pdf.page_count):
            extract_images(pdf,page,image_dir)
    
    # Step 4: Image summarization
    logger.info("Step 4: Summarizing extracted images")
    logger.info("Setting up the image summarizer...")
    setup_start_time = time.time()
    model, processor = setup_image_summarizer()
    setup_end_time = time.time()
    setup_time = setup_end_time - setup_start_time
    logger.info(f"Image summarizer setup completed in {setup_time:.2f} seconds")
    
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
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
    
    combined_summary = "\n".join(summaries)
    
    # Merge text extraction and image summaries
    logger.info("Merging text extraction and image summaries")
    final_output = f"{text_df}\n\nImage Summaries:\n{combined_summary}"
    
    # Save the final output
    output_file = os.path.join(output_dir, "final_output_2.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_output)
    
    
    # Step 5: Process merged output with Groq API
    logger.info("Step 5: Processing merged output with Groq API")
    groq_start_time = time.time()
    print("############starting chunking################")
    chunks = chunk_text(final_output)
    
    # Open the output file for writing
    output_file = os.path.join(output_dir, "1-s2.0-S1359645419308444-am_bibliography.txt")
    print("##############joined the output file###################")
    with open(output_file, 'w', encoding='utf-8') as file:
        for i, chunk in tqdm(enumerate(chunks)):
            processed_chunk = process_chunk(chunk)
            # Remove any list formatting (assuming lists are marked with numbers or bullet points)
            processed_chunk = re.sub(r'^\s*[\d•\-\"\'“”‘’]+\s*', '', processed_chunk, flags=re.MULTILINE)
            # Write the processed chunk to the file
            file.write(processed_chunk + "\n\n")
            logger.info(f"Processed and wrote chunk {i+1}/{len(chunks)}")
    # Step 5: Process combined summary with Groq API
    logger.info("Step 5: Processing combined summary with Groq API")
    groq_start_time = time.time()
    print("############starting chunking################")
    summary_chunks = chunk_text(combined_summary)

    processed_summaries = []
    print("##############processing summary chunks###################")
    for i, chunk in tqdm(enumerate(summary_chunks), total=len(summary_chunks), desc="Processing summary chunks"):
        processed_chunk = process_chunk(chunk)
        # Remove any list formatting (assuming lists are marked with numbers or bullet points)
        processed_chunk = re.sub(r'^\s*[\d•\-\"\'""'']+\s*', '', processed_chunk, flags=re.MULTILINE)
        processed_summaries.append(processed_chunk)
        logger.info(f"Processed summary chunk {i+1}/{len(summary_chunks)}")

    processed_combined_summary = "\n\n".join(processed_summaries)

    # Merge text extraction and processed image summaries
    logger.info("Merging text extraction and processed image summaries")
    final_output = f"{text_df}\n\nProcessed Image Summaries:\n{processed_combined_summary}"

    # Save the final output
    output_file = os.path.join(output_dir, "1-s2.0-S1359645419308444-am_groq_figure.txt")
    print("##############writing to output file###################")
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(final_output)
    total_end_time = time.time()
    groq_end_time = time.time()
    total_time = total_end_time - total_start_time
    groq_time = groq_end_time - groq_start_time
    logger.info(f"Groq API processing completed in {groq_time:.2f} seconds")
    
    logger.info(f"Pipeline completed. Final output saved to {output_file}")
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    logger.info(f"Breakdown:")
    logger.info(f"  - Setup time: {setup_time:.2f} seconds")
    logger.info(f"  - Summarization time: {summarization_time:.2f} seconds")
    logger.info(f"  - Groq API processing time: {groq_time:.2f} seconds")
    logger.info(f"  - Images processed: {len(image_files) - skipped_images}")
    logger.info(f"  - Images skipped: {skipped_images}")
    if len(image_files) - skipped_images > 0:
        logger.info(f"  - Average time per processed image: {summarization_time / (len(image_files) - skipped_images):.2f} seconds")
    


    logger.info("Starting RAG-MATERIALS processing steps")
    final_output=read_text_file("/teamspace/studios/this_studio/NeurIPS_Materials/pipeline_output/final_output.txt")
    # Set up Neo4j graph store
    graph_store = Neo4jPGStore(
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        url=NEO4J_URL
    )

    # Set up Relik extractor
    relik = RelikPathExtractor(
        model="relik-ie/relik-relation-extraction-small", 
        model_config={"skip_metadata": True}
    )

    # Set up Groq LLM
    llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)

    # Set up Jina embedding model
    embed_model = JinaEmbedding(
        api_key=JINA_API_KEY,
        model="jina-embeddings-v2-base-en",
    )
    
    # Configure global settings
    Settings.llm = llm
    Settings.embed_model = embed_model

    # Create documents from input text
    docs = create_documents(final_output, metadata=None)
    print(docs)

    # Create PropertyGraphIndex
    index = PropertyGraphIndex.from_documents(
        docs,
        kg_extractors=[relik],
        llm=llm,
        embed_model=embed_model,
        property_graph_store=graph_store,
        show_progress=True,
    )

    # Set up query engine
    query_engine = index.as_query_engine(include_text=True)

    # Example query (you may want to modify this or make it configurable)
    query = "Summarize the key findings about High Antropy Alloy from the documents"
    
    # Perform query
    response = query_engine.query(query)
    
    # Save the query response
    query_output_file = os.path.join(output_dir, "query_output.txt")
    with open(query_output_file, 'w', encoding='utf-8') as f:
        f.write(str(response))
    
    logger.info(f"Query response saved to {query_output_file}")

    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    logger.info(f"All processing completed.")
    logger.info(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()


