# from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# # loads BAAI/bge-small-en
# # embed_model = HuggingFaceEmbedding()

# # loads BAAI/bge-small-en-v1.5
# embed_model = HuggingFaceEmbedding(model_name="dunzhang/stella_en_1.5B_v5")

# embeddings = embed_model.get_text_embedding("Hello World!")
# print(len(embeddings))
# print(embeddings[:5])

# [-0.03926713764667511, 0.06128763407468796, 0.005980746820569038, 0.03663117438554764, 0.02998274192214012]
# [-0.008377508260309696, 0.03951486945152283, 0.05756789818406105, -0.017540007829666138, -0.018802067264914513]





# from relik import Relik
# from relik.inference.data.objects import RelikOutput
# from typing import List, Dict, Any

# def extract_definitions(relik_output: RelikOutput) -> List[Dict[str, Any]]:
#     definitions = []
    
#     if relik_output.candidates and relik_output.candidates.span:
#         for window in relik_output.candidates.span:
#             for documents in window:
#                 for doc in documents:
#                     # Access the properties of the Document object
#                     text = getattr(doc, 'text', '')
#                     doc_id = getattr(doc, 'id', '')
#                     metadata = getattr(doc, 'metadata', {})
#                     definition = metadata.get('definition', '')
                    
#                     if definition:
#                         definitions.append({
#                             'text': text,
#                             'id': doc_id,
#                             'definition': definition
#                         })
    
#     return definitions

# # Use the model and get the output
# relik = Relik.from_pretrained("sapienzanlp/relik-entity-linking-large", device="cuda")
# relik_out: RelikOutput = relik("Michael Jordan was one of the best players in the NBA.")

# # Extract definitions
# definitions = extract_definitions(relik_out)

# print("\nExtracted Definitions:")
# for index, def_dict in enumerate(definitions, 1):
#     print(f"{index}. Text: {def_dict['text']}")
#     print(f"   ID: {def_dict['id']}")
#     print(f"   Definition: {def_dict['definition']}")
#     print()

# print(f"Number of definitions extracted: {len(definitions)}")



# from relik import Relik
# from relik.inference.data.objects import RelikOutput
# from typing import List

# def extract_definitions(relik_output: RelikOutput) -> List[str]:
#     definitions = []
    
#     if relik_output.candidates and relik_output.candidates.span:
#         for window in relik_output.candidates.span:
#             for documents in window:
#                 for doc in documents:
#                     metadata = getattr(doc, 'metadata', {})
#                     definition = metadata.get('definition', '')
                    
#                     if definition:
#                         definitions.append(definition)
    
#     return definitions

# def save_definitions_to_file(definitions: List[str], filename: str):
#     with open(filename, 'w', encoding='utf-8') as f:
#         for definition in definitions:
#             f.write(f"{definition}\n")

# # Use the model and get the output
# relik = Relik.from_pretrained("sapienzanlp/relik-entity-linking-large", device="cuda",top_k=20)
# relik_out: RelikOutput = relik("Michael")

# # Extract definitions
# definitions = extract_definitions(relik_out)

# # Save definitions to file
# output_file = "michael_definitions.txt"
# save_definitions_to_file(definitions, output_file)

# print(f"\nDefinitions saved to {output_file}")
# print(f"Number of definitions extracted: {len(definitions)}")

# # Print the first few definitions as a sample
# print("\nSample of extracted definitions:")
# for index, definition in enumerate(definitions[:5], 1):
#     print(f"{index}. {definition[:100]}...")  # Print first 100 characters of each definition



# from relik import Relik
# from relik.inference.data.objects import RelikOutput
# from typing import List

# def extract_definitions(relik_output: RelikOutput) -> List[str]:
#     definitions = []
    
#     if relik_output.candidates and relik_output.candidates.span:
#         for window in relik_output.candidates.span:
#             for documents in window:
#                 for doc in documents:
#                     metadata = getattr(doc, 'metadata', {})
#                     definition = metadata.get('definition', '')
                    
#                     if definition:
#                         definitions.append(definition)
    
#     return definitions

# def save_definitions_to_file(definitions: List[str], filename: str):
#     with open(filename, 'w', encoding='utf-8') as f:
#         for definition in definitions:
#             f.write(f"{definition}\n")

# # Initialize the model
# relik = Relik.from_pretrained("sapienzanlp/relik-entity-linking-large", device="cuda")

# # # Modify top_k after initialization
# relik.top_k = 20
# relik.device="cuda"
# relik.window_size=64
# print(f"Set top_k to: {relik.top_k}")
# print(f"relik_device : {relik.relik_device}")
# print(f"relik_window_size: {relik.window_size}")


# # Use the model to get the output
# relik_out: RelikOutput = relik("Michael")

# # Extract definitions
# definitions = extract_definitions(relik_out)

# # Save definitions to file
# output_file = "michael_definitions_top_20.txt"
# save_definitions_to_file(definitions, output_file)

# print(f"\nDefinitions saved to {output_file}")
# print(f"Number of definitions extracted: {len(definitions)}")

# # Print all definitions
# print("\nExtracted definitions:")
# for index, definition in enumerate(definitions, 1):
#     print(f"{index}. {definition}")


import os
from typing import List
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.core import Document, PropertyGraphIndex, Settings
from llama_index.core.node_parser import TokenTextSplitter,SentenceSplitter
from llama_index.extractors.relik.base import RelikPathExtractor
from llama_index.llms.groq import Groq
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm
import time
import aiohttp
from relik import Relik
from relik.inference.data.objects import RelikOutput
from typing import List, Tuple



def read_text_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except IOError as e:
        print(f"Error reading file: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 1024, chunk_overlap: int = 0) -> List[str]:
    sentence_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return sentence_splitter.split_text(text)

def extract_definitions(relik_output: RelikOutput) -> List[Tuple[str, str]]:
    definitions = []
    if relik_output.candidates and relik_output.candidates.span:
        for window in relik_output.candidates.span:
            for documents in window:
                for doc in documents:
                    text = getattr(doc, 'text', '')
                    metadata = getattr(doc, 'metadata', {})
                    definition = metadata.get('definition', '')
                    if definition:
                        definitions.append((text, definition))
    return definitions

def process_chunks_with_relik(chunks: List[str], relik: Relik) -> List[List[Tuple[str, str]]]:
    all_definitions = []
    for chunk in tqdm(chunks, desc="Processing chunks"):
        relik_out: RelikOutput = relik(chunk)
        chunk_definitions = extract_definitions(relik_out)
        all_definitions.append(chunk_definitions)
    return all_definitions

def save_definitions(definitions: List[List[Tuple[str, str]]], output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk_defs in tqdm(definitions, desc="Saving definitions"):
            for text, definition in chunk_defs:
                f.write(f"{text}|{definition}\n")


def merge_files(file1, file2, output_file):
    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output_file, 'w') as out:
        out.write(f1.read() + f2.read())

# Main execution
input_file = "/teamspace/studios/this_studio/test1.txt"
output_definitions = "/teamspace/studios/this_studio/definitions3.txt"
merged_output = "/teamspace/studios/this_studio/merged_output3.txt"

# Initialize Relik
print("Initializing Relik...")
start_time = time.time()
relik = Relik.from_pretrained("sapienzanlp/relik-entity-linking-large", device="cuda")
relik.top_k = 1
relik.device = "cuda"
relik.window_size = 64
print(f"Relik initialization time: {time.time() - start_time:.2f} seconds")

# Read and process the file
print("Reading input file...")
start_time = time.time()
text = read_text_file(input_file)
print(f"File reading time: {time.time() - start_time:.2f} seconds")

print("Chunking text...")
start_time = time.time()
chunks = chunk_text(text, chunk_size=128, chunk_overlap=2)
print(f"Text chunking time: {time.time() - start_time:.2f} seconds")

print("Processing chunks with Relik...")
start_time = time.time()
all_definitions = process_chunks_with_relik(chunks, relik)
print(f"Relik processing time: {time.time() - start_time:.2f} seconds")

# Save definitions
print("Saving definitions...")
start_time = time.time()
save_definitions(all_definitions, output_definitions)
print(f"Definitions saving time: {time.time() - start_time:.2f} seconds")

# Merge files
print("Merging files...")
start_time = time.time()
merge_files(input_file, output_definitions, merged_output)
print(f"File merging time: {time.time() - start_time:.2f} seconds")

print(f"Processing complete. Merged output saved to {merged_output}")


