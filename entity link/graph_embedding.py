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
import aiohttp
import os
from typing import List, Dict




def read_text_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except IOError as e:
        print(f"Error reading file: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 1024, chunk_overlap: int = 5) -> List[str]:
    sentence_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return sentence_splitter.split_text(text)

def create_documents(input_string: str, metadata: Dict = None) -> List[Document]:
    chunks = chunk_text(input_string)
    return [Document(text=chunk, metadata=metadata or {}) for chunk in chunks]

def main(query_text: str, input_text: str):
    # Set up Neo4j graph store
    graph_store = Neo4jPGStore(
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        url=NEO4J_URL
    )

    # Set up Relik extractor
    relik = RelikPathExtractor(
        model="relik-ie/relik-cie-small",  # you should have access to the GPU with at least 24 GB VRAM and system ram about 60 GB
        skip_errors=True,
        num_workers=8,
        relationship_confidence_threshold=0.1,
        ignore_self_loops=False,
        model_config={"skip_metadata": False, "device": "cuda"}
    )
    
    # Set up Groq LLM
    llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)

    # Set up HuggingFace embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="dunzhang/stella_en_1.5B_v5",
        embed_batch_size=512,
        cache_folder=None,
        max_length=2048,
        device="cuda"
    )

    # Configure global settings
    Settings.llm = llm
    Settings.embed_model = embed_model

    # Create documents from input text
    docs = create_documents(input_text, metadata=None)
    print("###############docs_finished####################")

    # Batch processing for PropertyGraphIndex
    batch_size = 200
    index = None

    for i in range(300, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        
        if index is None:
            # Create initial index with first batch
            index = PropertyGraphIndex.from_documents(
                batch,
                kg_extractors=[relik],
                llm=llm,
                embed_model=embed_model,
                property_graph_store=graph_store,
                show_progress=True,
            )
        else:
            # Add documents to existing index
            index.insert_nodes(batch)
        
        print(f"Processed batch {i//batch_size + 1} of {(len(docs)-1)//batch_size + 1}")

    # Set up query engine
    query_engine = index.as_query_engine(include_text=True)

    # Perform query
    response = query_engine.query(query_text)
    return str(response)

# # Example usage
# if __name__ == "__main__":
#     input_file_path = "file.txt"
#     input_text = read_text_file(input_file_path)
#     query_text = "Your queries"
#     result = main(query_text, input_text)
#     print(result)

