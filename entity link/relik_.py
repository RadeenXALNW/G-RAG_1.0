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


# API keys and credentials (replace with your actual keys)
# NEO4J_USERNAME = "neo4j"
# NEO4J_PASSWORD = 
# NEO4J_URL = 

# GROQ_API_KEY = 
# JINA_API_KEY = 




def read_text_file(file_path: str) -> str:
    """
    Read a text file and return its content as a string.
    
    Args:
    file_path (str): The path to the text file.
    
    Returns:
    str: The content of the text file as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except IOError as e:
        print(f"Error reading file: {e}")
        return ""


def chunk_text(text: str, chunk_size: int =1024, chunk_overlap: int = 5) -> list[str]:
    """
    Split the input text into chunks.
    """
    # text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    sentence_splitter=SentenceSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    # return text_splitter.split_text(text)
    return sentence_splitter.split_text(text)

def create_documents(input_string: str, metadata: dict = None) -> list[Document]:
    """
    Create Document objects from a large input string after chunking.
    """
    chunks = chunk_text(input_string)
    return [Document(text=chunk, metadata=metadata or {}) for chunk in chunks]


def main(query_text: str, input_text: str):
    print("start")
    # Set up Neo4j graph store
    graph_store = Neo4jPGStore(
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        url=NEO4J_URL
    )

    # Set up Relik extractor
    relik = RelikPathExtractor(
        # model="relik-ie/relik-relation-extraction-small", 
        model="relik-ie/relik-cie-small", 
        # model="sapienzanlp/relik-entity-linking-large",
        # model="sapienzanlp/relik-relation-extraction-nyt-large",
        skip_errors=True,
        num_workers=8,
        relationship_confidence_threshold=0.1,
        ignore_self_loops=False,
        # model="relik-ie/relik-reader-deberta-v3-small-cie-wikipedia",
        model_config={"skip_metadata": False,"device":"cuda"}
    )
    
    # Set up Groq LLM
    llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)

    # # Set up Jina embedding model
    # embed_model1 = JinaEmbedding(
    #     api_key=JINA_API_KEY,
    #     model="jina-embeddings-v2-base-en",
    #     embed_batch_size=2,

    # )
    
    embed_model1 = HuggingFaceEmbedding(
        # model_name="jinaai/jina-embeddings-v2-base-en",
        model_name="dunzhang/stella_en_1.5B_v5",
        # max_length=1024,
        embed_batch_size=128,
        cache_folder=None,
        max_length=2048,
        device="cuda"
    )
    # # Configure global settings
    Settings.llm = llm
    Settings.embed_model1 = embed_model1



    # Create documents from input text
    
    docs = create_documents(input_text, metadata=None)
    print("###############docs_finished####################")

    # Create PropertyGraphIndex
    index = PropertyGraphIndex.from_documents(
        docs,
        kg_extractors=[relik],
        llm=llm,
        embed_model=embed_model1,
        property_graph_store=graph_store,
        show_progress=True,
    )

    # Set up query engine
    query_engine = index.as_query_engine(include_text=True)

    # Perform query
    response = query_engine.query(query_text)
    return str(response)



if __name__ == "__main__":
    # Example usage
    query = "What is cocktail effect?"
    input_text=read_text_file("materials.txt")
    print(main(query, input_text))