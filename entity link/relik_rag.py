import os
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.core import Document, PropertyGraphIndex, Settings
from llama_index.extractors.relik.base import RelikPathExtractor
from llama_index.llms.groq import Groq
from llama_index.embeddings.jinaai import JinaEmbedding
from text_chunk import SpacySentenceSplitter
from semantic_split import SimilarSentenceSplitter, SentenceTransformersSimilarity

from txt_reader import read_text_file
from typing import List
import os
from llama_index.graph_stores.neo4j import Neo4jPGStore
from llama_index.core import Document, PropertyGraphIndex, Settings
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.extractors.relik.base import RelikPathExtractor
from llama_index.llms.groq import Groq
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm
import aiohttp





def split_text(input_text: str) -> List[str]:
    """
    Split the input text into chunks and then convert the resulting 2D list into a 1D list.
    
    Args:
    input_text (str): The text to be split into chunks.
    
    Returns:
    List[str]: A 1D list where each element is a chunk of text.
    """
    # Initialize the necessary objects
    model = SentenceTransformersSimilarity()
    sentence_splitter = SpacySentenceSplitter()
    splitter = SimilarSentenceSplitter(model, sentence_splitter)
    chunked_text = splitter.split(input_text)
    # Convert 2D list to 1D list
    flattened_text = [' '.join(chunk) if isinstance(chunk, list) else chunk for chunk in chunked_text]
    
    return flattened_text

def create_documents(input_string: str, metadata: dict = None) -> list[Document]:
    """
    Create Document objects from a large input string after chunking.
    """
    chunks = split_text(input_string)
    return [Document(text=chunk, metadata=metadata or {}) for chunk in chunks]

def chunk_text(input_text: str, chunk_size: int = 100, chunk_overlap: int =20) -> list[str]:
    """
    Split the input text into chunks.
    """
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(input_text)







