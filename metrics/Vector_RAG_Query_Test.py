from llama_index.core.evaluation import (
    BatchEvalRunner,
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    ContextRelevancyEvaluator
)
import asyncio
import pandas as pd
from llama_index.extractors.relik import RelikPathExtractor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.indices import PropertyGraphIndex
from llama_index.core import Settings
from llama_index.llms.groq import Groq
import streamlit as st
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import (
    BatchEvalRunner,
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator
)
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex,ServiceContext

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

GROQ_API_KEY = "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"
JINA_API_KEY ="jina_4b8e936151714f37b7b5663386f9ed15lsFmfiLXOi3iSvGMvMSBTLguX9Mw"

# documents = SimpleDirectoryReader(input_files=["/teamspace/studios/this_studio/NeurIPS_Materials/documents_pdf/High-Entropy Alloys  A Critical Review.pdf"])

documents = SimpleDirectoryReader(input_dir='/teamspace/studios/this_studio/NeurIPS_Materials/documents_pdf').load_data()


llm=Groq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY
)
embed_model = JinaEmbedding(
    api_key=JINA_API_KEY,
    model="jina-embeddings-v2-base-en",
    embed_batch_size=4,

)

Settings.embed_model = embed_model
Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

text_splitter = TokenTextSplitter(
  chunk_size=512,
  chunk_overlap=20,
)

# node_parser = SimpleNodeParser.from_defaults(
#   text_splitter = TokenTextSplitter )

index = VectorStoreIndex.from_documents(
    documents
    )

query_engine = index.as_query_engine()
response = query_engine.query("What is Chromium?")
print(response)
