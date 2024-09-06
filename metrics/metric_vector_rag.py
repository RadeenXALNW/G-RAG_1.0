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
import asyncio
import pandas as pd
from llama_index.extractors.relik import RelikPathExtractor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.indices import PropertyGraphIndex
from llama_index.core import Settings
from llama_index.llms.groq import Groq

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


import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex,ServiceContext

from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.llms.groq import Groq
from llama_index.core.evaluation import (
    BatchEvalRunner,
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    ContextRelevancyEvaluator
)
import asyncio
import pandas as pd

# Load data from the specified directory
reader = SimpleDirectoryReader(input_files=["your-folder-name"])
documents = reader.load_data()

embed_model1 = JinaEmbedding(
    api_key=JINA_API_KEY,
    model="jina-embeddings-v2-base-en",
    embed_batch_size=4,

)

# Generate questions from the loaded documents
dataset_generator = RagDatasetGenerator.from_documents(
    documents=documents,
    llm=Groq(
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY
    )
)

# Take only the first 3 questions
eval_questions = dataset_generator.generate_dataset_from_nodes()[:10]


Settings.embed_model = embed_model1
Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

# Create a vector index from the loaded documents
vector_index = VectorStoreIndex.from_documents(documents)

# Initialize the Groq model
groq_model = Groq(
    system_prompt="You are a helpful material assistant. When asked a question, you must answer from the data documents. \
        If you don't know the answer, say 'Oh, snap! It seems I've hit a road bump in my knowledge highway. \
        No worries, though! How about we detour to another fantastic journey waiting for you in the directory?'. \
        If you know the answer, please provide trip information not in a list but in text.",
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    max_tokens=500
)

# Initialize the evaluators
correctness_evaluator = CorrectnessEvaluator(llm=groq_model)
faithfulness_evaluator = FaithfulnessEvaluator(llm=groq_model)
relevancy_evaluator = RelevancyEvaluator(llm=groq_model)
context_relevancy_evaluator=ContextRelevancyEvaluator(llm=groq_model)

# Define an asynchronous function for evaluation
async def evaluate_async():
    # Initialize the BatchEvalRunner
    runner = BatchEvalRunner(
        {
            "correctness": correctness_evaluator,
            "faithfulness": faithfulness_evaluator,
            "relevancy": relevancy_evaluator,
            "context_relevancy": context_relevancy_evaluator
        },
        show_progress=True
    )

    # Run the asynchronous evaluation
    eval_result = await runner.aevaluate_queries(
        query_engine=vector_index.as_query_engine(),
        queries=[question.query for question in eval_questions]
    )

    return eval_result

# Run the asynchronous function using asyncio
result = asyncio.run(evaluate_async())

# Extract relevant information from the evaluation results
data = []
for i, question in enumerate(eval_questions):
    correctness_result = result['correctness'][i]
    faithfulness_result = result['faithfulness'][i]
    relevancy_result = result['relevancy'][i]
    context_relevancy_result=result['context_relevancy'][i]
    data.append({
        'Query': question.query,
        'Correctness response': correctness_result.response,
        'Correctness passing': correctness_result.passing,
        'Correctness feedback': correctness_result.feedback,
        'Correctness score': correctness_result.score,
        'Faithfulness response': faithfulness_result.response,
        'Faithfulness passing': faithfulness_result.passing,
        'Faithfulness feedback': faithfulness_result.feedback,
        'Faithfulness score': faithfulness_result.score,
        'Relevancy response': relevancy_result.response,
        'Relevancy passing': relevancy_result.passing,
        'Relevancy feedback': relevancy_result.feedback,
        'Relevancy score': relevancy_result.score,
        'Context Relevancy score': context_relevancy_result.score
    })

# Create a pandas DataFrame
df = pd.DataFrame(data)

# Save the pandas DataFrame to an Excel file using xlsxwriter
with pd.ExcelWriter('eval_report_norm_new.xlsx', engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Sheet1', index=False)