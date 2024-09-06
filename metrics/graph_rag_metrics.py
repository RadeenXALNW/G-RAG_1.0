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


#NEO4J PASS AND URL
#GROQ API KEY
#JINA API KEY (IF You Use via API)


import streamlit as st
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.llms.openai import OpenAI
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

# Load data from the specified directory

reader = SimpleDirectoryReader(input_files=["/teamspace/studios/this_studio/HEA SHORT PDF_entity.pdf"])
documents = reader.load_data()
reader_q = SimpleDirectoryReader(input_files=["/teamspace/studios/this_studio/HEA SHORT PDF.pdf"])
documents_q = reader_q.load_data()


# Generate questions from the loaded documents
dataset_generator = RagDatasetGenerator.from_documents(
    documents=documents_q,
    llm=Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
)


# Take only the first 3 questions
eval_questions = dataset_generator.generate_dataset_from_nodes()[:20]

print("#######eval question finished####")

def create_query_engine():
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
        skip_errors=True,
        num_workers=4,
        relationship_confidence_threshold=0.2,
        ignore_self_loops=False,
        model_config={"skip_metadata": False, "device": "cuda"}
    )

    # # Set up HuggingFace embedding model
    # embed_model = HuggingFaceEmbedding(
    #     model_name="dunzhang/stella_en_1.5B_v5",
    #     embed_batch_size=1,
    #     cache_folder=None,
    #     device="cuda"
    # )
    embed_model = JinaEmbedding(
    api_key=JINA_API_KEY,
    model="jina-embeddings-v2-base-en",
    embed_batch_size=8,

)


    # Configure global settings
    Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    Settings.embed_model = embed_model

    # Batch processing for PropertyGraphIndex
    # batch_size = 4012
    # index = None

    # for i in range(0, len(documents), batch_size):
    #     batch = documents[i:i+batch_size]
        
    #     if index is None:
    #         # Create initial index with first batch
    #         index = PropertyGraphIndex.from_documents(
    #             batch,
    #             kg_extractors=[relik],
    #             llm=Settings.llm,
    #             embed_model=embed_model,
    #             property_graph_store=graph_store,
    #             show_progress=True,
    #         )
    #     else:
    #         # Add documents to existing index
    #         index.insert_nodes(batch)
        
    #     print(f"Processed batch {i//batch_size + 1} of {(len(documents)-1)//batch_size + 1}")
    index = PropertyGraphIndex.from_documents(
    documents,
    kg_extractors=[relik],
    llm=Settings.llm,
    embed_model=embed_model,
    property_graph_store=graph_store,
    show_progress=True,
)

    print(f"Processed all {len(documents)} documents at once")

    # Set up query engine
    return index.as_query_engine(include_text=True)

async def evaluate_async(query_engine):
    groq_model = Groq(
        system_prompt="You are a helpful material assistant. When asked a question, you must answer from the data documents. \
        Don't ask tough question.\
        If you don't know the answer, say 'Oh, snap! It seems I've hit a road bump in my knowledge highway. \
        No worries, though! How about we detour to another fantastic journey waiting for you in the directory?'. \
        If you know the answer, please provide trip information not in a list but in text.", 
        model="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
        max_tokens=150
    )

    # Initialize the evaluators
    correctness_evaluator = CorrectnessEvaluator(llm=groq_model)
    faithfulness_evaluator = FaithfulnessEvaluator(llm=groq_model)
    relevancy_evaluator = RelevancyEvaluator(llm=groq_model)
    context_relevancy_evaluator=ContextRelevancyEvaluator(llm=groq_model)


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
        query_engine=query_engine,
        queries=[question.query for question in eval_questions]
    )

    return eval_result

def main():
    # Create the query engine
    query_engine = create_query_engine()

    # Run the asynchronous evaluation
    result = asyncio.run(evaluate_async(query_engine))

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
    with pd.ExcelWriter('eval_report2_graph_entity_new.xlsx', engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)

if __name__ == "__main__":
    main()