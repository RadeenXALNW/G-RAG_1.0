import os
import sqlite3
import pandas as pd
from llmware.library import Library
from llmware.setup import Setup
from llmware.configs import LLMWareConfig, MilvusConfig
import logging
import json
from database_reader import display_table_data

# Set logging to WARNING to reduce output
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def document2text_extraction(library_name,files,pdf_folder_path):
    print("\nupdate: Step 1 - Creating library: {}".format(library_name))
    library = Library().create_new_library(library_name)
    print("update: Step 2 - Adding uploaded files")
    for file_path in files:
        library.add_files(input_folder_path=pdf_folder_path, chunk_size=3000, max_chunk_size=3200, smart_chunking=2)
    print("update: Step 3 - Retrieving data from the database")
    current_dir = os.getcwd()
    print(current_dir)

    db_path=os.path.join("/teamspace/studios/this_studio/NeurIPS_Materials/data1/", 'sqlite_llmware.db')
    db_path = os.path.normpath(db_path)
    print(db_path)

    df = display_table_data(f'{library_name}_content', db_path)
    return df

