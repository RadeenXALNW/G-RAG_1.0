import sqlite3
import pandas as pd
def display_table_data(table_name, db_path):
    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        print(f"Successfully connected to the database. Merging data from column 'c18' in table: {table_name}")
        
        # Merge all rows of 'c18' column into a single string
        merged_string = ' '.join(df['c18'].astype(str).tolist())
        
        return merged_string
    except sqlite3.Error as e:
        print(f"Error: Cannot connect to the database. {e}")
        return None