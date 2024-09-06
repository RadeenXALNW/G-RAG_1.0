import pandas as pd
def extract_metadata(df):
    metadata = {}

    # Number of columns
    metadata['Number of Columns'] = df.shape[1]

    # Column names
    metadata['Schema'] = df.columns.tolist()

    # Data types of each column
    metadata['Data Types'] = str(df.dtypes)

    # Summary statistics
    metadata['Sample'] = df.head(1).to_dict(orient="records")

    return metadata
df = pd.read_csv("/tmp/table_0.csv", sep=",", quotechar='"')

metadata = extract_metadata(df)