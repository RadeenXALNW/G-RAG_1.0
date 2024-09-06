import os
from typing import List
from groq import Groq
from dotenv import load_dotenv

# Import your custom functions
from text_chunk import SpacySentenceSplitter
from semantic_split import SimilarSentenceSplitter, SentenceTransformersSimilarity

from txt_reader import read_text_file
from relik_rag import split_text


client = Groq(
    api_key="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"
)

structure="""
    Chemical Component:
    Main Points:
"""
bibilography_structure= """
[169]  C. Ng, S. Guo, J. Luan, Q. Wang, J. Lu, S. Shi, C.T. Liu, Phase stability and tensile   properties of Co-free Al 0.5CrCuFeNi2 high-entropy alloys, J. Alloys Compd. 584 (2014) 530537. doi:10.1016/j.jallcom.2013.09.105.
[170]  C.C. Juan, M.H. Tsai, C.W. Tsai, W.L. Hsu, C.M. Lin, S.K. Chen, S.J. Lin, J.W. Yeh,   Simultaneously increasing the strength and ductility of a refractory high-entropy alloy via   grain refining, Mater. Lett. 184 (2016) 200203. doi:10.1016/j.matlet.2016.08.060.   
"""

empty_response=" "

def process_chunk(chunk: str) -> str:
    """
    Process a single chunk of text using the Groq API.
    
    Args:
    chunk (str): A chunk of text to be processed.
    
    Returns:
    str: Processed text focusing on material science components.
    """
    stream = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant specializing in text processing. 
                Your task is to remove any bibliography entries from the given text chunk.
                Bibliography entries typically start with a number in square brackets, followed by author names, title, and publication details.
                Return only the text content without any bibliography entries.
                If the entire chunk is a bibliography, return an empty string.
                Preserve all other content that is not part of the bibliography."""

            },
            {
                "role": "user",
                "content": chunk
            }
        ],
        model="llama-3.1-70b-versatile",
        temperature=0.7,
        max_tokens=2048,
        top_p=1,
        stop=None,
        stream=True,
    )

    processed_text = ""
    for chunk in stream:
        processed_text += chunk.choices[0].delta.content or ""

    return processed_text

# def main(input_file: str, output_file: str):
#     """
#     Main function to process the input file and save the results.
    
#     Args:
#     input_file (str): Path to the input text file.
#     output_file (str): Path to the output text file.
#     """
#     # Read the input file
#     with open(input_file, 'r') as file:
#         input_text = file.read()

#     # Split the text into chunks
#     chunks = split_text(input_text)

#     # Process each chunk and save the results
#     with open(output_file, 'w') as file:
#         for i, chunk in enumerate(chunks):
#             processed_chunk = process_chunk(chunk)
#             file.write(processed_chunk + "\n\n")
#             print(f"Processed chunk {i+1}/{len(chunks)}")

#     print(f"Processing complete. Results saved to {output_file}")

# if __name__ == "__main__":
#     input_file = "/teamspace/studios/this_studio/summarization_output/image_summaries_speed1.txt"
#     output_file = "/teamspace/studios/this_studio/groq_assist1.txt"
#     main(input_file, output_file)
# If there is no meaningful context in the content, you will just give ''. You don't add anything like {no context provided etc.}
#                 Don't add your own question or any apologetic answer.



                # "content": f"""You are a helpful material science expert. 
                # Your task is to analyze the given text and extract information related to material science.
                # Format your response using the following structure:
                # {structure}
                
                # Guidelines:
                # 1. For each chemical component identified, list its main points.
                # 2. Focus on extracting meaningful information relevant to material science.
                # 3. If no relevant information is found, respond with exactly "{empty_response}" (without quotes).
                # 4. Do not include any apologetic text, questions, or explanations if no information is found.
                # 5. Ensure your response fits the given structure precisely.
                # """