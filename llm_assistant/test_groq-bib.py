import re
from typing import List
import os
from typing import List
from groq import Groq
from dotenv import load_dotenv


client = Groq(
    api_key="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"
)


def is_bibliography_entry(text: str) -> bool:
    """
    Check if the given text matches the pattern of a bibliography entry.
    """
    # Pattern: [number] followed by authors, title, and other details
    pattern = r'^\[\d+\]\s+.+\.\s+.+\.\s+.+\.'
    return bool(re.match(pattern, text.strip()))

def remove_bibliography(chunk: str) -> str:
    """
    Remove bibliography entries from the given chunk of text.
    
    Args:
    chunk (str): A chunk of text to be processed.
    
    Returns:
    str: Processed text with bibliography entries removed.
    """
    lines = chunk.split('\n')
    non_bibliography_lines = [line for line in lines if not is_bibliography_entry(line)]
    return '\n'.join(non_bibliography_lines)

def process_chunk(chunk: str) -> str:
    """
    Process a single chunk of text using the Groq API.
    
    Args:
    chunk (str): A chunk of text to be processed.
    
    Returns:
    str: Processed text focusing on material science components.
    """
    # First, remove bibliography entries
    chunk_without_bibliography = remove_bibliography(chunk)
    
    # Then, use the LLM to process the remaining content
    stream = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are a helpful material science expert. 
                                Your task is to summarize the given text, focusing on material science aspects.
                                Ensure that you maintain proper sentence structure and meaningful content."""
            },
            {
                "role": "user",
                "content": chunk_without_bibliography
            }
        ],
        model="mixtral-8x7b-32768",
        temperature=0.5,
        max_tokens=500
    )
    
    return stream.choices[0].message.content

# Example usage
chunk = """
This is some text about material science.
[169]  C. Ng, S. Guo, J. Luan, Q. Wang, J. Lu, S. Shi, C.T. Liu, Phase stability and tensile   properties of Co-free Al 0.5CrCuFeNi2 high-entropy alloys, J. Alloys Compd. 584 (2014) 530537. doi:10.1016/j.jallcom.2013.09.105.
More text about material properties.
[170]  C.C. Juan, M.H. Tsai, C.W. Tsai, W.L. Hsu, C.M. Lin, S.K. Chen, S.J. Lin, J.W. Yeh,   Simultaneously increasing the strength and ductility of a refractory high-entropy alloy via   grain refining, Mater. Lett. 184 (2016) 200203. doi:10.1016/j.matlet.2016.08.060.   
Concluding remarks about the study.
"""

processed_chunk = process_chunk(chunk)
print(processed_chunk)