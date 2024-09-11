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


# relik = RelikPathExtractor(
#     # model="relik-ie/relik-relation-extraction-small", 
#     # model="relik-ie/relik-cie-small", 
#     model="sapienzanlp/relik-relation-extraction-nyt-large",
#     skip_errors=True,
#     num_workers=8,
#     relationship_confidence_threshold=0.2,
#     # model="relik-ie/relik-reader-deberta-v3-small-cie-wikipedia",
#     model_config={"skip_metadata": False,"device":"cuda"})

# print(create_documents("High entropy alloys:   A focused review of mechanical properties and deformation mechanisms"))

def main(query_text: str, input_text: str):
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
    # return relik
    
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


# if __name__ == "__main__":
#     # Example usage
#     query = "What is the capital of France?"
#     input_text = """
#     Paris is the capital and most populous city of France, with an estimated population 
#     of 2,175,601 residents as of 2018, in an area of more than 105 square kilometres 
#     (41 square miles). Since the 17th century, Paris has been one of Europe's major 
#     centres of finance, diplomacy, commerce, fashion, science, and arts.
#     """
#     print(main(query, input_text))

if __name__ == "__main__":
    # Example usage
    query = "What is cocktail effect?"
    input_text=read_text_file("/teamspace/studios/this_studio/merged_output1.txt")
    # input_text=" High entropy alloys:   A focused review of mechanical properties and deformation mechanisms"
    # input_text1=""" 1  High entropy alloys:   A focused review of mechanical properties and deformation mechanisms   E.P. George1,2, W.A. Curtin3, C.C. Tasan4, *  1Oak Ridge National Laboratory, Materials Science and Technology Division, Oak Ridge, Tennessee 37831-6115,   USA;  2Department of Materials Science and Engineering, University of Tennessee, Knoxville, Tennessee 37996-2100, USA  3Laboratory for Multiscale Mechanics Modeling, Ecole Polytechnique Fédérale de Lausanne, Lausanne CH-1015,   Switzerland  4Department of Materials Science and Engineering, Massachusetts Institute of Technology,     
    # Abstract   The high-entropy alloy (HEA) concept was based on the idea that high mixing entropy can promote formation of stable   single-phase  microstructures.  During  the  past  15  years,  various  alloy  systems  have  been  explored  to  identify  HEA   systems with improved property combinations, leading to an extraordinary growth of this field. In the large pool of   alloys  with  varying  characteristics,  the  first  single-phase  HEA  with  good  tensile  properties,  the  equiatomic   CrMnFeCoNi alloy has become the benchmark material,   amental   understanding of HEA mechanical behavior. As the field is evolving to the more broadly defined complex concentrated   alloys  (CCAs)  and  the  available  data  in  the  literature  increase  exponentially,  a  fundamental  question remains   unchanged: how special are these new materials? In the first part of this review, select mechanical properties of HEAs   and CCAs are compared with those of conventional engineering alloys. This task is difficult because of the limited   tensile data available  for HEAs and CCAs.  Additiona Nonetheless,  our  evaluations  have  not  revealed  many  HEAs  or  CCAs  with   properties  far  exceeding  those  of  conventional  engineering  alloys,  although  specific  alloys  can  show  notable   enhancements in specific properties. Consequently, it is reasonable to first approach the understanding of HEAs and   CCAs through the assessment of how the well-established deformation mechanisms in conventional alloys operate or   are  modified  in  the  presence  of  the  high  local  complexity  of  the  HEAs  and  CCAs.    The  second  part  of  the  paper   provides a detailed review of the deformation mechanisms of HEAs with the FCC and BCC structures. For the former,   we  chose  the  CrMnFeCoNi  (Cantor)  alloy  because  it  is  the  alloy  on  which  the  most  rigorous  and  thorough   investigations have been performed and, for the latter, we chose the TiZrHfNbTa (Senkov) alloy because this is one of   the few refractory HEAs that exhibits any tensile ductility at room temperature. 
    # As expected, our review shows that the   fundamental deformation mechanisms in these systems   , are broadly   similar  to  those  of  conventional  FCC  and  BCC  metals.  The  third  part  of  this  review  examines  the  theoretical  and   modeling  efforts  to  date  that  seek  to  provide  either  qualitative  or  quantitative  understanding  of  the mechanical   performance of FCC and BCC HEAs.  Since experiments reveal no fundamentally new mechanisms of deformation,   this section starts with an overview of modeling perspectives and fundamental considerations.  The review then turns to   the  evolution  of  modeling  and  predictions  as  compar   es  and   limitations.  Finally, in spite  of some significant successes, important directions  for further theory development are   discussed.   Overall, while the individual deformation mechanisms or properties of the HEAs and CCAs are not, by and   large,  special  relative  to  conventional  alloys,  the  present HEA  rush  remains  valuable  because  the  compositional   freedom that comes from the multi-element space will allow exploration of whether multiple mechanisms can operate   sequentially  or  simultaneously,  which  may  yet  lead     Keywords: solid solution; solute strengthening; ductility; microstructure design; plasticity mechanisms        
    # 1. Introduction   The  HEA  concept  has  created  an  enormous,  worldwide  drive  for  alloy  design,  one  that  is   unprecedented in the history of metallurgical research. Soon after the introduction of the original   concept which proposed that maximizing configurational entropy can favor the formation of stable,   single-phase,  substitutional  solid  solutions,  research  focusing  on  alloys  that  violate  the  founding  © 2019 published by Elsevier. This manuscript is made available under the Elsevier user license https://www.elsevier.com/open-access/userlicense/1.0/ Version of Record: https://www.sciencedirect.com/science/article/pii/S1359645419308444 Manuscript_c0ceecf0a8c4037579ea27706d1446fd """
    print(main(query, input_text))

# input_text1=""" Material science is a multidisciplinary field that sits at the intersection of physics, chemistry, 
# and engineering, focusing on the study, design, and application of materials in various forms. 
# It is a domain that investigates the properties of materials—such as metals, ceramics, polymers, 
# and composites—and how these properties can be manipulated to create new materials with desired characteristics. 
# The field is integral to advancements in technology and industry, driving innovations in everything from electronics and aerospace to biomedical devices 
# and sustainable energy solutions. 
# By understanding the atomic and molecular structure of materials, scientists and engineers can develop materials with specific mechanical, 
# electrical, thermal, and optical properties, leading to breakthroughs like stronger and lighter alloys, more efficient semiconductors, and biodegradable plastics. 
# Material science also plays a critical role in addressing global challenges, such as the development of materials for renewable energy storage, 
# improving the performance of batteries, 
# and creating materials that can withstand extreme environments. The field is ever-evolving, with researchers constantly exploring new ways to combine different 
# materials at the nanoscale, 
# leading to the creation of smart materials that can adapt to their environment, self-healing materials that can repair themselves, 
# and materials with unprecedented levels of strength, flexibility, or conductivity. As we continue to push the boundaries of what materials can do, 
# the impact of material science on society is profound, enabling the development of new technologies that improve the quality of life, 
# enhance industrial processes, and contribute to sustainable development. """

# from pprint import pprint 
# print(chunk_text(input_text1))



