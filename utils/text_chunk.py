from semantic_split import SimilarSentenceSplitter, SentenceTransformersSimilarity
from abc import ABC, abstractmethod
from typing import List
import spacy

class Splitter(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def split(self, text: str) -> List[str]:
        pass

class SpacySentenceSplitter(Splitter):
    def __init__(self, model="en_core_web_sm", max_length=1500000):
        self.nlp = spacy.load(model)
        self.nlp.max_length = max_length  # 

    def split(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [str(sent).strip() for sent in doc.sents]