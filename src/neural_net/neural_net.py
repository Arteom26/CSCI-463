import evaluate
import numpy as np

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import pipeline

class NeuralNet:
    MODEL_NAME = "facebook/bart-large-cnn"
    
    def __init__(self) -> None:
        self._summarizer_pipeline = pipeline(task="summarization", model=NeuralNet.MODEL_NAME)
        
    def summarize_text(self, text: str) -> list:
        return self._summarizer_pipeline(text)
    