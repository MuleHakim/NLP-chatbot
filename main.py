import os
import warnings
from typing import Dict
import nltk
import json
import random
import torch
from models.model import NLPModel
from models.utils import tokenize_sentence, bag_of_words
from openfabric_pysdk.utility import SchemaUtil
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText
from openfabric_pysdk.context import Ray, State
from openfabric_pysdk.loader import ConfigClass

class NLPHandler:
    def __init__(self):
        self.load_model()

    def load_model(self):
        with open("aims/aims.json", "r") as f:
            self.aims = json.load(f)

        data = torch.load("models/checkpoint/nlp_model.pth")
        self.model = NLPModel(data['input_size'], data['hidden_size'], data['output_size'])
        self.model.load_state_dict(data['model_state_dict'])
        self.model.eval()
        self.vocabulary = data['vocabulary']

    def config(self, configuration: Dict[str, ConfigClass], state: State):
        # TODO: Add configuration-related code here
        pass

    def execute(self, request: SimpleText, ray: Ray, state: State) -> SimpleText:
        output = []
        for text in request.text:
            response = self.get_response(text)
            output.append(response)

        return SchemaUtil.create(SimpleText(), dict(text=output))

    def get_response(self, question: str) -> str:
        try:
            preprocessed_question = self.preprocess_question(question)
        except LookupError:
            nltk.download("punkt")
            preprocessed_question = self.preprocess_question(question)

        with torch.inference_mode():
            predictions = self.model(preprocessed_question)

        predictions = torch.softmax(predictions, dim=1)
        probability, index = torch.max(predictions, dim=1)

        if probability.item() < 0.8:
            return "I don't know the answer to this question"

        response = random.choice(self.aims['aims'][index.item()]['responses'])
        return response

    def preprocess_question(self, question: str) -> torch.Tensor:
        bow = bag_of_words(tokenize_sentence(question), self.vocabulary)
        bow = bow.reshape(1, bow.shape[0])
        return torch.from_numpy(bow)

if __name__ == "__main__":
    nlp_handler = NLPHandler()
