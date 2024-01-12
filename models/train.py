import nltk
import torch
from torch import nn
from dataset import get_data
from model import NLPModel

class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, epochs: int, lr: float):
        print(f"Device set to {self.device}")

        try:
            dataloader, vocabulary, tags = get_data(batch_size=2)
        except LookupError:
            nltk.download("punkt")
            dataloader, vocabulary, tags = get_data(batch_size=2)

        model = self.initialize_model(vocabulary, tags)
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

        model.train()
        for epoch in range(epochs):
            for _, (sentence, label) in enumerate(dataloader):
                sentence, label = sentence.to(self.device), label.to(self.device).long()

                prediction = model(sentence)

                loss = loss_func(prediction, label.long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % 50 == 0:
                print(f"Epoch: {epoch} | Loss: {loss.item():.2f}")

        self.save_model(model, vocabulary, tags)

    def initialize_model(self, vocabulary, tags):
        model = NLPModel(input_size=len(vocabulary), hidden_size=16, output_size=len(tags)).to(self.device)
        return model

    def save_model(self, model, vocabulary, tags):
        data_to_save = {
            "model_state_dict": model.state_dict(),
            "input_size": len(vocabulary),
            "hidden_size": 16,
            "output_size": len(tags),
            "vocabulary": vocabulary,
            "tags": tags
        }

        torch.save(data_to_save, "models/checkpoint/nlp_model.pth")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(epochs=2000, lr=0.001)