import json
import numpy as np
from utils import tokenize_sentence, stem_word, bag_of_words
from torch.utils.data import Dataset, DataLoader

class NLPDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.data_length = len(x_data)
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.data_length

def get_data(batch_size: int):
    with open("./aims/aims.json", "r") as f:
        aims = json.load(f)
    
    all_words = set()
    tags = list()
    training_data = list()
    
    for aim in aims['aims']:
        tag = aim['tag']
        
        tags.append(tag)
        for pattern in aim['patterns']:
            stemmed_tokenized_sentence = [ stem_word(word) for word in tokenize_sentence(pattern)]
            all_words.update(stemmed_tokenized_sentence)
            
            training_data.append((stemmed_tokenized_sentence, tag))
    
    all_words = list(all_words)
    
    x_train = []
    y_train = []
    
    for (sentence, tag) in training_data:
        bow = bag_of_words(sentence, all_words)
        x_train.append(bow)
        
        label = tags.index(tag)
        y_train.append(label)
    
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    
    dataset = NLPDataset(x_train, y_train)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, all_words, tags