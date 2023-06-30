import torch
from torch import nn

class CountVectorizer(nn.Module):
    def __init__(self, d_model, vocab):
        super(CountVectorizer, self).__init__()
        self.vocab = vocab
        self.word_to_idx = {word: i for i, word in enumerate(vocab)}
        self.proj = nn.Linear(len(vocab), d_model)
        
    def forward(self, input_texts):
        count_vectors = []
        
        for input_text in input_texts:
            word_counts = torch.zeros(len(self.vocab))
            
            for word in input_text.split():
                word=word.lower()
                if word in self.vocab:
                    word_counts[self.vocab.index(word)] += 1
                else:
                    word_counts[self.vocab.index("[unknown]")] += 1
            count_vectors.append(word_counts)
        
        count_vectors = torch.stack(count_vectors, dim=0) 
        count_vectors = count_vectors.to(self.proj.weight.device)
        
        return self.proj(count_vectors).unsqueeze(1)

