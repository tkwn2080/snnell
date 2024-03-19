import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from pathlib import Path

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return self.rnn.init_hidden(batch_size)


def load_data(file_path, seq_length):
    with open(file_path, 'r') as f:
        text = f.read()
    
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    
    encoded = np.array([char2int[ch] for ch in text])
    
    inputs = []
    targets = []
    
    for i in range(0, len(encoded) - seq_length):
        inputs.append(encoded[i:i+seq_length])
        targets.append(encoded[i+1:i+seq_length+1])
    
    return inputs, targets, char2int, int2char

def get_batches(arr, batch_size, seq_length):
    total_batch_size = batch_size * seq_length
    n_batches = len(arr) // total_batch_size
    
    arr = arr[:n_batches * total_batch_size]
    arr = arr.reshape((batch_size, -1))
    
    for n in range(0, arr.shape[1], seq_length):
        x = arr[:, n:n+seq_length]
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y

def train(model, data, epochs, batch_size, seq_length, lr):
    model.train()
    optimizer = nn.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.losses.cross_entropy

    for epoch in range(epochs):
        hidden = model.init_hidden(batch_size)

        for x, y in get_batches(data, batch_size, seq_length):
            optimizer.zero_grad()
            
            x = mx.array(x)
            y = mx.array(y)
            hidden = tuple([each.data for each in hidden])

            output, hidden = model(x, hidden)
            loss = criterion(output.transpose(1, 2), y)

            loss.backward()
            optimizer.step()

def predict(model, char, char2int, int2char, hidden=None, top_k=None):
        x = np.array([[char2int[char]]])
        x = mx.array(x)
        
        hidden = tuple([each.data for each in hidden])
        out, hidden = model(x, hidden)

        p = nn.softmax(out).data
        if top_k is None:
            top_ch = np.arange(len(char2int))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
        
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())
        
        return int2char[char], hidden

def main(train_file='./seahawks.txt', epochs=10, lr=0.001, batch_size=64, seq_length=100, embed_dim=128, hidden_dim=256, num_layers=2):
    # Load data
    inputs, targets, char2int, int2char = load_data(train_file, seq_length)
    data = np.array(inputs), np.array(targets)

    vocab_size = len(char2int)

    # Initialize model, optimizer, and loss
    model = CharRNN(vocab_size, embed_dim, hidden_dim, num_layers)
    optimizer = optim.Adam(model.parameters(), learning_rate=lr)  # Corrected parameter name
    criterion = nn.losses.cross_entropy

    # Training loop
    train(model, data, epochs, batch_size, seq_length, lr, optimizer, criterion, char2int, int2char)


if __name__ == "__main__":
    main()