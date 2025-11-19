import random
import torch 
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import time


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def read_data():
    path_pos = f"./pos_examples.txt" 
    path_neg = f"./neg_examples.txt" 
    
    # Read lines from each file
    with open(path_pos, 'r') as f:
        pos = [line.strip() for line in f.readlines()]
    
    with open(path_neg, 'r') as f:
        neg = [line.strip() for line in f.readlines()]
    
    # Add labels
    labeled = [(x, 1) for x in pos] + [(x, 0) for x in neg]

    # Shuffle and split
    random.shuffle(labeled)
    # Split into train/val/test
    train_size = int(0.7 * len(labeled))   # 70% train
    val_size = int(0.15 * len(labeled))    # 15% validation

    train_data = labeled[:train_size]
    val_data = labeled[train_size:train_size + val_size]
    test_data = labeled[train_size + val_size:]
    
    return train_data, val_data, test_data


class AcceptorModel(nn.Module):
    def __init__(self, vocab_size,embedding_dim, hidden_size, hidden_dim_fc):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTMCell(embedding_dim, hidden_size)
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(hidden_size, hidden_dim_fc)
        self.fc2 = nn.Linear(hidden_dim_fc, 2)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape
        device = input_seq.device 
        h_t = torch.zeros(batch_size, self.hidden_size, device=device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=device)


        for t in range(seq_len):
            x_t = self.embedding(input_seq[:, t])  # [batch_size, embedding_dim]
            h_t, c_t = self.lstm(x_t, (h_t, c_t))

        x = self.fc1(h_t)
        x = torch.relu(x)
        x = self.fc2(x)   # output logits for 2 classes
        return x


class SequenceDataset(Dataset):
    def __init__(self, data, char_to_idx):
            self.data = data
            self.char_to_idx = char_to_idx

    def __len__(self):
            return len(self.data)

    def __getitem__(self, idx):
            seq_str, label = self.data[idx]
            indices = torch.tensor([self.char_to_idx[c] for c in seq_str], dtype=torch.long)
            return indices, label


def char2idx(data):
    char_idx_dict = {}
    # Extract all unique characters from the dataset
    unique_chars = sorted(set("".join(seq for seq, _ in data)))
    char_idx_dict = {"<PAD>": 0}
    # Assign an index to each character
    for i, char in enumerate(unique_chars, start=1):
         char_idx_dict[char] = i
    return char_idx_dict


def custom_collate_fn(batch):
    sequences, labels = zip(*batch)  # separate input and label
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)  # [B, max_len]
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, labels


def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_x)  # shape: [batch, 2]
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


if __name__== "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.CrossEntropyLoss()
    embedding_dim = 30
    hidden_size = 30
    hidden_dim_fc = 100
    train_data, val_data, test_data = read_data()
    char_to_idx= char2idx(train_data + val_data + test_data)
    model = AcceptorModel(len(char_to_idx),embedding_dim, hidden_size, hidden_dim_fc).to(device)         # Move model to GPU if available
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    train_dataset = SequenceDataset(train_data, char_to_idx)
    val_dataset = SequenceDataset(val_data, char_to_idx)
    test_dataset = SequenceDataset(test_data, char_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)
        
    print(f"\nTraining")
    start_time = time.time()

    for epoch in range(8):  
        train_loss, train_acc = train(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch:2d}: "
                f"Train Loss = {train_loss:.4f}, Acc = {train_acc:.4f} | "
                f"Val Loss = {val_loss:.4f}, Acc = {val_acc:.4f}")
    end_time = time.time()
    elapsed = end_time - start_time
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Training time: {elapsed:.2f} seconds\n")