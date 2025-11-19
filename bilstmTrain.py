import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def load_data(file_path):
    documents = load_documents(file_path)
    return documents

def load_documents(filename):
    data = []
    with open(filename, 'r') as f:
        sentence = []
        for line in f:
            line = line.strip()
            if line == '':
                if sentence:
                    data.append(sentence)
                    sentence = []
            else:
                parts = line.split()
                if len(parts) == 1 and filename.endswith("test"):
                    word = parts[0]
                    sentence.append(word)
                elif len(parts) == 2:
                    word, tag = parts
                    sentence.append((word, tag))
        if sentence:
            data.append(sentence)
    return data


def build_vocab(data):
    tag_vocab = {}
    word_vocab = {'<PAD>': 0, '<UNK>': 1}
    char_vocab = {'<PAD>': 0, '<UNK>': 1}
    prefix_vocab = {'<PAD>': 0, '<UNK>': 1}
    suffix_vocab = {'<PAD>': 0, '<UNK>': 1}

    for sentence in data:
        for word, tag in sentence:
            if word not in word_vocab:
                word_vocab[word] = len(word_vocab)
            if tag not in tag_vocab:
                tag_vocab[tag] = len(tag_vocab)
            for char in word:
                if char not in char_vocab:
                    char_vocab[char] = len(char_vocab)
            prefix = word[:3]
            if prefix not in prefix_vocab:
                prefix_vocab[prefix] = len(prefix_vocab)
            suffix = word[-3:]
            if suffix not in suffix_vocab:
                suffix_vocab[suffix] = len(suffix_vocab)

    return word_vocab, tag_vocab, char_vocab, prefix_vocab, suffix_vocab

class SentenceDataset(Dataset):
    def __init__(self, data, word_vocab, tag_vocab, prefix_vocab, suffix_vocab, max_char_len=20):
        self.data = data
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab
        self.prefix_vocab = prefix_vocab
        self.suffix_vocab = suffix_vocab
        self.max_char_len = max_char_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, tags = zip(*self.data[idx])
        word_ids = [self.word_vocab.get(word, self.word_vocab['<UNK>']) for word in words]
        tag_ids = [self.tag_vocab[tag] for tag in tags]

        prefix_ids = []
        suffix_ids = []
        for word in words:
            prefix = word[:3]
            suffix = word[-3:]
            prefix_ids.append(self.prefix_vocab.get(prefix, self.prefix_vocab['<UNK>']))
            suffix_ids.append(self.suffix_vocab.get(suffix, self.suffix_vocab['<UNK>']))

        return (
            torch.tensor(word_ids),
            torch.tensor(tag_ids),
            torch.tensor(prefix_ids),
            torch.tensor(suffix_ids),
            len(word_ids),
            words
        )


def collate_fn(batch):
    batch.sort(key=lambda x: x[5], reverse=True)
    inputs, tags, prefixes, suffixes, lengths, words = zip(*batch)
    max_len = max(lengths)
    padded_inputs = torch.zeros(len(inputs), max_len, dtype=torch.long)
    padded_tags = torch.zeros(len(inputs), max_len, dtype=torch.long)
    padded_prefixes = torch.zeros(len(inputs), max_len, dtype=torch.long)
    padded_suffixes = torch.zeros(len(inputs), max_len, dtype=torch.long)

    for i in range(len(inputs)):
        padded_inputs[i, :lengths[i]] = inputs[i]
        padded_tags[i, :lengths[i]] = tags[i]
        padded_prefixes[i, :lengths[i]] = prefixes[i]
        padded_suffixes[i, :lengths[i]] = suffixes[i]

    return padded_inputs, padded_tags, padded_prefixes, padded_suffixes, torch.tensor(lengths), words

def evaluate_model(model, dataloader, device, char_vocab=None, is_ner_task=False, tag_o_idx=0):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, tags, prefix_inputs, suffix_inputs, lengths, words in dataloader:
            inputs = inputs.to(device)
            tags = tags.to(device)
            prefix_inputs = prefix_inputs.to(device)
            suffix_inputs = suffix_inputs.to(device)
            outputs = model(inputs, lengths, prefix_inputs, suffix_inputs, words, char_vocab)
            predictions = outputs.argmax(dim=-1)
            if is_ner_task:
                mask = (tags != tag_o_idx)
                correct += ((predictions == tags) * mask).sum().item()
                total += mask.sum().item()
            else:
                correct += (predictions == tags).sum().item()
                total += tags.numel()


    return correct / total if total > 0 else 0.0


class CustomBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fw_cell1 = nn.LSTMCell(input_dim, hidden_dim)
        self.bw_cell1 = nn.LSTMCell(input_dim, hidden_dim)
        self.fw_cell2 = nn.LSTMCell(hidden_dim * 2, hidden_dim)
        self.bw_cell2 = nn.LSTMCell(hidden_dim * 2, hidden_dim)

    def forward(self, x, lengths):
        batch_size, seq_len, input_dim = x.size()

        h_fw1 = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c_fw1 = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        h_bw1 = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c_bw1 = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        fw_out1, bw_out1 = [], []

        for t in range(seq_len):
            h_fw1, c_fw1 = self.fw_cell1(x[:, t], (h_fw1, c_fw1))
            fw_out1.append(h_fw1)
            h_bw1, c_bw1 = self.bw_cell1(x[:, seq_len - t - 1], (h_bw1, c_bw1))
            bw_out1.insert(0, h_bw1)

        out1 = [torch.cat([f, b], dim=1) for f, b in zip(fw_out1, bw_out1)]
        out1 = torch.stack(out1, dim=1)

        h_fw2 = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c_fw2 = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        h_bw2 = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c_bw2 = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        fw_out2, bw_out2 = [], []
        for t in range(seq_len):
            h_fw2, c_fw2 = self.fw_cell2(out1[:, t], (h_fw2, c_fw2))
            fw_out2.append(h_fw2)
            h_bw2, c_bw2 = self.bw_cell2(out1[:, seq_len - t - 1], (h_bw2, c_bw2))
            bw_out2.insert(0, h_bw2)

        out2 = [torch.cat([f, b], dim=1) for f, b in zip(fw_out2, bw_out2)]
        out2 = torch.stack(out2, dim=1)

        return out2

class CustomCharBiLSTM(nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim, char_hidden_dim):
        super(CustomCharBiLSTM, self).__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim)
        self.char_hidden_dim = char_hidden_dim
        self.fw_lstm = nn.LSTMCell(char_emb_dim, char_hidden_dim)
        self.bw_lstm = nn.LSTMCell(char_emb_dim, char_hidden_dim)

    def forward(self, word_list, char_vocab):
        device = self.char_embedding.weight.device
        outputs = []

        for word in word_list:
            char_ids = [char_vocab.get(c, char_vocab['<UNK>']) for c in word]
            char_tensor = torch.tensor(char_ids, dtype=torch.long).to(device)
            char_embs = self.char_embedding(char_tensor)

            # Forward LSTM
            h_f, c_f = torch.zeros(self.char_hidden_dim, device=device), torch.zeros(self.char_hidden_dim, device=device)
            for t in range(char_embs.size(0)):
                h_f, c_f = self.fw_lstm(char_embs[t], (h_f, c_f))

            # Backward LSTM
            h_b, c_b = torch.zeros(self.char_hidden_dim, device=device), torch.zeros(self.char_hidden_dim, device=device)
            for t in reversed(range(char_embs.size(0))):
                h_b, c_b = self.bw_lstm(char_embs[t], (h_b, c_b))

            # Concatenate last forward and backward hidden states
            output = torch.cat((h_f, h_b), dim=0)  
            outputs.append(output)

        return torch.stack(outputs).to(device)


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tag_vocab_size, char_vocab_size, prefix_vocab_size, suffix_vocab_size,
                 embedding_dim, hidden_dim, repr_type='a'):
        super(BiLSTMTagger, self).__init__()
        self.repr_type = repr_type
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.prefix_embedding = nn.Embedding(prefix_vocab_size, embedding_dim)
        self.suffix_embedding = nn.Embedding(suffix_vocab_size, embedding_dim)

        self.char_embedding_dim = 20
        self.char_hidden_dim = 30

        if repr_type in ['b', 'd']:
            self.char_embedding = CustomCharBiLSTM(char_vocab_size, self.char_embedding_dim, self.char_hidden_dim // 2)

        if repr_type == 'a' or repr_type == 'c':
            input_dim = embedding_dim
        elif repr_type == 'b':
            input_dim = self.char_hidden_dim
        else: #if repr_type == 'd':
            input_dim = embedding_dim + self.char_hidden_dim
            self.repr_proj = nn.Linear(input_dim, embedding_dim)
            input_dim = embedding_dim

        self.custom_bilstm = CustomBiLSTM(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, tag_vocab_size)

    def forward(self, x, lengths, prefix_inputs=None, suffix_inputs=None, words_list=None, char_vocab=None):
        x = x.to(self.embedding.weight.device)

        if self.repr_type == 'a' or self.repr_type == 'd':
            embeds = self.embedding(x)
            if self.repr_type == 'd':
                word_embedded = embeds

        if self.repr_type == 'b' or self.repr_type == 'd':
            total_reprs = []
            for sentence in words_list:
                words_reprs = self.char_embedding(sentence, char_vocab)
                total_reprs.append(words_reprs)
            embeds = nn.utils.rnn.pad_sequence(total_reprs, batch_first=True)
            if self.repr_type == 'd':
                char_embedded = embeds
                embedded = torch.cat([word_embedded, char_embedded], dim=-1)
                embeds = self.repr_proj(embedded)

        if self.repr_type == 'c':
            word_embedded = self.embedding(x)
            prefix_embedded = self.prefix_embedding(prefix_inputs)
            suffix_embedded = self.suffix_embedding(suffix_inputs)
            embeds = word_embedded + prefix_embedded + suffix_embedded


        lstm_out = self.custom_bilstm(embeds, lengths)
        logits = self.fc(lstm_out)
        return logits



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('repr', choices=['a', 'b', 'c', 'd'])
    parser.add_argument('trainFile')
    parser.add_argument('modelFile')
    parser.add_argument('--devFile', required=True)
    parser.add_argument('--task', required=True)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = load_data(args.trainFile)
    dev_data = load_data(args.devFile)

    word_vocab, tag_vocab, char_vocab, prefix_vocab, suffix_vocab = build_vocab(train_data)
    o_label_idx = None
    if args.task == 'ner':
        is_ner_task = True 
        o_label_idx = tag_vocab['O']
    else:
        is_ner_task = False
    repr_type = args.repr
    batch_size = 25

    train_set = SentenceDataset(train_data, word_vocab, tag_vocab, prefix_vocab, suffix_vocab)
    dev_set = SentenceDataset(dev_data, word_vocab, tag_vocab, prefix_vocab, suffix_vocab)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model parameters
    embedding_dim = 100
    hidden_dim = 64

    # Model
    model = BiLSTMTagger(vocab_size=len(word_vocab),
                         tag_vocab_size=len(tag_vocab),
                         char_vocab_size=len(char_vocab),
                         prefix_vocab_size=len(prefix_vocab),
                         suffix_vocab_size=len(suffix_vocab),
                         embedding_dim=embedding_dim,
                         hidden_dim=hidden_dim,
                         repr_type=repr_type).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Training loop
    epochs = 5
    model.train()
    dev_acc = []
    for epoch in range(epochs):
        for i, (inputs, tags, prefixes, suffixes, lengths, words) in enumerate(train_loader):
            inputs = inputs.to(device)
            tags = tags.to(device)
            prefixes = prefixes.to(device)
            suffixes = suffixes.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, lengths, prefixes, suffixes, words, char_vocab)
            loss = criterion(outputs.view(-1, outputs.shape[-1]), tags.view(-1))
            loss.backward()
            optimizer.step()

            if i * batch_size % 500 == 0:
                print(f"Epoch {epoch+1}, Sentences {i * batch_size}, Loss: {loss.item():.4f}")
                dev_accuracy = evaluate_model(model, dev_loader, device=device, char_vocab=char_vocab, is_ner_task=is_ner_task, tag_o_idx=o_label_idx)
                print(f"[Dev set] Epoch {epoch+1}, Sentences {i *  batch_size}, Dev Accuracy: {dev_accuracy:.4f}")
                dev_acc.append(dev_accuracy)


    torch.save({
        'model_state_dict': model.state_dict(),
        'word_vocab': word_vocab,
        'tag_vocab': tag_vocab,
        'char_vocab': char_vocab,
        'prefix_vocab': prefix_vocab,
        'suffix_vocab': suffix_vocab
    }, args.modelFile)


if __name__ == '__main__':
    main()


