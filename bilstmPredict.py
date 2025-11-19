import argparse
import torch
import torch.nn as nn
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


class SentenceDatasetPredict(Dataset):
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
        words = self.data[idx]
        word_ids = [self.word_vocab.get(word, self.word_vocab['<UNK>']) for word in words]

        prefix_ids = []
        suffix_ids = []
        for word in words:
            prefix = word[:3]
            suffix = word[-3:]
            prefix_ids.append(self.prefix_vocab.get(prefix, self.prefix_vocab['<UNK>']))
            suffix_ids.append(self.suffix_vocab.get(suffix, self.suffix_vocab['<UNK>']))

        return (
            torch.tensor(word_ids),
            torch.tensor(prefix_ids),
            torch.tensor(suffix_ids),
            len(word_ids),
            words
        )

def collate_fn_predict(batch):
    batch.sort(key=lambda x: x[4], reverse=True)
    inputs, prefixes, suffixes, lengths, words = zip(*batch)
    max_len = max(lengths)
    padded_inputs = torch.zeros(len(inputs), max_len, dtype=torch.long)
    padded_prefixes = torch.zeros(len(inputs), max_len, dtype=torch.long)
    padded_suffixes = torch.zeros(len(inputs), max_len, dtype=torch.long)

    for i in range(len(inputs)):
        padded_inputs[i, :lengths[i]] = inputs[i]
        padded_prefixes[i, :lengths[i]] = prefixes[i]
        padded_suffixes[i, :lengths[i]] = suffixes[i]

    return padded_inputs, padded_prefixes, padded_suffixes, torch.tensor(lengths), words


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
    parser.add_argument('modelFile')
    parser.add_argument('inputFile')
    parser.add_argument('--predFile', required=True)

    args = parser.parse_args()

    model_checkpoint = torch.load(args.modelFile, map_location=torch.device('cpu'))

    test_data = load_data(args.inputFile)

    word_vocab, tag_vocab, char_vocab, prefix_vocab, suffix_vocab = (model_checkpoint['word_vocab'],
                                                                     model_checkpoint['tag_vocab'],
                                                                     model_checkpoint['char_vocab'],
                                                                     model_checkpoint['prefix_vocab'],
                                                                     model_checkpoint['suffix_vocab'])

    index_to_tag = {v: k for k, v in tag_vocab.items()}

    repr_type = args.repr
    batch_size = 1

    test_set = SentenceDatasetPredict(test_data, word_vocab, tag_vocab, prefix_vocab, suffix_vocab)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_predict)

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
                         repr_type=repr_type)

    model.load_state_dict(model_checkpoint['model_state_dict'])
    model.eval()

    with open(args.predFile, "w", encoding="utf-8") as f:
        with torch.no_grad():
            for inputs, prefix_inputs, suffix_inputs, lengths, words in test_loader:
                outputs = model(inputs, lengths, prefix_inputs, suffix_inputs, words, char_vocab)
                predictions = torch.argmax(outputs, dim=-1)

                for i, sent_words in enumerate(words):
                    for j, word in enumerate(sent_words):
                        if word == "<PAD>":
                            continue
                        tag_idx = predictions[i][j].item()
                        tag = index_to_tag[tag_idx]
                        f.write(f"{word}    {tag}\n")
                    f.write("\n") 
    print(f"Test predictions written to {args.predFile}")

if __name__ == '__main__':
    main()


