import os
import torch
import random
from torch.utils.data import Dataset
from collections import defaultdict

def get_data_path(folder_name, task):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(base_dir, '..', task, folder_name))
    return data_dir

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

