# RNN-Acceptors-and-BiRNN-Transducers


---

## Overview
This repository contains our implementation for the three parts of the assignment:

1. **RNN Acceptor (LSTM-based binary classifier)**  
2. **BiLSTM Tagger for POS and NER**  
---

## RNN Acceptor

This component implements an LSTM-based **acceptor** that classifies whether a sequence belongs to a target formal language.

### Files
- `experiment.py` – Trains and evaluates the LSTM acceptor  
- `gen_examples.py` – Generates positive/negative examples  
- `pos_examples.txt`, `neg_examples.txt` – Training data  

### Run
```bash
python gen_examples.py     # generates datasets
python experiment.py       # trains and evaluates model
```

---

## BiLSTM Transducer (POS & NER Tagger)

Implements a **2-layer BiLSTM tagger** supporting multiple representations:
- Word embeddings  
- Character LSTM embeddings  
- Subword features (prefix/suffix)  
- Combined embeddings  

### Files
- `bilstmTrain.py` – Training script  
- `bilstmPredict.py` – Prediction script  
- `test4.pos`, `test4.ner` – Predictions on blind test sets  

### Train
```bash
python bilstmTrain.py repr trainFile modelFile --devFile devFile --task pos|ner
```

### Predict
```bash
python bilstmPredict.py repr modelFile inputFile --predFile outputFile
```

---
