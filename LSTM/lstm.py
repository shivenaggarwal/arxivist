import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import nltk
import time

with open("/content/nietzsche.txt", "r", encoding="utf-8") as f:
    document = f.read()

# tokenization imports
nltk.download('punkt')
nltk.download('punkt_tab')

tokens = word_tokenize(document.lower())

# building a vocab dict
vocab = {'<unk>': 0}

for token in Counter(tokens).keys(): # this will given us unique token along with their counts
  if token not in vocab:
    vocab[token] = len(vocab) # gives unique index to each token

# extract sentences from data
input_sentences = document.split('\n')

# this fn makes each token from sentence into a numerical repr, value taken from the vocab dict
def text_to_indices(sentence, vocab):
  numerical_sentence = []

  for token in sentence:
    if token in vocab:
      numerical_sentence.append(vocab[token])
    else:
      numerical_sentence.append(vocab['<unk>'])

  return numerical_sentence

input_numerical_sentences = [] # a 2d list with all of our sentences in numerical format
for sentence in input_sentences:
  input_numerical_sentences.append(text_to_indices(word_tokenize(sentence.lower()), vocab))

# this helps us make sequences for each of our sentence
# for eg lets say the first sentence from input_numerical_sentences is [1,2,3]
# then this fn will return [1,2], [1,2,3]
training_sequence = []
for sentence in input_numerical_sentences:
  for i in range(1, len(sentence)):
    training_sequence.append(sentence[:i+1])

print(training_sequence)

# now when we are going to send our data for training to lstm we are going to send them in batches
# because otherwise training so many training sequences will take a lot of time
# and when we send the batches, we want each of our training examples to have the same size
# so to make our examples the same size we will have to add padding
# for eg. [1,2], [1,2,3] => [0,1,2], [1,2,3]
len_list = []
for sequence in training_sequence:
  len_list.append(len(sequence))

padded_training_sequences = []
for sequence in training_sequence:
  padded_training_sequences.append([0] * (max(len_list) - len(sequence)) + sequence)

# converting our list to a tensor
padded_training_sequences = torch.tensor(padded_training_sequences, dtype=torch.long)

# splitting into training and test
X = padded_training_sequences[:, :-1]
y = padded_training_sequences[:, -1]

class CustomDataset(Dataset):
  def __init__(self, x, y):
    self.X = x
    self.y = y

  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

dataset = CustomDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# now we need to represent each of our word with a embedding vector
# this is going to be done by an embedding layer
# this will result in 32 sentences(batches), each sentence has a certain number of words which are padded,
# and now every word is represented by a ,say, a 100 dim embedding vector
# this will be sent to our lstm cell

class LSTMModel(nn.Module):
  def __init__(self, vocab_size):
    super(LSTMModel, self).__init__()

    self.embedding = nn.Embedding(vocab_size, 100) # 100 is our dimension
    self.lstm = nn.LSTM(100, 150, batch_first=True)
    self.fc = nn.Linear(150, vocab_size)

  def forward(self, x):
    embedded = self.embedding(x)
    intermediate_hidden_states, (final_hidden_state, final_cell_state) = self.lstm(embedded)
    output = self.fc(final_hidden_state.squeeze(0)) # this removes the extra dimension
    return output

model = LSTMModel(len(vocab))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 50
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(epochs):
  total_loss = 0

  for batch_x, batch_y in dataloader:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    optimizer.zero_grad()
    output = model(batch_x)
    loss = criterion(output, batch_y)
    loss.backward()
    optimizer.step()
    total_loss = total_loss + loss.item()

  print(f"Epoch: {epoch + 1}, Loss: {total_loss:.4f}")

# prediction
def pad_sequence(seq, max_len):
    return [0] * (max_len - len(seq)) + seq

def prediction(model, vocab, text):
    model.eval()
    #tokenize
    tokenized_text = word_tokenize(text.lower())
    # convert the text to numerical data like we did above
    numerical_text = text_to_indices(tokenized_text, vocab)
    max_len = X.shape[1]  # reuse training max length
    # padding to match the sizes
    padded_text = torch.tensor(pad_sequence(numerical_text, max_len), dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(padded_text)
        _, index = torch.max(output, dim=1)

    inv_vocab = {v: k for k, v in vocab.items()}
    return text + " " + inv_vocab[index.item()] # merging it with our input

prediction(model, vocab, "It is perhaps just dawning on five or six minds that natural")

num_tokens = 10
input_text = "To study physiology"

for i in range(num_tokens):
  output_text = prediction(model, vocab, input_text)
  print(output_text)
  input_text = output_text
  time.sleep(0.5)
