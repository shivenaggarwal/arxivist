import io
import os
import unicodedata
import string
import glob
import torch
import random

# alphabet small + capital letters + " .,''"
# we only allow the letters mentioned above
ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)

# turn unicode string to plain ascii
def unicode_to_ascii(s):
  return ''.join(
    c for c in unicodedata.normalize('NFD', s)
    if unicodedata.category(c) != 'Mn'
    and c in ALL_LETTERS
  )

def load_data():
  # building the category lines dict, a list of names per language
  category_lines = {}
  all_categories = []

  def find_files(path):
    return glob.glob(path)

  def read_lines(filename): # this reads the file and splits it into lines
    lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

  for filename in find_files('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)

    lines = read_lines(filename)
    category_lines[category] = lines

  return category_lines, all_categories

def letter_to_index(letter): # find letter index from all letter eg 'a' = 0
  return ALL_LETTERS.find(letter)

def letter_to_tensor(letter): # turn a letter a <1 x n_letters> tensor
  tensor = torch.zeros(1, N_LETTERS)
  tensor[0][letter_to_index(letter)] = 1 # this make it so a one hot vector is filled with 0s except for a 1 at the index of the current letter
  return tensor

# turning a line into <line_length x 1 x n_letters> Tensor
def line_to_tensor(line):
  tensor = torch.zeros(len(line), 1, N_LETTERS)
  for i, letter in enumerate(line):
    tensor[i][0][letter_to_index(letter)] = 1
  return tensor

def random_training_example(category_lines, all_categories):
  def random_choice(a):
    return a[random.randint(0, len(a) - 1)]


  category = random_choice(all_categories)
  line = random_choice(category_lines[category])
  category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
  line_tensor = line_to_tensor(line)
  return category, line, category_tensor, line_tensor

if __name__ == '__main__':
    print(ALL_LETTERS)
    print(unicode_to_ascii('Ślusàrski'))

    category_lines, all_categories = load_data()
    print(category_lines['Italian'][:5])

    print(letter_to_tensor('J')) # [1, 57]
    print(line_to_tensor('Jones').size()) # [5, 1, 57] => 5 is because of the number of chars, 1 is what out model expects and 57 is because of all different chars
