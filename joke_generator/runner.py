print('Importing libraries...')
import torch
import numpy as np
import argparse
from model import Model, MODEL_FILE_NAME
from dataset import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--sequence-length', type=int, default=4)
args = parser.parse_args()
dataset = Dataset(args)
model = Model(dataset)
model.load_state_dict(torch.load(MODEL_FILE_NAME))

def predict(dataset, model, text, next_words=100):
    model.eval()

    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        try:
            x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        except:
            continue
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return words

while True:
    text = input('Enter text to prompt model: ')
    num_words_to_predict = input('Enter amount of words to predict: ')
    try:
        num_words_to_predict = int(num_words_to_predict)
    except:
        print('Invalid input')
        break
    output = ' '.join(predict(dataset, model, text=text, next_words=num_words_to_predict))
    print(output)