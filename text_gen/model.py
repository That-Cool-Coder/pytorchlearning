import typing
import string

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

MODEL_FILE_NAME = 'text_gen.pth'
CHARSET = list(string.ascii_lowercase + ' .')
LOOK_BACK = 20

class TextGen(nn.Module):
    def __init__(self, charset: typing.List[str], look_back = 4):
        super(TextGen, self).__init__()
        self.charset = charset
        self.look_back = look_back

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(len(self.charset) * look_back, 100)
        self.linear2 = nn.Linear(100, 40)
        self.linear3 = nn.Linear(40, len(self.charset))

    def forward(self, input):
        x = self.linear(input)
        val = self.relu(x)
        x = self.linear2(val)
        val = self.sigmoid(x)
        yh = self.linear3(val)
        return yh
    
    def char_to_tensor(self, char: str):
        value = [0] * len(self.charset)
        value[self.charset.index(char)] = 1
        return value
    
    def text_to_tensor(self, prompt: str):
        prompt = prompt.lower()
        if len(prompt) > self.look_back:
            prompt = prompt[-self.look_back:]
        
        prompt_modified = []
        for char in prompt:
            prompt_modified += self.char_to_tensor(char)
        return prompt_modified
    
    def tensor_to_char(self, data: list):
        data = [max(n, 0) for n in data]
        divisor = sum(data)
        data = [x / divisor for x in data]
        return self.charset[np.random.choice(len(data), p=data)]
        return self.charset[data.index(max(data))]
    
    def next_char(self, prompt: str):
        network_input = torch.Tensor([self.text_to_tensor(prompt)])
        output = self(network_input).detach().numpy().tolist()
        return self.tensor_to_char(output[0])