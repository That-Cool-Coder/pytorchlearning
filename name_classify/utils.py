import torch

MAX_STR_LENGTH = 30

def str_to_list(string: str):
    if len(string) > MAX_STR_LENGTH:
        string = string[:MAX_STR_LENGTH]
    ascii_values = list(bytes(string, 'utf8'))

    # pad to correct length with null char
    ascii_values += [0] * (MAX_STR_LENGTH - len(ascii_values))

    ascii_values = [float(c) for c in ascii_values]

    return ascii_values

def list_to_str(lst: list):
    return ''.join(chr(int(c)) for c in lst)