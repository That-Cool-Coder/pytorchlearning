import torch

MAX_STR_LENGTH = 30
CHARSET = list('abcdefghijklmnopqrstuvwxyz')
NULL_CHAR = [0.] * len(CHARSET)

def str_to_list(string: str):
    if len(string) > MAX_STR_LENGTH:
        string = string[:MAX_STR_LENGTH]

    valid_charset_string = ''
    for char in string:
        if char in CHARSET:
            valid_charset_string += char
    
    output = []
    for char in valid_charset_string:
        this_char = NULL_CHAR.copy()
        this_char[CHARSET.index(char)] = 1.
        output += this_char

    # pad to correct length with null char
    output += NULL_CHAR * (MAX_STR_LENGTH - len(valid_charset_string))

    return output

def list_to_str(lst: list):
    return ''.join(chr(int(c)) for c in lst)