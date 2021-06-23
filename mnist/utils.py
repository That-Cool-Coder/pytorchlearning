import torch
import matplotlib.pyplot as plt

MAX_STR_LENGTH = 30
CHARSET = list('abcdefghijklmnopqrstuvwxyz')
NULL_CHAR = [0.] * len(CHARSET)

def nearest(number, *args):
    # Find which value in args is closest (up or down) to number
    closest_delta = None
    closest_target = None
    for value in args:
        delta = abs(value - number)
        if closest_delta is None or delta < closest_delta:
            closest_delta = delta
            closest_target = value
    return closest_target

def pyplot_draw_image(image):
    plt.imshow(image, cmap='gray', interpolation='none')
    plt.show()