print('Importing libraries..')

import tkinter as tk

import torch
import torchvision

from model import *
import utils

MODEL_FILE_NAME = 'model.pth'

print('Initialising model...')
model = MNIST()

print('Loading training...')
model.load_state_dict(torch.load(MODEL_FILE_NAME))

class App(tk.Tk):
    HEADING_FONT = ('helvetica', 20)
    MAIN_FONT = ('helvetica', 16)

    INITIAL_GEOMETRY = '600x500'

    IMAGE_SIZE = 28
    IMAGE_SCALE = 10
    CANVAS_SIZE = IMAGE_SIZE * IMAGE_SCALE

    BLACK = -0.424
    WHITE = 2.8

    UPDATE_RATE = int(1000 / 120)

    def __init__(self, model):
        super().__init__()
        self.title('Digit Recognition')
        self.geometry(self.INITIAL_GEOMETRY)

        self.model = model

        self.mouse_down = False
        self.mouse_x = 0
        self.mouse_y = 0

        self.drawn_image = torch.Tensor(1, 1, self.IMAGE_SIZE, self.IMAGE_SIZE)

        tk.Label(self, text='Digit Recognition',
            font=self.HEADING_FONT).pack()
        self.add_line_break()
        tk.Label(self, text='Powered by PyTorch', font=self.MAIN_FONT).pack()
        self.add_line_break()
        tk.Button(self, text='Predict', font=self.MAIN_FONT,
            command=self.predict_number).pack()
        tk.Button(self, text='Clear canvas', font=self.MAIN_FONT,
            command=self.clear_canvas).pack()

        self.output_label = tk.Label(self, text='', font=self.MAIN_FONT)
        self.output_label.pack()

        self.canvas = tk.Canvas(self, width=self.CANVAS_SIZE,
            height=self.CANVAS_SIZE)
        self.canvas.bind('<Button-1>', self.register_mouse_down)
        self.canvas.bind('<ButtonRelease-1>', self.register_mouse_up)
        self.canvas.bind('<Motion>', self.register_mouse_move)
        self.canvas.pack()

        self.create_canvas_rects()
        self.clear_canvas()
        self.update_canvas()

        self.updateloop()
        self.mainloop()

    def add_line_break(self, widget=None):
        # if widget is undefined, use self (can't set self as arg value)
        if widget is None:
            tk.Label(self, font=self.MAIN_FONT).pack()
        else:
            tk.Label(widget, font=self.MAIN_FONT).pack()
    
    def clear_canvas(self):
        self.drawn_image.fill_(self.BLACK)
        self.update_canvas()
    
    def register_mouse_down(self, event):
        self.mouse_down = True
    
    def register_mouse_up(self, event):
        self.mouse_down = False
    
    def register_mouse_move(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y
    
    def updateloop(self):
        if self.mouse_down:
            row = int(self.mouse_x / self.IMAGE_SCALE)
            col = int(self.mouse_y / self.IMAGE_SCALE)
            self.drawn_image[0][0][col][row] = self.WHITE
        self.update_canvas()

        self.after(self.UPDATE_RATE, self.updateloop)
    
    def predict_number(self):
        with torch.no_grad():
            output = model(self.drawn_image)
        result = output.data.max(1, keepdim=True)[1][0].item()
        self.output_label['text'] = f'Prediction: {result}'

    def create_canvas_rects(self):
        self.canvas_rects = []
        for row_num in range(self.IMAGE_SIZE):
            row = []
            for col_num in range(self.IMAGE_SIZE):
                pos_x = col_num * self.IMAGE_SCALE
                pos_y = row_num * self.IMAGE_SCALE
                rect = self.canvas.create_rectangle(pos_x,
                    pos_y, pos_x + self.IMAGE_SCALE, pos_y + self.IMAGE_SCALE,
                    outline='')
                row.append(rect)
            self.canvas_rects.append(row)

    def update_canvas(self):
        image_data = self.drawn_image[0][0]
        for row_num in range(len(image_data)):
            data_row = image_data[row_num]
            rect_row = self.canvas_rects[row_num]
            for col_num in range(len(rect_row)):
                brightness = int(self.map_number(data_row[col_num],\
                    self.BLACK, self.WHITE, 0, 100))
                fill_color = f'grey{brightness}'
                self.canvas.itemconfig(rect_row[col_num],
                    fill=fill_color)
                
        self.update()

    def map_number(self, num, old_min, old_max, new_min, new_max):
        old_range = old_max - old_min
        new_range = new_max - new_min

        scaled = (num - old_min) / old_range
        return scaled * new_range + new_min

App(model)