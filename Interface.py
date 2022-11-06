import tkinter as tk
from tkinter import StringVar, messagebox, ttk
# from dataeng import DataEng
import pickle


MODEL_NAME = "lr_model.sav"
WIDTH = 500
HEIGHT = 500
#The main page when open the app
class Main_Page(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Main Page")

        wid_screen = self.winfo_screenwidth()
        height_screen = self.winfo_screenheight()
        canva = tk.Canvas(self,width=500, height=150)
        canva.grid(columnspan=2, rowspan=4)
        x = (wid_screen/2) - (WIDTH/2)
        y = (height_screen/2) - (HEIGHT/2)
        self.geometry('%dx%d+%d+%d' % (WIDTH, HEIGHT, x, y))

        tk.Label(text="Welcome to \n Article Grade Marking System", font=("Calibri", 20)).grid()
        self.input = StringVar()
        self.output = StringVar()
        tk.Entry(self, textvariable = input, font=('calibre',10,'normal')).grid()
        tk.Button(self, text="Mark", width=12, height=1, command=self.mark).grid()
        tk.Label(textvariable=self.output, font=("Calibri", 20)).grid()

    def close(self):
        self.destroy()

    def mark(self):
        text = self.input.get()
        # data = DataEng(text).Engeering()
        # model = pickle.load(open(MODEL_NAME, "rb"))
        # results = model.predict(data)
        results = "1"
        self.output.set(results)


Main_Page().mainloop()