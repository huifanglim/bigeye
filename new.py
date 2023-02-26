import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageTk

root = tk.Tk()

# Function for opening the
# file explorer window
def browseFile():
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("Text files",
                                                        "*.jpg*"),
                                                       ("all files",
                                                        "*.*")))

    # Change label contents
    LabelFile.configure(text="File Opened: "+filename)

# title
root.title('Choose image')

# window size
root.geometry("500x500")

# background colour
root.config(background="white")

# create label
LabelFile = tk.Label(root,
                  text = "File Explorer",
                  width = 75, height = 4,
                  fg = "blue")

ButtonExplorer = tk.Button(root,
                          text = "Browse and select image",
                          command= browseFile)

ButtonExit = tk.Button(root,
                       text= "Exit",
                       command = exit)

LabelFile.grid(column=1,row=1)

ButtonExplorer.grid(column=1,row=2)

ButtonExit.grid(column=1,row=3)

tk.mainloop()