import os
import tkinter as tk
from tarfile import PAX_FIELDS
from tkinter import Text, filedialog

from PIL import Image, ImageTk

root = tk.Tk()
apps = []

if os.path.isfile('save.txt'): # add on if already exists
    with open('save.txt','r') as f:
        tempApps = f.read()
        apps = tempApps.split(',')
        apps = [x for x in tempApps if x.strip()] #strip empty spaces

# open file explorer
def addApp():
    for widget in frame.winfo_children(): #access everything atttached to frame
        widget.destroy() #remove everything

    filename = filedialog.askopenfilename(initialdir="/",title = "Select File", filetypes=(("images", "*.jpg"),("all files",".")))
    apps.append(filename)
    for app in apps:
        label = tk.Label(frame, text=app, bg = "gray")
        label.pack()
        image1 = Image.open(filename)
        image1.thumbnail((300,300),Image.ANTIALIAS)
        test = ImageTk.PhotoImage(image1)
        label1 = tk.Label(image=test)
        label1.image = test
        label1.place(x=350,y=350)

def runApps():
    for app in apps:
        os.startfile(app)

# make GUI bigger, change colour
canvas = tk.Canvas(root, height = 700, width = 700, bg = "#FFFFFF")
canvas.pack()

frame = tk.Frame(root, bg = "white")
frame.place(relwidth = 0.8,relheight=0.8,relx=0.1,rely=0.1)

# add button to open file
openfile = tk.Button(root, text="Open File", padx=10,pady=5, fg ="white", bg = "#808080", command = addApp)
openfile.pack()

runApps = tk.Button(root, text="Run Apps", padx=10,pady=5, fg ="white", bg = "#33030F",command= runApps)
runApps.pack()

for app in apps:
    label = tk.Label(frame,text=app)
    label.pack

root.mainloop()

# write apps to file when closing
with open('save.text','w') as f:
    for app in apps:
        f.write(app +',')
