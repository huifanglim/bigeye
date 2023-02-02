import tkinter as tk

k = 0

# Create main window
root = tk.Tk()
for x in range(2):
    root.columnconfigure(x,weight=1,minsize=150)
    root.rowconfigure(x,weight=1,minsize=150)

def sayHello():
    pass
    return

# Create click button
instance = tk.Button(root,text="hello",height=5,width=15,border=1,borderwidth=1,background='pink')
instance.bind('<Button>',sayHello())
instance.grid(row=0,column=0, padx= 3,pady=3)

instance = tk.Label(root,text = sayHello(), height=5,width=1,border=1,borderwidth=1)
instance.grid(row=1,column=1,padx=5,pady=5)

# Show main window
root.mainloop()
