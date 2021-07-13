import tkinter as tk
import lanes


root = tk.Tk()

width = 500
height = 200

s_width = root.winfo_screenwidth()
s_height = root.winfo_screenheight()

x = (s_width/2) - (width/2)
y = (s_height/2) - (height/2)

root.title("Lane Line Detection | Control Panel")
root.geometry("%dx%d+%d+%d" % (width, height, x, y))
frame = tk.Frame(root)
frame.pack()

button = tk.Button(frame, text="Run Algorithm", command=lanes.main).pack()

root.mainloop()