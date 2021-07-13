import tkinter as tk
from tkinter import filedialog
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


select_file = tk.Label(frame, text="Select video file").grid(row=0, column=0)
filename = tk.StringVar()
entry_select_file = tk.Entry(frame, textvariable=filename).grid(row=0, column=1)

def browse_file():
    file_path = filedialog.askopenfilename()
    filename.set(file_path)


browse = tk.Button(frame, text="Browse", command=browse_file).grid(row=0, column=2)

roi = tk.IntVar()


button = tk.Button(frame, text="Run Algorithm", command=lanes.main).grid(row=1, column=0)
tk.Checkbutton(frame, text="Region of Interest", variable=roi).grid(row=1, column=1)
button = tk.Button(frame, text="SHOW", command=roi.get).grid(row=1, column=2)

control_keys = tk.Label(frame, text=
                        """Control Keys:
Press "q" to exit the algorithm process.
Press "s" to stop frame during algorithm process""",
                        justify=tk.LEFT,
                        bg="#dbdbdb").grid(row=2, column=0, pady=100)

root.mainloop()