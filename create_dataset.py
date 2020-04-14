import tkinter as tk
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from PIL import Image, ImageTk
import time

np.random.seed(int(time.time())) # random seed

"""
    function for setting up the screen. It draws the drawing area with the grid.
"""

def clearScreen():
    # Draw drawing area rectangle
    canvas.create_rectangle( (x_min,y_min,x_max,y_max), fill= "white")

    # Create grid dots

    offset_x = (x_max - x_min) / 20
    offset_y = (y_max - y_min) / 20
    x_grid = x_min + offset_x
    y_grid = y_min + offset_y

    while(x_grid < x_max):
        while(y_grid < y_max):
            canvas.create_rectangle( (x_grid,y_grid)*2, fill= "#555", width=0)
            y_grid += offset_y
        x_grid += offset_x
        y_grid = y_min + offset_y


# Class that handles the drawn digit

class digitDrawing():
    def __init__(self):
        # Initialize two lists that will receive mouser pointer's x and y informations
        self.x = []
        self.y = []

    def clearPath(self):
        # Reinitialize the lists
        self.x = []
        self.y = []
    
    def addPoint(self,x,y):
        self.x.append(x)
        self.y.append(y)
    
    """
        Method for treating the drawn digit. It returns the counter of files in the selected digit directory
    """
    def treatDigit(self):
        
        global selectedDigit, count

        # Here we are going to tranform an image of indefinite size into an image of 28x28 pixel that the model can treat
        #We start by getting the max and min points of the drawing

        x_length = x_max-x_min
        y_length = y_max-y_min
        
        if x_length > y_length:
            y_length = x_length
        else:
            x_length = y_length
        
        offset = 2 # We leave a little space to the border

        # Then we make a transformation from (x_min,y_min,x_max,y_max) into (0,0,28,28)

        for i in range(len(self.x)):
            self.x[i]=(self.x[i]-x_min)/x_length*(28-2*offset) + offset
            self.y[i]=(self.y[i]-y_min)/y_length*(28-2*offset) + offset

        # We initiatialize the image as a 28x28 matrix of zeros
        image = np.zeros((28,28))

        # We finally calculate the new pixels
        for pos_x,pos_y in zip(self.x,self.y):
            image[math.floor(pos_x)][math.floor(pos_y)] = 1.0

            # Add a little noise to the surrounding pixels
            for xx,yy in zip([math.floor(pos_x)-1,math.floor(pos_x),math.floor(pos_x)+1],[math.floor(pos_y)-1,math.floor(pos_y),math.floor(pos_y)+1]):
                image[xx][yy] += np.random.random()/3 
        
        # Set the max pixel value to 1
        for x in range(28):
            for y in range(28):
                if image[int(x)][int(y)] > 1.0:
                    image[int(x)][int(y)] = 1.0
        
        image = np.transpose(image) # Rotate the image

        # Save the generated image as digit.jpg
        fig = plt.figure(figsize=(4.05, 4.05), dpi=100)
        plt.imshow(image, cmap='cividis')
        plt.savefig("digit.jpg")
        file = os.path.join("datasets",str(selectedDigit)) 
        plt.close(fig)
        try:
            f = open(file+"\\var.txt", 'r')
            count = int(f.readline())
        except:
            f = open(file+"\\var.txt", 'w')
            f.write("0")
            count = 0
        finally:
            f.close()
        
        count += 1

        np.save(file+"\\{}.npy".format(count), image)

        f = open(file+"\\var.txt", 'w')
        f.write(str(count))
        f.close()
        self.clearPath()
        return count

# -------------------------------------------------
# ------------------ BEGINNING --------------------
# -------------------------------------------------

digit = digitDrawing() # Creat a digitDrawing object

root = tk.Tk()
root.title("Handwritten Digit Dataset Creation Tool") 
root.resizable(False, False)
root.iconbitmap(os.path.join("Images","icon.ico"))
# Create canvas

box_width = 810
box_height = 540
canvas = tk.Canvas(root, width = box_width, height = box_height, bg = "#01224D")
canvas.pack()

# Some global variables initialisation

canvas.old_coords = None
text_id = None
last_button_evt = None
draw = False
selectedDigit = None
x_min, x_max = 0, box_width/2 
y_min, y_max = 0, box_width/2 


# -------------------------------------------------
# --------------- Event management ----------------
# -------------------------------------------------
"""
    function called when the left button is clicked.
    If the mouser pointer is inside the drawing area, it allows to draw and delete everything on the canvas
    (rectangles, texts and previous drawings). This way, the program runs smoother.
"""
def press(event):
    global draw, x, y
    if (x_min < x < x_max and y_min < y < y_max):
        draw = True # Enables drawing
        canvas.delete("all")  # Clear rectangles, texts and previous drawings
        clearScreen() # Draw original screen and keep images
    root.update()
    
"""
    function called when the left button is released.
    It disables the ability to draw and calls the function to treat the drawn digit, which returns the counter
    of images in the digit directory. 
    The function the opens the matplotlib's saved image (drawn digit ) and places it on the screen.
    It displays the digit save location.
"""
def release(event):
    global draw, x, y

    draw = False  # Disables drawing
    if (x_min < x < x_max and y_min < y < y_max):  
        num = digit.treatDigit() # Get counter and treat drawn digit
        # Open drawn digit image and place on screen
        load = Image.open("digit.jpg")
        render = ImageTk.PhotoImage(load)
        img = tk.Label(image=render)
        img.image = render
        img.place(x=405, y=0)

        # Print recorded digit and location
        canvas.create_text(350, 490, text='Saving {} to .\\datasets\\{}\\{}.npy...'.format(selectedDigit, selectedDigit, num), fill="white", font=("Purisa", 22), width=700)

    root.update()

"""
    function called when the mouse moves.
    It gets the mouser pointer position on the GUI's screen and, if drawing is enables, it draws in the
    drawing area and add point point position to digit instance of digitDrawing. 
"""

def motion(event):
    global draw, canvas, x, y
    # Get pointer position
    x = root.winfo_pointerx() - root.winfo_rootx()
    y = root.winfo_pointery() - root.winfo_rooty()

    if draw and (x_min < x < x_max and y_min < y < y_max):
        digit.addPoint(x,y)
        x1, y1 = canvas.old_coords
        canvas.create_line((x, y, x1, y1), fill = "black", width=1)
    
    canvas.old_coords = x, y
    

#Bind keys to their functions

root.bind('<Motion>', motion)
root.bind("<Button-1>", press)
root.bind("<ButtonRelease-1>", release)

# -------------------------------------------------
# --------------- Button creation -----------------
# -------------------------------------------------

pixelVirtual = tk.PhotoImage(width=1, height=1)

class but():
    id = -1 # Class variable for setting the id of the button
    def __init__(self):
        but.id += 1
        self.id = but.id # Set button id with the class' counter but.id
        """
            Create the button. It is linked to root and has as text its id. The image argument is for
            higher size configuration precision when handling with height and width parameters.
            When clicked, the button calls the callback() method 
        """
        self.btn = tk.Button(root, text=self.id, image=pixelVirtual,height = 25, width = 35.5, 
                     bg="white", border=1, compound="c", command=self.callback)

        self.btn.place(x=40.45*self.id,y=405) # Placement of the button

    """
        Function for making all buttons turn white and the clicked button turn green
    """
    def callback(self):
        for i in range(but.id + 1):
            bt[i].btn.configure(bg="white")
        self.btn.configure(bg="#3f8c00")
        global selectedDigit
        selectedDigit = self.id


bt=[i for i in range(10)] # Create 10 buttons for digits 0 - 9
for i in range(10):
    bt[i] = but()

count = 0

clearScreen() # Setup the screen

root.mainloop()