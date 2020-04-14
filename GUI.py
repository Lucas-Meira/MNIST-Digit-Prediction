import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from PIL import Image, ImageTk
import time

np.random.seed(int(time.time())) # random seed

"""
    function for loading the model from a dialog box and changing window's title
"""

def loadModel():
    global model, root
    path =  tk.filedialog.askopenfilename(initialdir = os.path.join(os.getcwd(), "Models"), title = "Select a model",filetypes = [("h5 Model Files","*.h5")])
    try:
        model = tf.keras.models.load_model(path) 
        print("Model loaded successfully!")
        model_name = os.path.relpath(path, os.getcwd())
    except:
        print("Failed to load model. Loading hybrid model")
        model = tf.keras.models.load_model(os.path.join(os.getcwd(), "Models", "Dataset_and_MNIST_Digit_Recognition_Models.h5"))
        model_name = "Dataset_and_MNIST_Digit_Recognition_Models.h5"
    
    root.title("MNIST Digit Prediction GUI - " + model_name) 
    #return model, model_name

"""
    function for creating the bar graph plot with the probabilities.
    If a button has been clicked, it uses this information as it is the 
    digit the user intends to draw and plots a red bar if the prediction is incorrect
    and a blue bar if the prediction is correct. All other bars are yellow.
    If no button was clicked, then it plots only the blue bar for the predicted digit
    and yellow bars for all the others.
    It saves the plot to bar.jpg
"""

def plot_value_array(prediction, true_label):
    prediction = prediction.reshape(10)
    fig = plt.figure(figsize=(4.05, 1.35), dpi=100)
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), prediction, color="#FDE837")
    plt.ylim([0, 1])
    predicted_label = np.argmax(prediction)

    if(true_label != None):
        thisplot[predicted_label].set_color('#b00c00')
        thisplot[true_label].set_color('#01224D')
    else:
        thisplot[predicted_label].set_color('#01224D')

    plt.tight_layout()
    plt.savefig("bar.jpg")
    plt.close(fig)


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
        # Initialize two lists that will receive mouser pointer's x and y informations and correct and counter to calculate accuracy
        self.x = []
        self.y = []
        self.correct = 0
        self.counter = 0

    def clearPath(self):
        # Reinitialize the lists
        self.x = []
        self.y = []
    
    def addPoint(self,x,y):
        self.x.append(x)
        self.y.append(y)
    
    """
        Method for treating the drawn digit. It returns the prediction array.
    """
    def treatDigit(self):
        
        global selectedDigit

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
                image[xx][yy] += np.random.random()/10 

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
        plt.close(fig)

        # Reshape the image to be utilized by the NN model
        image = np.array(np.array(image).reshape((1,28,28,1)))
        prediction = model.predict(image) # Make the prediction
        # Create bar plot
        plot_value_array(prediction, selectedDigit)
        predicted_digit = np.argmax(prediction) # get the predicted digit
        
        if(selectedDigit != None):
            self.counter += 1
            self.correct += 1 if predicted_digit == selectedDigit else 0
            self.accuracy = self.correct/self.counter
            print("Predicted Digit = {}, Accuracy = {:.2f}%".format(predicted_digit, self.accuracy*100))
        else:
            print("Predicted Digit = {}".format(predicted_digit))
        self.clearPath()
        return prediction

# -------------------------------------------------
# ------------------ BEGINNING --------------------
# -------------------------------------------------

digit = digitDrawing() # Creat a digitDrawing object

root = tk.Tk()
root.title("MNIST Digit Prediction GUI") 
root.resizable(False, False)
# Dialog box to select model

loadModel() # Load model and change window' title

# Create menu bar
menubar = tk.Menu(root)
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Load Model", command = loadModel)
menubar.add_cascade(label="File", menu=filemenu)

helpmenu = tk.Menu(menubar, tearoff=0)
helpmenu.add_command(label="Guide")
menubar.add_cascade(label="Help", menu=helpmenu)
root.config(menu=menubar)

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
    It disables the ability to draw and calls the function to treat the drawn digit, which returns the prediction
    array. 
    The function the opens the matplotlib's saved images (drawn digit and bar plot) and places them on the screen.
    It also displays the predicted digit and the prediction probability.
"""
def release(event):
    global draw, x, y

    draw = False  # Disables drawing

    if (x_min < x < x_max and y_min < y < y_max):
        prediction = digit.treatDigit() # Treat drawn digit
        # Open drawn digit image and place on screen
        load = Image.open("digit.jpg")
        render = ImageTk.PhotoImage(load)
        img = tk.Label(image=render)
        img.image = render
        img.place(x=405, y=0)

        # Open drawn bar plot image and place on screen
        load = Image.open("C:\\Users\\User\\OneDrive\\Documents\\Códigos\\Tensorflow\\MNIST-Digit-Prediction\\bar.jpg")
        render = ImageTk.PhotoImage(load)
        img = tk.Label(image=render)
        img.image = render
        img.place(x=405, y=405)
        
        # Print predicted digit and prediction probability
        canvas.create_text(125, 465, text='Prediction = {}'.format(np.argmax(prediction)), fill="white", font=("Purisa", 22))
        canvas.create_text(180, 515, text='Probability = {:.2f}%'.format(np.max(prediction)*100), fill="white", font=("Purisa", 22))

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

clearScreen() # Setup the screen

root.mainloop()