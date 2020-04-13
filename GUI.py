import tkinter as tk
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image, ImageTk
import time


np.random.seed(int(time.time()))
# load NN model

model = tf.keras.models.load_model("C:\\Users\\User\\OneDrive\\Documents\\Códigos\\Tensorflow\\MNIST-Digit-Prediction\\Digit_Recognition_Models.h5")

def plot_value_array(prediction):
    prediction = prediction.reshape(10)
    fig = plt.figure(figsize=(4.05, 1.35), dpi=100)
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), prediction, color="#FDE837")
    plt.ylim([0, 1])
    predicted_label = np.argmax(prediction)

    thisplot[predicted_label].set_color('#01224D')
    plt.tight_layout()
    plt.savefig("C:\\Users\\User\\OneDrive\\Documents\\Códigos\\Tensorflow\\MNIST-Digit-Prediction\\bar.jpg")
 

def clearScreen():
    canvas.create_rectangle( (x_min,y_min,x_max,y_max), fill= "white")

    # create grid dots

    offset_x = (x_max - x_min) / 20
    offset_y = (y_max - y_min) / 20
    x_grid = x_min + offset_x
    y_grid = y_min + offset_y

    while(x_grid < x_max):
        while(y_grid < y_max):
            canvas.create_rectangle( (x_grid,y_grid)*2, fill= "gray")
            y_grid += offset_y
        x_grid += offset_x
        y_grid = y_min + offset_y


class digitDrawing():
    def __init__(self):
        self.x = []
        self.y = []
    
    def clearPath(self):
        self.x = []
        self.y = []
    
    def addPoint(self,x,y):
        self.x.append(x)
        self.y.append(y)
    
    def treatDigit(self):
        
        # Here we are going to tranform an image of indefinite size into an image of 28x28 pixel that the model can treat
        #We start by getting the max and min values of the drawing
        

        x_length = x_max-x_min
        y_length = y_max-y_min
        
        if x_length > y_length:
            y_length = x_length
        else:
            x_length = y_length
        
        offset = 2 # We leave a little spacing in the border

        # Then we make a transformation from (x_min,y_min,x_max,y_max) into (0,0,28,28)

        for i in range(len(self.x)):
            self.x[i]=(self.x[i]-x_min)/x_length*(28-2*offset) + offset
            self.y[i]=(self.y[i]-y_min)/y_length*(28-2*offset) + offset


        image = np.zeros((28,28))
        # We finally calculate the new pixels
        for pos_x,pos_y in zip(self.x,self.y):
            image[math.floor(pos_x)][math.floor(pos_y)] = 1.0
            for xx,yy in zip([math.floor(pos_x)-1,math.floor(pos_x),math.floor(pos_x)+1],[math.floor(pos_y)-1,math.floor(pos_y),math.floor(pos_y)+1]):
                image[xx][yy] += np.random.random()/3 # add a little noise

        for x in range(28):
            for y in range(28):
                if image[int(x)][int(y)] > 1.0:
                    image[int(x)][int(y)] = 1.0
        
        image = np.transpose(image)
        
        fig = plt.figure(figsize=(4.05, 4.05), dpi=100)
        plt.imshow(image, cmap='cividis')
        plt.savefig("C:\\Users\\User\\OneDrive\\Documents\\Códigos\\Tensorflow\\MNIST-Digit-Prediction\\digit.jpg")
        
        image = np.array(np.array(image).reshape((1,28,28,1)))
        prediction = model.predict(image)
        # print(prediction)
        plot_value_array(prediction)
        predicted_image = np.argmax(prediction)
        print("predicted digit = {}".format(predicted_image))
        self.clearPath()
        return prediction

digit = digitDrawing()

root = tk.Tk()

root.resizable(False, False)

box_width = 810 # 900
box_height = 540 # 600
canvas = tk.Canvas(root, width = box_width, height = box_height, bg = "#01224D")
canvas.pack()
canvas.old_coords = None
text_id = None
last_button_evt = None
write = False

x_min, x_max = 0, box_width/2 #0.1*box_width, 0.9*box_width/2
y_min, y_max = 0, box_width/2 #0.1*box_height, 0.9*box_width/2

clearScreen()

def press(event):
    canvas.delete("all")
    clearScreen()
    global write
    write = True
    root.update()
    

def release(event):
    global write, text_id
    write = False
    prediction = digit.treatDigit()
    
    load = Image.open("C:\\Users\\User\\OneDrive\\Documents\\Códigos\\Tensorflow\\MNIST-Digit-Prediction\\digit.jpg")
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render)
    img.image = render
    img.place(x=405, y=0)

    load = Image.open("C:\\Users\\User\\OneDrive\\Documents\\Códigos\\Tensorflow\\MNIST-Digit-Prediction\\bar.jpg")
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render)
    img.image = render
    img.place(x=405, y=405)
    #plt.show()
    canvas.create_text(120, 450, text='Prediction = {}'.format(np.argmax(prediction)), fill="white", font=("Purisa", 22))
    canvas.create_text(175, 500, text='Probability = {:.2f}%'.format(np.max(prediction)*100), fill="white", font=("Purisa", 22))

    root.update()

# Bind keypress event to handle_keypress()

def motion(event):
    global write, canvas
    x, y = event.x, event.y

    if write and (x_min < x < x_max and y_min < y < y_max):
        digit.addPoint(x,y)
        x1, y1 = canvas.old_coords
        #canvas.create_oval((x-5,y-5,x+5,y+5), fill = "black")
        canvas.create_line((x, y, x1, y1), fill = "black", width=1)
 
    canvas.old_coords = x, y
    

   # if x_min < x < x_max and y_min < y < y_max:
   #     canvas.create_rectangle( (x,y)*2, fill= "black")




root.bind('<Motion>', motion)
root.bind("<Button-1>", press)
root.bind("<ButtonRelease-1>", release)

root.mainloop()