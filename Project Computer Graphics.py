# Final Project Computer Graphics 204382
# Code By
# Nuttakorn Masphan 600510545
# Juthaporn Simmalee 600510537

from tkinter import *
from tkinter import filedialog
import numpy as np
import cv2

root = Tk()
cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
path = ""
             
def open_file():
        global root
        root.geometry('150x80') 
        # Cap image from webcam
        CapVideo = Button(root, text="Capture webcam", command=Cap_webcam)
        CapVideo.pack(side = BOTTOM , pady=5)
        # Select image
        select = Button(root, text="Select an image", command=select_image)
        select.pack(side = BOTTOM , pady=10)
        # kick off the GUI
        root.mainloop()
               
def select_image():
        global path
        print("You choose select image from your computer")
        path = filedialog.askopenfilename()
        root.destroy()
        
def Cap_webcam():
        global path
        print("You choose capture image from your webcam")
        cap = cv2.VideoCapture(0)
        while(True):
            ret, frame = cap.read()
            cv2.imshow('frame',frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('c'):
                    cv2.imwrite('Capture.png', frame)
                    break

        cap.release()
        cv2.destroyAllWindows()
        path = 'Capture.png'
        root.destroy()

def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping
 
    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
 
    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
 
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished
 
        refPoint = [(x_start, y_start), (x_end, y_end)]
 
        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imwrite('Crop.png', roi)
            cv2.imshow("Cropped", roi)
            
# main
open_file()

# crop function
image = cv2.imread(path)
oriImage = image.copy()

cv2.namedWindow("Show image")
cv2.setMouseCallback("Show image", mouse_crop)
 
while True:
 
    i = image.copy()
 
    if not cropping:
        cv2.imshow("Show image", image)
        # press key 'q' if close
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    elif cropping:
        cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("Show image", i)
        # press key 'q' if close
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    cv2.waitKey(1)

# Close all cv2 Windows
cv2.destroyAllWindows()

# ทำต่อนี่

Crop = cv2.imread('Crop.png')
cv2.imshow("Crop show", Crop)

