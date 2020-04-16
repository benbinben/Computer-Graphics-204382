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


####################### increase gamma
# from __future__ import print_function
import numpy as np
# import argparse
import cv2
def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(image, table)

class myImage:
	def __init__(self, img_name):
		self.img=cv2.imread(img_name)
		self.__name=img_name
	def __str__(self):
		return self.__name
	#example 
	# x = MyImage('1.jpg')
	# str(x) => 1.jpg
	# x.img  => numpy array store img

x = myImage("Crop.png")
if x.img.mean()>127:
    save=0.0
    filename="Crop.png"
else:
    save=1.5
# print(x.img.shape)
# cv2.imshow(" ",x.img)
if x.img.shape[1]>450:
    scale_percent=int((450*100)/x.img.shape[1])# percent of original size if width>450 make to 450 & cal %
else:
    scale_percent=100
width = int(x.img.shape[1] * scale_percent / 100)
height = int(x.img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(x.img, dim, interpolation = cv2.INTER_AREA)
# cv2.imshow("Resized image", resized)
# cv2.waitKey(0)

original = resized
for gamma in np.arange(0.0, 2.0, 0.5):
	if gamma == 1:
		continue
	gamma = gamma if gamma > 0 else 0.1
	adjusted = adjust_gamma(original, gamma=gamma)
	cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
	# cv2.imshow("Images", np.hstack([original, adjusted]))
	# filename=str(gamma)+"_"+str(x)
	if(gamma==save):
		cv2.imwrite(filename,adjusted)
	# cv2.waitKey(0)

################## Classify
from keras.models import load_model
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
# print(filename)
model = load_model('my_model.h5')
my_image = plt.imread(filename)# read file

my_image_resize = resize(my_image, (32,32,3))#resize
probabilities = model.predict(np.array([my_image_resize,] ) )

probabilities

number_to_class = ['airplan', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'hourse', 'ship', 'truck']
index = np.argsort(probabilities[0,:])
print('Most likly class:',number_to_class[index[9]], '--probability:', probabilities[0, index[9]])
print('Second likly class:',number_to_class[index[8]], '--probability:', probabilities[0, index[8]])
print('Third likly class:',number_to_class[index[7]], '--probability:', probabilities[0, index[7]])
print('Fourth likly class:',number_to_class[index[6]], '--probability:', probabilities[0, index[6]])
print('Fifth likly class:',number_to_class[index[5]], '--probability:', probabilities[0, index[5]])
#save the model
# model.save('my_model.h5')