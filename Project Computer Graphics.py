# Final Project Computer Graphics 204382
# Code By
# Nuttakorn Masphan 600510545
# Juthaporn Simmalee 600510537


from tkinter import *
from tkinter import filedialog
import cv2

def select_image():
	path = filedialog.askopenfilename()
	show_image(path)
	
def show_image(path):
        img = cv2.imread(path)
        cv2.imshow('Show image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
def main():
        root = Tk()
        root.geometry('150x50') 
        btn = Button(root, text="Select an image", command=select_image)
        btn.pack(side = TOP, pady = 10)
        # kick off the GUI
        root.mainloop()
if __name__ == "__main__":
    main()


