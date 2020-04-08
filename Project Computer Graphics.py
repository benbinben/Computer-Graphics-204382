# Final Project Computer Graphics 204382
# Code By
# Nuttakorn Masphan 600510545
# Juthaporn Simmalee 600510537


from tkinter import *
from tkinter import filedialog
import cv2



def select_image():
        print("You choose select image from your computer")
        path = filedialog.askopenfilename()
        show_image(path)

def Cap_webcam():
        print("You choose capture image from your webcam")
        cap = cv2.VideoCapture(0)
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            # Display the resulting frame
            cv2.imshow('frame',frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('c'):
                    cv2.imwrite('Capture.png', frame)
                    break

        cap.release()
        cv2.destroyAllWindows()
        #img = cv2.imread('Capture.png')
        #cv2.imshow('Capture show',img)
        path = 'Capture.png'
        show_image(path)
        
def show_image(path):
        print(path)
        img = cv2.imread(path)
        cv2.imshow('Show image',img)
         
def main():
        root = Tk()
        root.geometry('150x80') 
        # Cap image from webcam
        CapVideo = Button(root, text="Capture webcam", command=Cap_webcam)
        CapVideo.pack(side = BOTTOM , pady=5)
        # Select image
        select = Button(root, text="Select an image", command=select_image)
        select.pack(side = BOTTOM , pady=10)
        # kick off the GUI
        root.mainloop()
if __name__ == "__main__":        
    main()


