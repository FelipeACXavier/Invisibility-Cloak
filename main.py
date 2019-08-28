#!/usr/bin/env python
import signal
import os
import cv2
import sys
import csv
import numpy as np

# Initialize camera object
cam = cv2.VideoCapture(0)

H_MIN = 0;
H_MAX = 255;
S_MIN = 0;
S_MAX = 255;
V_MIN = 0;
V_MAX = 255;

def on_trackbar(x):
    pass

# Create trackbars to set color to make invisible
def createTrackBar(cap):
    cv2.namedWindow('test')
    cv2.namedWindow('image')

    cv2.createTrackbar("H_MIN", "test", H_MIN, H_MAX, on_trackbar)
    cv2.createTrackbar("H_MAX", "test", H_MIN, H_MAX, on_trackbar)
    cv2.createTrackbar("S_MIN", "test", S_MIN, S_MAX, on_trackbar)
    cv2.createTrackbar("S_MAX", "test", S_MIN, S_MAX, on_trackbar)
    cv2.createTrackbar("V_MIN", "test", V_MIN, V_MAX, on_trackbar)
    cv2.createTrackbar("V_MAX", "test", V_MIN, V_MAX, on_trackbar)

    while True:
        ret, img = cap.read()

        # get current positions of four trackbars
        hmin = cv2.getTrackbarPos('H_MIN','test')
        hmax = cv2.getTrackbarPos('H_MAX','test')
        smin = cv2.getTrackbarPos('S_MIN','test')
        smax = cv2.getTrackbarPos('S_MAX','test')
        vmin = cv2.getTrackbarPos('V_MIN','test')
        vmax = cv2.getTrackbarPos('V_MAX','test')

        # converting from BGR to HSV color space
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        
        # Generating mask to detect red color
        lower_red = np.array([hmin,smin,vmin])
        upper_red = np.array([hmax,smax,vmax])
        mask1 = cv2.inRange(hsv,lower_red,upper_red)

        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8),iterations=2)
        mask1 = cv2.dilate(mask1,np.ones((3,3),np.uint8),iterations = 1)

        cv2.imshow('image',hsv)
        cv2.imshow('test', mask1)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(100) & 0xFF == ord(' '):
            saveFile(hmin, hmax, smin, smax, vmin, vmax)
            break

# Save HSV values to a file so they can be used later
def saveFile(a,b,c,d,e,f):
    with open("hsv.txt", "w") as fp:
        fp.write(str(a) + ',' + str(b) + ',' + str(c) + ',' + str(d) + ',' + str(e) + ',' + str(f))

# Read HSV values from saved file
def readFile(file):
    with open(str(file), "r") as fp:
        a = fp.readline();

    return list(a.split(','))

def destroyCamera(cap):
    cap.release()
    cv2.destroyAllWindows()

# Do the actual "invisibility" thingy
def colorCompare(cap, values):
    cv2.namedWindow("final")
    cv2.namedWindow("mask1")
    cv2.namedWindow("mask2")
    
    background = cv2.imread("Background/background_29.png")
    while True:
        # Capturing the live frame
        ret, img = cap.read()
        
        # Laterally invert the image / flip the image
        img  = np.flip(img, axis=1)
        
        # converting from BGR to HSV color space
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        
        # Generating mask to detect red color
        lower_red = np.array([int(values[0]),int(values[2]),int(values[4])])
        upper_red = np.array([int(values[1]),int(values[3]),int(values[5])])
        mask1 = cv2.inRange(hsv,lower_red,upper_red)
        
        # Range for upper range
        """ lower_red = np.array([170,120,70])
        upper_red = np.array([180,255,255])
        mask2 = cv2.inRange(hsv,lower_red,upper_red)
        
        # Generating the final mask to detect red color
        mask1 = mask1 + mask2 """

        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8),iterations=2)
        mask1 = cv2.dilate(mask1,np.ones((3,3),np.uint8),iterations = 1)
        
        #creating an inverted mask to segment out the cloth from the frame
        mask2 = cv2.bitwise_not(mask1)
        
        #Segmenting the cloth out of the frame using bitwise and with the inverted mask
        res1 = cv2.bitwise_and(img,img,mask=mask2) 

        # Generating the final output
        res1 = cv2.bitwise_and(background,background,mask=mask1)
        res2 = cv2.bitwise_and(img,img,mask=mask2)
        final_output = cv2.addWeighted(res1,1,res2,1,0)

        # cv2.imshow("original", img)
        #cv2.imshow("mask1", mask1)
        #cv2.imshow("mask2", mask2)
        cv2.imshow("original", final_output)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

# Save background for overlay
def takePicture(cap):
    cv2.namedWindow("Saving Background")

    # Wait for user command to start saving photos
    while True:
        ret, frame = cap.read()
        cv2.imshow("Saving Background", frame)

        if cv2.waitKey(100) & 0xFF == ord(' '):
            break
    
    background = 0
    img_counter = 0

    for i in range(30):
        # Read image
        ret, background = cap.read()

        # Laterally invert the image / flip the image.
        background = np.flip(background,axis=1)

        # Save to file
        img_name = "Background/background_{}.png".format(img_counter)
        cv2.imwrite(img_name, background)
        img_counter += 1
    cv2.destroyAllWindows()

def overlay():
    pass

def signal_handler(signum, frame):
    print("Interrupting due to signal " + str(signum))
    destroyCamera(cam)
    sys.exit(0)


if __name__ == "__main__":
    # Define signals if needed
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    inp = input("Take picture? 1 or 0 or 2--> ")
    if(inp == 1):
        print("Initializing background capture")
        takePicture(cam)
        colorCompare(cam, readFile("hsv.txt"))
    elif(inp == 0):
        print("Initializing color capture")
        colorCompare(cam, readFile("hsv.txt"))
    else:
        createTrackBar(cam)
        colorCompare(cam, readFile("hsv.txt"))
    # Clear and destroy windows
    destroyCamera(cam)
    print("Shutter cleared")
