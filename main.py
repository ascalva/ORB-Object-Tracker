import time
import cv2
import random
from vWriter import VideoWriterWrapper
import numpy as np

# Define global variables
vidWriter = None


def main():
    global vidWriter
    cap = cv2.VideoCapture(0)#"IMG_3385.MOV")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    vidWriter = VideoWriterWrapper(frame_width, frame_height)
    time.sleep(0.3)
    prevImg = None

    # always comparing two frames, so do something else seeing first frame
    # prompt for ROI, then set to cframe
    ret,img = cap.read()
    bbox    = cv2.selectROI(img, False)
    prevImg = img
    
    x = int(bbox[0])
    y = int(bbox[1])
    w = int(bbox[2])
    h = int(bbox[3])
    x_i = x + w/2
    y_i = y + h/2
    cframe = (x_i, y_i, w, h)
    pframe = cframe
    while(True):
        # Capture frame-by-frame
        ret, img = cap.read()

        # constraint to end when reading from a video file instead of a device
        # video stream
        if img is None:
            return

        else:
            pframe, cframe = processLiveFeed(prevImg, img, pframe, cframe)
            prevImg = img

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    vidWriter.cleanup()
    # cv2.destryoAllWindows()
