import numpy as np
import cv2
import imutils
import dlib

class Face_detection:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        
    def detect(self, frame):
        if frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        
        return rects

import utils
if __name__ == "__main__":
    fd = Face_detection()
    img = cv2.imread("ed.jpg")

    rects = fd.detect(img)

    if len(rects) == 0:
        print("no face detected")
    
    else:
        for rect in rects:
            #rect = rects[0]
            rect = utils.read_dlib_rect(rect)

            face = img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]

            cv2.imshow("face", face)
            cv2.waitKey()
            cv2.destroyAllWindows()