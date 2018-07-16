import cv2
import numpy as np
import time

class Video(object):
    def __init__(self):
        self.dirname = ""
        self.cap = None
        t0 = 0
        
    def start(self):
        print("Start video")
        if self.dirname == "":
            print("invalid filename!")
            return
            
        self.cap = cv2.VideoCapture(self.dirname)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(fps)
        self.t0 = time.time()
        print(self.t0)
        self.valid = False
        try:
            resp = self.cap.read()
            self.shape = resp[1].shape
            self.valid = True
        except:
            self.shape = None
    
    
    def stop(self):
        if self.cap is not None:
            self.cap.release()
            print("Stop video")
    
    def get_frame(self):
        if self.valid:
            _,frame = self.cap.read()
            if frame is None:
                print("End of video")
                self.stop()
                print(time.time()-self.t0)
                return
            else:    
                frame = cv2.resize(frame,(640,480))
        else:
            frame = np.ones((480,640,3), dtype=np.uint8)
            col = (0,256,256)
            cv2.putText(frame, "(Error: Can not load the video)",
                       (65,220), cv2.FONT_HERSHEY_PLAIN, 2, col)
        return frame
        
        
        
        