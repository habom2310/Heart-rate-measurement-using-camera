import cv2
from face_detection import Face_detection
from face_alignment import Face_alignment
import utils
import time

video = cv2.VideoCapture("hci-tagging-database_download_2019-09-16_04_49_25/Sessions/784/P7-Rec1-2009.07.22.16.46.48_C1 trigger _C_Section_4.avi")

fd = Face_detection()
fa = Face_alignment()

count = 0
with open("hrv.data","w+") as f:
    t0 = 0
    while video.isOpened():
        ret, frame = video.read()
        if count%500 == 0:
            print(count)
            if t0 == 0:
                t0 = time.time()
            print(time.time() - t0)
            t0 = time.time()

        count += 1

        if ret == True:
            rects = fd.detect(frame)

            if len(rects) < 1:
                continue
                
            rect = rects[0]
            rect = utils.read_dlib_rect(rect)
            face = frame[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]

            cv2.rectangle(frame,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]), (255,0,0), 2)

            shape = fa.facial_landmark(face)
            aligned_face, aligned_shape = fa.align(face, shape)

            roi1, roi2 = utils.select_ROI(aligned_face, aligned_shape)

            mean_val = utils.extract_mean_val([roi1, roi2])
            cv2.imshow("video", frame)
            cv2.waitKey(1)
            f.write(str(mean_val) + "\n")
        else: 
            break
     

cv2.destroyAllWindows()