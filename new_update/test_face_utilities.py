'''
This script is to test the performance of all functions in the face_utilities class.
'''

from face_utilities import Face_utilities
import cv2
from imutils import face_utils
import numpy as np
import time


def flow_process(frame):
    display_frame = frame.copy()  
    rects = last_rects
    age = last_age
    gender = last_gender
    shape = last_shape
    
    # convert the frame to gray scale before performing face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # get all faces as rectangles every 3 frames
    if(i%3==0):
        rects = face_ut.face_detection(frame)
    
    #check if there is any face in the frame, if not, show the frame and move to the next frame
    if len(rects)<0:
        return frame, None
    
    # draw face rectangle, only grab one face in the frame
    (x, y, w, h) = face_utils.rect_to_bb(rects[0])
    cv2.rectangle(display_frame,(x,y),(x+w,y+h),(255,0,0),2)
    
    # crop the face from frame
    face = frame[y:y+h,x:x+w]
    
    if(i%6==0):
    # detect age and gender and put it into the frame every 6 frames
        age, gender = face_ut.age_gender_detection(face)
        
    overlay_text = "%s, %s" % (gender, age)
    cv2.putText(display_frame, overlay_text ,(x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
    
    if(i%3==0):
        # get 68 facial landmarks and draw it into the face every 3 frames
        shape = face_ut.get_landmarks(frame, "5")
    
    for (x, y) in shape: 
        cv2.circle(face, (x, y), 1, (0, 0, 255), -1)
        
    # get the mask of the face
    remapped_landmarks = face_ut.facial_landmarks_remap(shape)
    mask = np.zeros((face.shape[0], face.shape[1]))
    cv2.fillConvexPoly(mask, remapped_landmarks[0:27], 1) 
    
    aligned_face = face_ut.face_alignment(frame, shape)
    
    aligned_shape = face_ut.get_landmarks(aligned_face, "68")
    
    cv2.rectangle(aligned_face, (aligned_shape[54][0], aligned_shape[29][1]), #draw rectangle on right and left cheeks
            (aligned_shape[12][0],aligned_shape[33][1]), (0,255,0), 0)
    cv2.rectangle(aligned_face, (aligned_shape[4][0], aligned_shape[29][1]), 
            (aligned_shape[48][0],aligned_shape[33][1]), (0,255,0), 0)
    
    
    #assign to last params
    last_rects = rects
    last_age = age
    last_gender = gender
    last_shape = shape
    
    return display_frame, aligned_face

if __name__ == "__main__":
    
    cap = cv2.VideoCapture("1.mp4")
    fu = Face_utilities()
    i=0
    last_rects = None
    last_shape = None
    last_age = None
    last_gender = None
    
    face_detect_on = False
    age_gender_on = False

    t = time.time()
    
    while True:
        # grab a frame -> face detection -> crop the face -> 68 facial landmarks -> get mask from those landmarks

        # calculate time for each loop
        t0 = time.time()
        
        if(i%1==0):
            face_detect_on = True
            if(i%10==0):
                age_gender_on = True
            else:
                age_gender_on = False
        else: 
            face_detect_on = False
        
        ret, frame = cap.read()
        #frame_copy = frame.copy()
        
        if frame is None:
            print("End of video")
            break
        
        #display_frame, aligned_face = flow_process(frame)
        
        
        ret_process = fu.no_age_gender_face_process(frame, "68")
        
        if ret_process is None:
            cv2.putText(frame, "No face detected", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
            cv2.imshow("frame",frame)
            print(time.time()-t0)
            
            cv2.destroyWindow("face")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        rects, face, shape, aligned_face, aligned_shape = ret_process
        
        (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        #overlay_text = "%s, %s" % (gender, age)
        #cv2.putText(frame, overlay_text ,(x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
        
        if(len(aligned_shape)==68):
            cv2.rectangle(aligned_face,(aligned_shape[54][0], aligned_shape[29][1]), #draw rectangle on right and left cheeks
                    (aligned_shape[12][0],aligned_shape[33][1]), (0,255,0), 0)
            cv2.rectangle(aligned_face, (aligned_shape[4][0], aligned_shape[29][1]), 
                    (aligned_shape[48][0],aligned_shape[33][1]), (0,255,0), 0)
        else:
            #print(shape[4][1])
            #print(shape[2][1])
            #print(int((shape[4][1] - shape[2][1])))
            cv2.rectangle(aligned_face, (aligned_shape[0][0],int((aligned_shape[4][1] + aligned_shape[2][1])/2)),
                        (aligned_shape[1][0],aligned_shape[4][1]), (0,255,0), 0)
            
            cv2.rectangle(aligned_face, (aligned_shape[2][0],int((aligned_shape[4][1] + aligned_shape[2][1])/2)),
                        (aligned_shape[3][0],aligned_shape[4][1]), (0,255,0), 0)
        
        for (x, y) in aligned_shape: 
            cv2.circle(aligned_face, (x, y), 1, (0, 0, 255), -1)
        
        # display
        cv2.imshow("frame",frame)
        cv2.imshow("face",aligned_face)
        #cv2.imshow("mask",mask)
        i = i+1
        print(time.time()-t0)
        
        # waitKey to show the frame and break loop whenever 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
    cap.release()
    cv2.destroyAllWindows()

    print(time.time() - t)






