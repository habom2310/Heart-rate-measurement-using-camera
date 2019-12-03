import numpy as np
import cv2
import dlib
import os
import utils
from collections import OrderedDict

class Face_alignment:
    def __init__(self):
        model_path = os.getcwd() + "/shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(model_path)
        self.FACIAL_LANDMARKS_68_IDXS = OrderedDict([
            ("mouth", (48, 68)),
            ("right_eyebrow", (17, 22)),
            ("left_eyebrow", (22, 27)),
            ("right_eye", (36, 42)),
            ("left_eye", (42, 48)),
            ("nose", (27, 36)),
            ("jaw", (0, 17))
        ])

    def facial_landmark(self, face):
        h, w, c = face.shape
        shape = self.predictor(face, dlib.rectangle(0, 0, w, h))

        shape = utils.read_dlib_shape(shape)
        return np.array(shape)

    def align(self, face, shape, face_width = 256):
        desiredLeftEye=(0.35, 0.35)
        desiredFaceWidth = face_width
        desiredFaceHeight = desiredFaceWidth

        (lStart, lEnd) = self.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (rStart, rEnd) = self.FACIAL_LANDMARKS_68_IDXS["right_eye"]

        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]

        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
        
        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        
        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - desiredLeftEye[0]
        
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - desiredLeftEye[0])
        desiredDist *= desiredFaceWidth
        scale = desiredDist / dist
        
        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
        
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

                # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        
        # update the translation component of the matrix
        tX = desiredFaceWidth * 0.5
        tY = desiredFaceHeight * desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
        
        # apply the affine transformation
        (w, h) = (desiredFaceWidth, desiredFaceHeight)
        aligned_face = cv2.warpAffine(face, M, (w, h),
            flags=cv2.INTER_CUBIC)

        shape = np.reshape(shape,(68,1,2))

        aligned_shape = cv2.transform(shape, M)
        aligned_shape = np.squeeze(aligned_shape)

        return aligned_face, aligned_shape

from face_detection import Face_detection
if __name__ == "__main__":
    fd = Face_detection()
    fa = Face_alignment()

    img = cv2.imread("ed.jpg")
    # img = cv2.imread("flower.jpg")

    rects = fd.detect(img)

    if len(rects) == 0:
        print("no face detected")
    
    else:
        rect = rects[0]
        rect = utils.read_dlib_rect(rect)

        face = img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        shape = fa.facial_landmark(face)

        for (x, y) in shape:
            cv2.circle(face, (x, y), 1, (0, 0, 255), -1)
        cv2.imshow("face", face)
        cv2.waitKey()

        aligned_face, aligned_shape = fa.align(face, shape)

        cv2.imshow("aligned_face", aligned_face)
        cv2.waitKey()

        roi1, roi2 = utils.select_ROI(aligned_face, aligned_shape)
        cv2.imshow("roi1", roi1)
        cv2.imshow("roi2", roi2)
        cv2.waitKey()

        cv2.destroyAllWindows()
