import numpy as np
import cv2

def calculate_bright_intensity_of_environment(image, percentage = 0.1):
    image = np.array(image).reshape(-1)
    pixels = np.random.choices(image, int(image.shape[0]*percentage, replace=False))
    return np.average(pixels)

def read_dlib_rect(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return x,y,w,h

def read_dlib_shape(shape):
    np_shape = []
    for i in range(shape.num_parts):
        x = shape.part(i).x
        y = shape.part(i).y
        np_shape.append([x,y])

    return np_shape

def select_ROI(face, shape):
    roi1 = face[shape[29][1]:shape[33][1], #right cheek
            shape[54][0]:shape[12][0]]
            
    roi2 =  face[shape[29][1]:shape[33][1], #left cheek
            shape[4][0]:shape[48][0]]

    return roi1, roi2

def extract_mean_val(rois):
    g = []
    for roi in rois:
        g.append(np.mean(roi[:,:,1]))
    #b = np.mean(ROI[:,:,2])
    #return r, g, b
    mean_val = np.mean(g)
    return mean_val