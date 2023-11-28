import cv2
import numpy as np
import time
from face_detection import FaceDetection
from scipy import signal
from face_utilities import Face_utilities
from signal_processing import Signal_processing
from imutils import face_utils
# from sklearn.decomposition import FastICA

class Process(object):
    def __init__(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.buffer_size = 100
        self.times = [] 
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.fd = FaceDetection()
        self.bpms = []
        self.peaks = []
        self.fu = Face_utilities()
        self.sp = Signal_processing()

        #self.red = np.zeros((256,256,3),np.uint8)
        
    def extractColor(self, frame):
        
        #r = np.mean(frame[:,:,0])
        g = np.mean(frame[:,:,1])
        #b = np.mean(frame[:,:,2])
        #return r, g, b
        return g
        
    def run(self):
        # frame, face_frame, ROI1, ROI2, status, mask = self.fd.face_detect(self.frame_in)
        
        frame = self.frame_in
        ret_process = self.fu.no_age_gender_face_process(frame, "5")
        if ret_process is None:
            return False
        rects, face, shape, aligned_face, aligned_shape = ret_process

        (x, y, w, h) = face_utils.rect_to_bb(rects[0])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        if(len(aligned_shape)==68):
            cv2.rectangle(aligned_face,(aligned_shape[54][0], aligned_shape[29][1]), #draw rectangle on right and left cheeks
                    (aligned_shape[12][0],aligned_shape[33][1]), (0,255,0), 0)
            cv2.rectangle(aligned_face, (aligned_shape[4][0], aligned_shape[29][1]), 
                    (aligned_shape[48][0],aligned_shape[33][1]), (0,255,0), 0)
        else:
            cv2.rectangle(aligned_face, (aligned_shape[0][0],int((aligned_shape[4][1] + aligned_shape[2][1])/2)),
                        (aligned_shape[1][0],aligned_shape[4][1]), (0,255,0), 0)
            
            cv2.rectangle(aligned_face, (aligned_shape[2][0],int((aligned_shape[4][1] + aligned_shape[2][1])/2)),
                        (aligned_shape[3][0],aligned_shape[4][1]), (0,255,0), 0)
        
        for (x, y) in aligned_shape: 
            cv2.circle(aligned_face, (x, y), 1, (0, 0, 255), -1)


        ROIs = self.fu.ROI_extraction(aligned_face, aligned_shape)
        green_val = self.sp.extract_color(ROIs)

        self.frame_out = frame
        self.frame_ROI = aligned_face
        
        # g1 = self.extractColor(ROI1)
        # g2 = self.extractColor(ROI2)
        #g3 = self.extractColor(ROI3)
        
        L = len(self.data_buffer)
        
        #calculate average green value of 2 ROIs
        #r = (r1+r2)/2
        #g = (g1+g2)/2
        #b = (b1+b2)/2

        g = green_val
        
        if(abs(g-np.mean(self.data_buffer))>10 and L>99): #remove sudden change, if the avg value change is over 10, use the mean of the data_buffer
            g = self.data_buffer[-1]
        
        self.times.append(time.time() - self.t0)
        self.data_buffer.append(g)

        #only process in a fixed-size buffer
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            self.bpms = self.bpms[-self.buffer_size//2:]
            L = self.buffer_size
            
        processed = np.array(self.data_buffer)
        
        # start calculating after the first 10 frames
        if L == self.buffer_size:
            
            self.fps = float(L) / (self.times[-1] - self.times[0])#calculate HR using a true fps of processor of the computer, not the fps the camera provide
            even_times = np.linspace(self.times[0], self.times[-1], L)
            
            processed = signal.detrend(processed)#detrend the signal to avoid interference of light change
            interpolated = np.interp(even_times, self.times, processed) #interpolation by 1
            interpolated = np.hamming(L) * interpolated#make the signal become more periodic (advoid spectral leakage)
            #norm = (interpolated - np.mean(interpolated))/np.std(interpolated)#normalization
            norm = interpolated/np.linalg.norm(interpolated)
            raw = np.fft.rfft(norm*30)#do real fft with the normalization multiplied by 10
            
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)
            freqs = 60. * self.freqs
            
            # idx_remove = np.where((freqs < 50) & (freqs > 180))
            # raw[idx_remove] = 0
            
            self.fft = np.abs(raw)**2#get amplitude spectrum
        
            idx = np.where((freqs > 50) & (freqs < 180))#the range of frequency that HR is supposed to be within 
            pruned = self.fft[idx]
            pfreq = freqs[idx]
            
            self.freqs = pfreq 
            self.fft = pruned
            
            idx2 = np.argmax(pruned)#max in the range can be HR
            
            self.bpm = self.freqs[idx2]
            self.bpms.append(self.bpm)
            
            processed = self.butter_bandpass_filter(processed,0.8,3,self.fps,order = 3)
            #ifft = np.fft.irfft(raw)
        self.samples = processed # multiply the signal with 5 for easier to see in the plot
        #TODO: find peaks to draw HR-like signal.
        
        # if(mask.shape[0]!=10): 
        #     out = np.zeros_like(aligned_face)
        #     mask = mask.astype(np.bool)
        #     out[mask] = aligned_face[mask]
        #     if(processed[-1]>np.mean(processed)):
        #         out[mask,2] = 180 + processed[-1]*10
        #     aligned_face[mask] = out[mask]
            
            
        #cv2.imshow("face", face_frame)
        #out = cv2.add(face_frame,out)
        # else:
            # cv2.imshow("face", face_frame)
        return True
    
    def reset(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.times = [] 
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.freqs = []
        self.t0 = time.time()
        self.bpm = 0
        self.bpms = []
        
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a


    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y 
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
