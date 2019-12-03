import cv2
import numpy as np
import imutils
import scipy.signal as signal
import scipy.fftpack as fftpack
import time
import sys
from webcam import Webcam
from video import Video
from face_detection import FaceDetection
from interface import waitKey, plotXY


class VidMag():
    def __init__(self):
        self.webcam = Webcam()
        self.buffer_size = 40
        self.fps = 0
        self.times = []
        self.t0 = time.time()
        self.data_buffer = []
        #self.vidmag_frames = []
        self.frame_out = np.zeros((10,10,3),np.uint8)
        self.webcam.start()
        print("init")
        
    #--------------COLOR MAGNIFICATIONN---------------------#    
    def build_gaussian_pyramid(self,src,level=3):
        s=src.copy()
        pyramid=[s]
        for i in range(level):
            s=cv2.pyrDown(s)
            pyramid.append(s)
        return pyramid
        
    def gaussian_video(self,video_tensor,levels=3):
        for i in range(0,video_tensor.shape[0]):
            frame=video_tensor[i]
            pyr=self.build_gaussian_pyramid(frame,level=levels)
            gaussian_frame=pyr[-1]
            if i==0:
                vid_data=np.zeros((video_tensor.shape[0],gaussian_frame.shape[0],gaussian_frame.shape[1],3))
            vid_data[i]=gaussian_frame
        return vid_data
        
    def temporal_ideal_filter(self,tensor,low,high,fps,axis=0):
        fft=fftpack.fft(tensor,axis=axis)
        frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)
        bound_low = (np.abs(frequencies - low)).argmin()
        bound_high = (np.abs(frequencies - high)).argmin()
        fft[:bound_low] = 0
        fft[bound_high:-bound_high] = 0
        fft[-bound_low:] = 0
        iff=fftpack.ifft(fft, axis=axis)
        return np.abs(iff)   
            
        
    def amplify_video(self,gaussian_vid,amplification=70):
        return gaussian_vid*amplification
        
    def reconstract_video(self,amp_video,origin_video,levels=3):
        final_video=np.zeros(origin_video.shape)
        for i in range(0,amp_video.shape[0]):
            img = amp_video[i]
            for x in range(levels):
                img=cv2.pyrUp(img)
            img=img+origin_video[i]
            final_video[i]=img
        return final_video    

    def magnify_color(self,data_buffer,fps,low=0.4,high=2,levels=3,amplification=30):
        gau_video=self.gaussian_video(data_buffer,levels=levels)
        filtered_tensor=self.temporal_ideal_filter(gau_video,low,high,fps)
        amplified_video=self.amplify_video(filtered_tensor,amplification=amplification)
        final_video = self.reconstract_video(amplified_video,data_buffer,levels=levels)
        #print("c")
        return final_video
    #-------------------------------------------------------------#    
    
    #-------------------MOTION MAGNIFICATIONN---------------------#
    #build laplacian pyramid for video
    def laplacian_video(self,video_tensor,levels=3):
        tensor_list=[]
        for i in range(0,video_tensor.shape[0]):
            frame=video_tensor[i]
            pyr=self.build_laplacian_pyramid(frame,levels=levels)
            if i==0:
                for k in range(levels):
                    tensor_list.append(np.zeros((video_tensor.shape[0],pyr[k].shape[0],pyr[k].shape[1],3)))
            for n in range(levels):
                tensor_list[n][i] = pyr[n]
        return tensor_list   
        
    #Build Laplacian Pyramid
    def build_laplacian_pyramid(self, src,levels=3):
        gaussianPyramid = self.build_gaussian_pyramid(src, levels)
        pyramid=[]
        for i in range(levels,0,-1):
            GE=cv2.pyrUp(gaussianPyramid[i])
            L=cv2.subtract(gaussianPyramid[i-1],GE)
            pyramid.append(L)
        return pyramid
        
    #reconstract video from laplacian pyramid
    def reconstract_from_tensorlist(self,filter_tensor_list,levels=3):
        final=np.zeros(filter_tensor_list[-1].shape)
        for i in range(filter_tensor_list[0].shape[0]):
            up = filter_tensor_list[0][i]
            for n in range(levels-1):
                up=cv2.pyrUp(up)+filter_tensor_list[n + 1][i]
            final[i]=up
        return final    
    
    #butterworth bandpass filter
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        omega = 0.5 * fs
        low = lowcut / omega
        high = highcut / omega
        b, a = signal.butter(order, [low, high], btype='band')
        y = signal.lfilter(b, a, data, axis=0)
        return y
    
    def magnify_motion(self,video_tensor,fps,low=0.4,high=1.5,levels=3,amplification=30):
        lap_video_list=self.laplacian_video(video_tensor,levels=levels)
        filter_tensor_list=[]
        for i in range(levels):
            filter_tensor=self.butter_bandpass_filter(lap_video_list[i],low,high,fps)
            filter_tensor*=amplification
            filter_tensor_list.append(filter_tensor)
        recon=self.reconstract_from_tensorlist(filter_tensor_list)
        final=video_tensor+recon
        return final
    #-------------------------------------------------------------# 
    
    
    def buffer_to_tensor(self, buffer): 
        tensor = np.zeros((len(buffer), 192, 256, 3), dtype = "float")
        i = 0
        for i in range(len(buffer)):
            tensor[i] = buffer[i]
        return tensor
        
    def run_color(self):
        self.times.append(time.time() - self.t0)
        L = len(self.data_buffer)
        #print(self.data_buffer)
        
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            #self.vidmag_frames = self.vidmag_frames[-self.buffer_size:]
            L = self.buffer_size
        
        if len(self.data_buffer) > self.buffer_size-1:
            self.fps = float(L) / (self.times[-1] - self.times[0])
            tensor = self.buffer_to_tensor(self.data_buffer)
            final_vid = self.magnify_color(data_buffer = tensor, fps = self.fps)
            #print(final_vid[0].shape)
            #self.vidmag_frames.append(final_vid[-1])
            #print(self.fps)
            self.frame_out = final_vid[-1]
    
    def run_motion(self):
        self.times.append(time.time() - self.t0)
        L = len(self.data_buffer)
        #print(L)
        
        if L > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            #self.vidmag_frames = self.vidmag_frames[-self.buffer_size:]
            L = self.buffer_size
        
        if len(self.data_buffer) > self.buffer_size-1:
            self.fps = float(L) / (self.times[-1] - self.times[0])
            tensor = self.buffer_to_tensor(self.data_buffer)
            final_vid = self.magnify_motion(video_tensor = tensor, fps = self.fps)
            #print(self.fps)
            #self.vidmag_frames.append(final_vid[-1])
            self.frame_out = final_vid[-1]
    
    def key_handler(self):
        """
        A plotting or camera frame window must have focus for keypresses to be
        detected.
        """
        self.pressed = waitKey(1) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print("[INFO] Exiting")
            self.webcam.stop()
            sys.exit()    
            
    def mainLoop(self):
        frame = self.webcam.get_frame()
        f1 = imutils.resize(frame, width = 256)
        #crop_frame = frame[100:228,200:328]
        self.data_buffer.append(f1)
        self.run_color()
        #print(frame)
        
        #if len(self.vidmag_frames) > 0:
            #print(self.vidmag_frames[0])
        cv2.putText(frame, "FPS "+str(float("{:.2f}".format(self.fps))),
                       (20,420), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0),2)
            
        #frame[100:228,200:328] = cv2.convertScaleAbs(self.vidmag_frames[-1])
        cv2.imshow("Original",frame)
        #f2 = imutils.resize(cv2.convertScaleAbs(self.vidmag_frames[-1]), width = 640)
        f2 = imutils.resize(cv2.convertScaleAbs(self.frame_out), width = 640)
            
        cv2.imshow("Color amplification",f2)
            
            
        self.key_handler()  #if not the GUI cant show anything
    
    
if __name__ == "__main__":
    #print("a")
    app = VidMag()
    while True:
        app.mainLoop()
    
    
    
    
    
    
    
    
    
    