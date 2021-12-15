# Heart-rate-measurement-using-camera
[![Alt text](https://github.com/habom2310/Heart-rate-measurement-using-camera/blob/master/result.JPG)](https://youtu.be/JeQGSEXk_BQ)
# Abstract
- Heart Rate (HR) is one of the most important Physiological parameter and a vital indicator of people‘s physiological state
- A non-contact based system to measure Heart Rate: real-time application using camera
- Principal: extract heart rate information from facial skin color variation caused by blood circulation 
- Application: monitoring drivers‘ physiological state

# Methods 
- Detect face, align and get ROI using facial landmarks
- Apply band pass filter with fl = 0.8 Hz and fh = 3 Hz, which are 48 and 180 bpm respectively
- Average color value of ROI in each frame is calculate pushed to a data buffer which is 150 in length
- FFT the data buffer. The highest peak is Heart rate
- Amplify color to make the color variation visible 

# Requirements
```
pip install -r requirements.txt
```


# Implementation
```
python GUI.py
```
- In case of plotting graphs, run "graph_plot.py" 
- For the Eulerian Video Magnification implementation, run "amplify_color.py"

# Results
- Data from a specialized device, Compact 5 medical Econet, is used for the ground truth. In certain circumstances, the Heart rate values measured using the application and the device are the same

# Reference
- Real Time Heart Rate Monitoring From Facial RGB Color Video Using Webcam by H. Rahman, M.U. Ahmed, S. Begum, P. Funk
- Remote Monitoring of Heart Rate using Multispectral Imaging in Group 2, 18-551, Spring 2015 by Michael Kellman Carnegie (Mellon University), Sophia Zikanova (Carnegie Mellon University) and Bryan Phipps (Carnegie Mellon University)
- Non-contact, automated cardiac pulse measurements using video imaging and blind source separation by Ming-Zher Poh, Daniel J. McDuff, and Rosalind W. Picard
- Camera-based Heart Rate Monitoring by Janus Nørtoft Jensen and Morten Hannemose
- Graphs plotting is based on https://github.com/thearn/webcam-pulse-detector
- https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

# Note
- Application can only detect HR for 1 people at a time
- Sudden change can cause incorrect HR calculation. In the most case, HR can be correctly detected after 10 seconds being stable infront of the camera
- This github project is for study purpose only. For other purposes, please contact me at khanhhanguyen2310@gmail.com

