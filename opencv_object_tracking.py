# python opencv_object_tracking.py --video 1.mp4 --tracker csrt

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import rainflow
from scipy.interpolate import UnivariateSpline
from scipy.signal import wiener, filtfilt, butter, gaussian, freqz
from scipy.ndimage import filters
import scipy.optimize as op
import pandas as pd
from scipy import fftpack
from scipy import signal
from scipy import optimize
import ctypes

def Mbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)
def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

class Plotter:
    def __init__(self, plot_width, plot_height,sample_buffer=None):
        self.width = plot_width
        self.height = plot_height
        self.color = (255, 0 ,0)
        self.plot_canvas = np.ones((self.height, self.width, 3))*255    
        self.ltime = 0
        self.plots = {}
        self.plot_t_last = {}
        self.margin_l = 10
        self.margin_r = 10
        self.margin_u = 10
        self.margin_d = 50
        self.sample_buffer = self.width if sample_buffer is None else sample_buffer


    # Update new values in plot
    def plot(self, val, label = "plot"):
        if not label in self.plots:
            self.plots[label] = []
            self.plot_t_last[label] = 0
            
        self.plots[label].append(int(val))
        
        while len(self.plots[label]) > self.sample_buffer:
            self.plots[label].pop(0)
            self.show_plot(label)
   
    def plotarray(self, val, label = "plot"):
        if not label in self.plots:
            self.plots[label] = []
            self.plot_t_last[label] = 0
        for i in val :    
            self.plots[label].append(int(i))
        
        while len(self.plots[label]) > self.sample_buffer:
            self.plots[label].pop(0)
            self.show_plot(label)        

    # Show plot using opencv imshow
    def show_plot(self, label):

        self.plot_canvas = np.zeros((self.height, self.width, 3))*255
        cv2.line(self.plot_canvas, (self.margin_l, int((self.height-self.margin_d-self.margin_u)/2)+self.margin_u ), (self.width-self.margin_r, int((self.height-self.margin_d-self.margin_u)/2)+self.margin_u), (0,0,255), 1)        

        # Scaling the graph in y within buffer
        scale_h_max = max(self.plots[label])
        scale_h_min = min(self.plots[label]) 
        scale_h_min = -scale_h_min if scale_h_min<0 else scale_h_min
        scale_h = scale_h_max if scale_h_max > scale_h_min else scale_h_min
        scale_h = ((self.height-self.margin_d-self.margin_u)/2)/scale_h if not scale_h == 0 else 0
        

        for j,i in enumerate(np.linspace(0,self.sample_buffer-2,self.width-self.margin_l-self.margin_r)):
            i = int(i)
            cv2.line(self.plot_canvas, (j+self.margin_l, int((self.height-self.margin_d-self.margin_u)/2 +self.margin_u- self.plots[label][i]*scale_h)), (j+self.margin_l, int((self.height-self.margin_d-self.margin_u)/2  +self.margin_u- self.plots[label][i+1]*scale_h)), self.color, 1)
        
        
        cv2.rectangle(self.plot_canvas, (self.margin_l,self.margin_u), (self.width-self.margin_r,self.height-self.margin_d), (255,255,255), 1) 
        cv2.putText(self.plot_canvas,f" {label} : {self.plots[label][-1]} , dt : {int((time.time() - self.plot_t_last[label])*1000)}ms",(int(0),self.height-20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        cv2.circle(self.plot_canvas, (self.width-self.margin_r, int(self.margin_u + (self.height-self.margin_d-self.margin_u)/2 - self.plots[label][-1]*scale_h)), 2, (0,200,200), -1)
        
        self.plot_t_last[label] = time.time()
        cv2.imshow(label, self.plot_canvas)
        cv2.waitKey(1)

def parse_ICA_results(ICA, buffer_window): #time
	

	# ** FOR RGB CHANNELS & ICA **
	#one = np.squeeze(np.asarray(ICA[:, 0])).tolist()
	
	
	#one = (np.hamming(len(one)) * one)
	

	# print "one: ", one.astype(float).tolist()
	

	#one = np.absolute(np.square(np.fft.irfft(one))).astype(float).tolist()
	

	#power_ratio = [0, 0, 0]
	#power_ratio[0] = np.sum(one)/np.amax(one)
	

	
	#signals["array"] = one
	
	# print power_ratio
	# print signals
	#return signals

	# # ** FOR GREEN CHANNEL ONLY **
	hamming = (np.hamming(len(ICA)) * ICA)
	fft = np.fft.rfft(hamming)
	fft = np.absolute(np.square(fft))
	signals = fft.astype(float).tolist()

	return signals


	# ** experiments **
	# ** for interpolation and hamming **
	# even_times = np.linspace(time[0], time[-1], len(time))
	# interpolated_two = np.interp(even_times, time, np.squeeze(np.asarray(ICA[:, 1])).tolist())
	# interpolated_two = np.hamming(len(time)) * interpolated_two
	# interpolated_two = interpolated_two - np.mean(interpolated_two)

	# signals["two"] = interpolated_two.tolist()

	# #fft = np.fft.rfft(np.squeeze(np.asarray(ICA[:, 1])))
	# #signals["two"] = fft.astype(float).tolist()


def normalize_matrix(matrix):
	# ** for matrix
	for array in matrix:
		average_of_array = np.mean(array)
		std_dev = np.std(array)

		for i in range(len(array)):
			array[i] = ((array[i] - average_of_array)/std_dev)
	return matrix

def normalize_array(array):
	#** for array
	average_of_array = np.mean(array)
	std_dev = np.std(array)

	for i in range(len(array)):
		array[i] = ((array[i] - average_of_array)/std_dev)
	return array

def butter_lowpass_filter(data, cutoff, fs, order):
    print("Cutoff freq " + str(cutoff))
    nyq = 0.5 * fs # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a,data)
    return y

def get_similarity(image1, image2):
    """This function returns the absolute
    value of the entered number"""
    nsumFrameColor = 0.0
    
    gray_image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray_image], [0], 
                         None, [256], [0, 256])

    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    histogram2 = cv2.calcHist([gray_image2], [0], 
                          None, [256], [0, 256])
    i = 0
    while i<len(histogram) and i<len(histogram2):
        nsumFrameColor+=(histogram[i]-histogram2[i])**2
        i+= 1
    nsumFrameColor = nsumFrameColor**(1 / 2)
    
    return nsumFrameColor

def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def test_func(x, a, b,c):
    return a * np.sin(b * x)+c

def get_frame_value(image, width, height):
    """This function returns the absolute
    value of the entered number"""
    nsumFrameColor = 0.0
    for i in range(width):
        for j in range(height):
            color = image[j,i]
            nsumFrameColor += float(color[0]/(width*height))
   
    return nsumFrameColor

def plotScatter(plt, x, y):
    plt.scatter(x, y)
    plt.pause(0.05)

def filterFreq(fftArray, freqs, framerate):
    filteredFFT = list()
    filteredFreqBin = list()

    for i in range(len(fftArray)):
        if (fftArray[i] > 0.80) and (fftArray[i] < 3):
            
            filteredFFT.append(freqs[i])
            filteredFreqBin.append((fftArray[i])/1)
		
    for i in range(len(filteredFFT)):
        filteredFFT[i] = abs(i) ** 2
    normalizedFreqs = filteredFFT

    idx = normalizedFreqs.index(min(normalizedFreqs))
    freq_in_hertz = filteredFreqBin[idx]
	
    freqs = {'normalizedFreqs': normalizedFreqs, 'filteredFreqBin': filteredFreqBin, 'freq_in_hertz': freq_in_hertz}
    return freqs

def frequencyExtract(spectrum, fps_video):
    
    val = None
    n = None 
    d = None
    N = None
    p1 = list()
    p2 = list()
    results = list()
    freqs = list()

    #from numpy.fft.fftfreq doc: "Sample spacing (inverse of the sampling rate). Defaults to 1. For instance, if the sample spacing is in seconds, then the frequency unit is cycles/second."
   	# if my frequency is 15 fps, or 15 Hz, or every 66ms, then the sample spacing is .066 seconds (1/15)
   
    n = len(spectrum)
    d = 1.0/fps_video
   
    val = 1.0/(n * d)
    N = int(((n - 1)/2 + 1))
    for i in range(N):
        p1.append(i)
    i = int(-(n/2))
    while i<0:
        
        p2.append(i)
        i += 1
        
    for i in p2 :
        p1.append(p2[i])
    
    results = p1
    for i in range(len(results)):
        p1[i] = i * val
    freqs = p1
    return filterFreq(spectrum, freqs, fps_video)

def findPeak(signal_arr):
    dary = signal_arr
    dary -= np.average(dary)

    step = np.hstack((np.ones(len(dary)), -1*np.ones(len(dary))))

    dary_step = np.convolve(dary, step, mode='valid')

    # Get the peaks of the convolution
    peaks = signal.find_peaks(dary_step, width=30)[0]
    return peaks
#plt.ion()  ## Note this correction
#fig = plt.figure()

#plt.title("Real Time plot")
#plt.xlabel("frame")
#plt.ylabel("result")

xlist = list()
ylist = list()
frameno = 0

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
                help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
                help="OpenCV object tracker type")
args = vars(ap.parse_args())

# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]

# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
    tracker = cv2.Tracker_create(args["tracker"].upper())

# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "mil": cv2.TrackerMIL_create,

    }

    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

# initialize the FPS throughput estimator
fps = None
fps_video = vs.get(cv2.CAP_PROP_FPS)

index = 0
indexFrm = 0
v = 0
# loop over frames from the video stream
p = Plotter(400, 200,sample_buffer=200)
first_Image = None
maxcl = 0
pplotpix = []
index_per_cycle = 0
rate_per_ten = list()
heart_rate_realtime = list()
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    # check to see if we have reached the end of the stream
    if frame is None:
        break

    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    frame = imutils.resize(frame, width=1000)
    (H, W) = frame.shape[:2]

    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = (1, initBB)

        # check to see if the tracking was a success
        if success:
            
            
          #plotScatter(plt,index,int(math.sin(index*3.14/180)*100))
              
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            crop_img = frame[y:y + h, x:x + w]
            if(int(index) == 0):
                first_Image = crop_img
                
            fileName = "cropped_" + str(index) + "_.jpg"
            index += 1
            cv2.imshow("cropped field", crop_img)
            #p.plotarray(y_data,label='result')
            
            if index < fps_video * 3 :
                gray_image = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                Binary = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY,11,2)
                cl = np.average(Binary)
                
                if cl > maxcl:
                    
                    maxcl = cl
                    first_Image = crop_img
                    text = "{}: {}".format(index+1, cl)
                    cv2.putText(crop_img, text,( 10,  20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    fileName = str(index+1)+"_train.png"
                    #cv2.imwrite(fileName, crop_img)
            else:
                
                cl = get_similarity(first_Image,crop_img)
                pplotpix.append(cl)
                p.plot(cl,label='result')
                indexFrm += 1
                
                if indexFrm >= fps_video * 7 and indexFrm % fps_video == 0:
                    y_data = np.array(pplotpix)
                    x_data = np.arange(0,len(y_data))
                    y_data = normalize_array(y_data[:,0])
                    
                    #
                    s = UnivariateSpline(x_data, y_data, s=50)
                    yy_data = s(x_data)
                    
                    peaks = findPeak(yy_data)
                    
                    cycle = 0
                    heatrate = 0
                    if(len(peaks) > 1):
                        for ii in range(len(peaks)):
                            if ii < len(peaks)-1 :
                                cycle += (peaks[ii+1]-peaks[ii])
                        cycle  = cycle/(len(peaks)-1)
                        cycle = cycle/fps_video
                        heatrate = 60/ cycle
                        heatrate *= 6

                    if(heatrate != 0):
                        index_per_cycle += 1
                        # if index_per_cycle % 10 == 0:
                        rate_per_ten.append(heatrate)
                        heart_rate_realtime.append(heatrate)
                    
                        print("Heart rate=>"+str(heatrate))
            #cv2.imwrite(fileName, crop_img)
        area = (w - x) * (h - y)

        #print(x, y, w, h)
        #print(area)
        
        frameno += 1
        xlist.append(frameno)
        ylist.append(area)
        

        fps.update()

        fps.stop()

        # initialize the set of information we'll be displaying on
        # the frame
        info = [
            ("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),

        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                               showCrosshair=False)
        #print(initBB)
        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(frame, initBB)
        fps = FPS().start()
        
        
    # cv2.imshow('image',initBB)


    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break


y_data = np.array(pplotpix)
x_data = np.arange(0,len(y_data))
y_data = normalize_array(y_data[:,0])

#
s = UnivariateSpline(x_data, y_data, s=50)
yy_data = s(x_data)

peaks = findPeak(yy_data)

# cycle = 0
# heatrate = 0
# if(len(peaks) > 1):
#     for ii in range(len(peaks)):
#         if ii < len(peaks)-1 :
#             cycle += (peaks[ii+1]-peaks[ii])
#     cycle  = cycle/(len(peaks)-1)
#     cycle = cycle/fps_video
#     heatrate = 60/ cycle

# print("total_Heart rate=>"+str(heatrate))
# heatrate = round_half_up(heatrate, 0)
# heatrate *= 6

heartrate = 0
for i in range(0,index_per_cycle):
    heartrate += rate_per_ten[i]

heartrate = heartrate / index_per_cycle
print("total_Heart rate=>"+str(heartrate))

heartrate = round_half_up(heartrate, 0)
if(heartrate < 10):
    heartrate = heart_rate_realtime[-1]
else:
    heartrate = heartrate - 10

# Mbox("Measure result", "Total heartbeat of this animal is " + str(heatrate), 0)
Mbox("Measure result", "Total heartbeat of this animal is " + str(heartrate), 0)

# plots
plt.figure()

plt.plot(yy_data)

#plt.plot(dary_step/10)

for ii in range(len(peaks)):
    plt.plot((peaks[ii], peaks[ii]), (-3, 3), 'r')

plt.show()


# data_save = {'Heartbeat every 10 seconds': heart_rate_realtime, 'Average heart rate(/min)': heatrate}
data_save = {'Heartbeat every 10 seconds': heart_rate_realtime, 'Average heart rate(/min)': heartrate}
df_save = pd.DataFrame(data = data_save)
df_save.to_csv('result_bps.csv')

Mbox("Success message", "Result saved to 'result_bps.csv' successfully!", 0)

# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()

# otherwise, release the file pointer
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()