#!/usr/bin/python3
import cv2
import numpy as np;
import configparser
import sys
import logging
from threading import Thread
import time

class VideoStreamWidget(object):
    def __init__(self, name, uri, gaussian_kernel_size, pixel_delta_threshold, fps):
        self.fps = fps
        self.gaussian_kernel_size = gaussian_kernel_size
        self.pixel_delta_threshold = pixel_delta_threshold
        self.name = name
        self.capture = cv2.VideoCapture(uri)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.last_frame = None
        self.gray = None

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            self.gray = cv2.GaussianBlur(self.gray, (gaussian_kernel_size,gaussian_kernel_size), 0)
             
            if self.last_frame is None:
                self.last_frame = self.gray
            
            frameDelta = cv2.absdiff(self.last_frame, self.gray)
            self.last_frame = self.gray
            thresh = cv2.threshold(frameDelta, self.pixel_delta_threshold, 255, cv2.THRESH_BINARY)[1]
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for contour in contours[:1]:
                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(self.frame,(x,y),(x+w,y+h),(0,0,255),3)
                logger.info("%s had object at %d,%d:%d,%d", self.name, x,y,w,h)

             # sleep within framerate for each camera (separate thread)
            time.sleep(1/fps/2)
    
    def show_frame(self):
        cv2.imshow(self.name, self.frame)
        key = cv2.waitKey(1)

def read_config(config_file):
    global config
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
	
    global logfile
    logfile = config['general']['logfile']
    
    global cameras
    cameras = dict(config['cameras']) 


cameras = {}
streams = {}

read_config(sys.argv[1]) 

log_level_info = {  "DEBUG" : logging.DEBUG, 
                    "INFO": logging.INFO,
                    "WARNING": logging.WARNING,
                    "ERROR": logging.ERROR,
                    }

my_log_level_from_config = config['general']['log_level']
my_log_level = log_level_info[my_log_level_from_config]

gaussian_kernel_size = int(config['motion']['gaussian_kernel_size'])
pixel_delta_threshold = int(config['motion']['pixel_delta_threshold'])
fps = int(config['motion']['fps'])
display_camera_windows = int(config['general']['display_camera_windows'])

logging.basicConfig(    handlers=[
								logging.FileHandler(logfile),
								logging.StreamHandler()],
						format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
						datefmt='%Y-%m-%d:%H:%M:%S',
						level=my_log_level)

logger = logging.getLogger(__name__)

logger.info("OpenCVCam started")

for name,uri in cameras.items():
    streams[name] = VideoStreamWidget(name, uri, gaussian_kernel_size, pixel_delta_threshold, fps)

while True:
    for name,uri in cameras.items():
        try:
            if display_camera_windows:
                streams[name].show_frame()
        except AttributeError:
            pass
    # sleep within framerate over all cameras (single thread)
    time.sleep(1/fps/2)