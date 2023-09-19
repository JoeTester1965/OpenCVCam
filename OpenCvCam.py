#!/usr/bin/python3
import cv2
import numpy as np;
import configparser
import sys
import logging
from threading import Thread
import time
import os

class MotionVectorWeight(object):
    def __init__(self, threshold, fade, gain):
        self.threshold = threshold
        self.fade = fade
        self.gain = gain
        self.weight = 0
        self.lastx = 0
        self.lasty = 0
        self.lastw = 0
        self.lasth = 0

    def update(self, x, y, w, h):
        if (((x+w) < self.lastx) or (x > (self.lastx+self.lastw))) and (((y+h) < self.lasty) or (y > (self.lasty+self.lasth))):
            # motion rectangles not overlapping
            self.weight = self.weight - (1 * self.fade)
        else:
            self.weight = self.weight + (1 * self.gain)

        if self.weight < 0:
           self.weight = 0 
        
        if self.weight > self.threshold:
            self.weight = 0
            return True
         
        return False

class VideoStreamWidget(object):
    def __init__(self, name, uri, gaussian_kernel_size, pixel_delta_threshold, 
                    fps, masks_directory, minimum_motion_screen_percent):
        self.fps = fps
        self.masks_directory = masks_directory
        self.minimum_motion_screen_percent = minimum_motion_screen_percent
        self.gaussian_kernel_size = gaussian_kernel_size
        self.pixel_delta_threshold = pixel_delta_threshold
        self.name = name
        self.capture = cv2.VideoCapture(uri)
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.last_frame = None
        self.gray = None
        self.mask = None
        self.image_pixels = None

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            
            if self.last_frame is None:
                self.last_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                self.last_frame = cv2.GaussianBlur(self.last_frame, (gaussian_kernel_size,gaussian_kernel_size), 0)
                width,height = self.last_frame.shape
                self.image_pixels = width * height
                mask_template_name = f'{masks_directory}/{self.name}-mask.jpg'
                if os.path.exists(mask_template_name):
                    #use existing mask
                    self.mask = cv2.imread(mask_template_name)
                    self.mask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
                else:
                    #save new candidate template mask                 
                    candidate_mask_template_name = f'{masks_directory}/{self.name}-mask.candidate.jpg'
                    cv2.imwrite(candidate_mask_template_name, self.frame)

            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            if self.mask is not None:
                self.gray = cv2.bitwise_and(self.gray,self.mask)

            self.gray = cv2.GaussianBlur(self.gray, (gaussian_kernel_size,gaussian_kernel_size), 0)
            frameDelta = cv2.absdiff(self.last_frame, self.gray)
            self.last_frame = self.gray
            thresh = cv2.threshold(frameDelta, self.pixel_delta_threshold, 255, cv2.THRESH_BINARY)[1]
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for contour in contours[:1]:
                x,y,w,h = cv2.boundingRect(contour)
                if (((w * h) / self.image_pixels) * 100) > minimum_motion_screen_percent:
                    logger.debug("%s had object at %d,%d:%d,%d", self.name, x,y,w,h)
                    if motion[self.name].update( x, y, w, h) == True:
                        cv2.rectangle(self.frame,(x,y),(x+w,y+h),(0,0,255),3)
                        logger.info("%s has motion detected", self.name)
                    else:
                        cv2.rectangle(self.frame,(x,y),(x+w,y+h),(0,255,0),3)

             # sleep within framerate for each camera (separate thread)
            time.sleep(1/fps/2)
    
    def show_frame(self):
        cv2.imshow(self.name, self.frame)
        key = cv2.waitKey(1)

def read_config(config_file):
    global config
    config = configparser.ConfigParser()
    config.read(sys.argv[1])

read_config(sys.argv[1]) 

cameras = dict(config['cameras']) 
	
logfile = config['general']['logfile']

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
minimum_motion_screen_percent = float(config['motion']['minimum_motion_screen_percent'])
threshold  = float(config['motion']['threshold'])
fade  = float(config['motion']['fade'])
gain  = float(config['motion']['gain'])

display_camera_windows = int(config['general']['display_camera_windows'])
masks_directory = config['general']['masks_directory']


logging.basicConfig(    handlers=[
								logging.FileHandler(logfile),
								logging.StreamHandler()],
						format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
						datefmt='%Y-%m-%d:%H:%M:%S',
						level=my_log_level)

logger = logging.getLogger(__name__)

logger.info("OpenCVCam started")

streams = {}

motion = {}

for name,uri in cameras.items():
    streams[name] = VideoStreamWidget(name, uri, gaussian_kernel_size, pixel_delta_threshold, 
                                            fps, masks_directory, minimum_motion_screen_percent)
    motion[name] = MotionVectorWeight(threshold, fade, gain)

while True:
    for name,uri in cameras.items():
        try:
            if display_camera_windows:
                streams[name].show_frame()
        except AttributeError:
            pass
    # sleep within framerate over all cameras (single thread)
    time.sleep(1/fps/2)