#!/usr/bin/python3
import cv2
import numpy as np;
import configparser
import sys
import logging
from threading import Thread
import time
import os

class VideoStreamWidget(object):
    def __init__(self, name, uri, gaussian_kernel_size, pixel_delta_threshold, 
                    fps, masks_directory, minimum_motion_screen_percent,
                    image_rescaling_factor):
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
            
            if self.status:
                if image_rescaling_factor != 1.0:
                    self.frame = cv2.resize(self.frame, (0, 0), fx = image_rescaling_factor, fy = image_rescaling_factor, interpolation = cv2.INTER_LINEAR)
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
                        width,height = self.mask.shape
                        if width*height != self.image_pixels:
                            #video size has changed, cannot use current mask
                            self.mask = None
                    else:
                        #save new candidate template mask                 
                        candidate_mask_template_name = f'{masks_directory}/{self.name}-mask.candidate.jpg'
                        cv2.imwrite(candidate_mask_template_name, self.frame)

                self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

                if self.mask is not None:
                    self.gray = cv2.bitwise_and(self.gray,self.mask)

                self.gray = cv2.GaussianBlur(self.gray, ( gaussian_kernel_size,gaussian_kernel_size ), 0)
                frameDelta = cv2.absdiff(self.last_frame, self.gray)
                self.last_frame = self.gray
                thresh = cv2.threshold(frameDelta, self.pixel_delta_threshold, 255, cv2.THRESH_BINARY)[1]
                kernel = np.ones((delta_frame_dilation,delta_frame_dilation),np.uint8)
                thresh = cv2.dilate(thresh, kernel, iterations = 1)
                cv2.imshow("Debug", thresh)
                cv2.waitKey(1)
                contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
                contours = contours[0] if len(contours) == 2 else contours[1]
                contours = sorted(contours, key=cv2.contourArea, reverse=True)

                if display_contour_debug:
                    for contour in contours:
                        x,y,w,h = cv2.boundingRect(contour)
                        cv2.rectangle(self.frame,(x,y),(x+w,y+h),(255,255,255),3)

                for contour in contours[:motion_contours_to_consider ]:
                    x,y,w,h = cv2.boundingRect(contour)
                    if (((w * h) / self.image_pixels) * 100) > minimum_motion_screen_percent:
                        logger.info("%s : significant object at %d,%d:%d,%d", self.name, x,y,w,h)
                        cv2.rectangle(self.frame,(x,y),(x+w,y+h),(0,0,255),5)

                # sleep within framerate for each camera (separate thread)
                time.sleep(1/fps/2)
            else:
                #
                # Camera down !
                #
                pass
    
    def show_frame(self):
        try:
            cv2.imshow(self.name, self.frame)
            key = cv2.waitKey(1)
        except:
            pass

def read_config(config_file):
    global config
    config = configparser.ConfigParser()
    config.read(sys.argv[1])

read_config(sys.argv[1]) 

#camera config
cameras = dict(config['cameras']) 
	
#motion config
image_rescaling_factor = float(config['motion']['image_rescaling_factor'])
gaussian_kernel_size = int(config['motion']['gaussian_kernel_size'])
pixel_delta_threshold = int(config['motion']['pixel_delta_threshold'])
delta_frame_dilation = int(config['motion']['delta_frame_dilation']) 
fps = int(config['motion']['fps'])
minimum_motion_screen_percent = float(config['motion']['minimum_motion_screen_percent'])
display_contour_debug = int(config['motion']['display_contour_debug'])
motion_contours_to_consider  = int(config['motion']['motion_contours_to_consider'])

#general config
masks_directory = config['general']['masks_directory']
logfile = config['general']['logfile']
my_log_level_from_config = config['general']['log_level']

log_level_info = {  "DEBUG" : logging.DEBUG, 
                    "INFO": logging.INFO,
                    "WARNING": logging.WARNING,
                    "ERROR": logging.ERROR,
                    }
my_log_level = log_level_info[my_log_level_from_config]


logging.basicConfig(    handlers=[
								logging.FileHandler(logfile),
								logging.StreamHandler()],
						format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
						datefmt='%Y-%m-%d:%H:%M:%S',
						level=my_log_level)

logger = logging.getLogger(__name__)

logger.info("OpenCVCam started")

streams = {}

for name,uri in cameras.items():
    streams[name] = VideoStreamWidget(name, uri, gaussian_kernel_size, pixel_delta_threshold, 
                                            fps, masks_directory, minimum_motion_screen_percent,
                                                image_rescaling_factor)
while True:
    for name,uri in cameras.items():
        try:
            if display_contour_debug:
                streams[name].show_frame()
        except AttributeError:
            pass
    # sleep within framerate over all cameras (single thread)
    time.sleep(1/fps/2)