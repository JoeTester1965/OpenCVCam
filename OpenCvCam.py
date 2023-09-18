#!/usr/bin/python3
import cv2
import numpy as np;
import configparser
import sys
import logging

cameras = {}

def read_config(config_file):
    global config
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
	
    global logfile
    logfile = config['general']['logfile']
    
    global cameras
    cameras = dict(config['cameras']) 



pixel_delta_threshold = 10
gaussian_kernel_size = 13

caps = {}
frame = {}
lastFrame = {}

read_config(sys.argv[1]) 

log_level_info = {  "DEBUG" : logging.DEBUG, 
                    "INFO": logging.INFO,
                    "WARNING": logging.WARNING,
                    "ERROR": logging.ERROR,
                    }

my_log_level_from_config = config['general']['log_level']
my_log_level = log_level_info[my_log_level_from_config]

logging.basicConfig(    handlers=[
								logging.FileHandler(logfile),
								logging.StreamHandler()],
						format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
						datefmt='%Y-%m-%d:%H:%M:%S',
						level=my_log_level)

logger = logging.getLogger(__name__)

logger.info("OpenCVCam started")

for camera,camera_uri in cameras.items():
    caps[camera] = cv2.VideoCapture(camera_uri)
    retval, frame[camera] = caps[camera].read()
    lastFrame[camera] = cv2.cvtColor(frame[camera], cv2.COLOR_BGR2GRAY)
    lastFrame[camera] = cv2.GaussianBlur(lastFrame[camera], (gaussian_kernel_size, gaussian_kernel_size), 0)
  
while True:
    
    for camera,camera_uri in cameras.items():
        retval, frame[camera] = caps[camera].read()
    
    for camera,camera_uri in cameras.items():
        gray = cv2.cvtColor(frame[camera], cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (gaussian_kernel_size,gaussian_kernel_size), 0)

        frameDelta = cv2.absdiff(lastFrame[camera], gray)
        lastFrame[camera] = gray
        thresh = cv2.threshold(frameDelta, pixel_delta_threshold, 255, cv2.THRESH_BINARY)[1]

        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts[:1]:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(frame[camera],(x,y),(x+w,y+h),(0,0,255),3)

    for camera,camera_uri in cameras.items():
        cv2.imshow(camera, frame[camera])

    c = cv2.waitKey(1)