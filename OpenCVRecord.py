#!/usr/bin/python3
import cv2
from threading import Thread
import configparser
import time
import logging
import sys
from pathlib import Path

class camThread(Thread):
    def __init__(self, previewName, camID, recording_path):
        Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
        self.recording_path = recording_path
    def run(self):
        camPreview(self.previewName, self.camID, recording_path)

def camPreview(previewName, camID, recording_path):
    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)
    if cam.isOpened(): 
        rval, frame = cam.read()
    else:
        rval = False

    while rval:
        cv2.imshow(previewName, frame)
        rval, frame = cam.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow(previewName)

def read_config(config_file):
    global config
    config = configparser.ConfigParser()
    config.read(sys.argv[1])

read_config(sys.argv[1]) 

cameras_recording = dict(config['cameras_recording']) 
	
motion_config=dict(config['motion']) 
general_config=dict(config['general']) 

logfile = general_config['logfile']
my_log_level_from_config = general_config['log_level']
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

logger.info("OpenCVRecord started")

threads = {}

Path(cameras_recording['cameras_recording_directory']).mkdir(exist_ok=True)

for name,uri in cameras_recording.items():
    recording_path = cameras_recording['cameras_recording_directory'] + "/" + name
    Path(recording_path).mkdir(exist_ok=True)
    threads[name] = camThread(name, uri, recording_path)
    threads[name].start()

while True:
    time.sleep(.001)