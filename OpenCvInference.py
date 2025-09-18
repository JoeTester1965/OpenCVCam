import configparser
import socket
import sys
import logging
import time
import numpy as np
import cv2
from yolo_od_utils import yolo_object_detection
from datetime import datetime
import os

def read_config(config_file):
    global config
    config = configparser.ConfigParser()
    config.read(sys.argv[1])

read_config(sys.argv[1]) 

general_config=dict(config['general']) 

motion_config=dict(config['motion']) 

inference_config=dict(config['inference-opencv']) 

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

logger.info("OpenCVCamInference started")

ipc_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

ipc_port = int(general_config['ipc_port'])
ipc_ip = general_config['ipc_ip']

ipc_socket.bind((ipc_ip, ipc_port))


# set filenames for the model
coco_names_file = inference_config['classes']
yolov3_weight_file = inference_config['weights']
yolov3_config_file = inference_config['config']
yolov3_confidence = float(inference_config['dnn_confidence'])
yolov3_threshold = float(inference_config['dnn_threshold'])

LABELS = open(coco_names_file).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
net = cv2.dnn.readNetFromDarknet(yolov3_config_file, yolov3_weight_file)

while True:
    data, addr = ipc_socket.recvfrom(1024)
    data_string = data.decode('utf-8')
    logger.info("Recieved data_string %s from %s:%d", data_string, addr[0], addr[1])
    try:
        # file name is the camera name here for brevity
        camera_name,x,y,width,height = data_string.split(',')
        retval = yolo_object_detection(camera_name, motion_config['temp_motion_directory'], net, yolov3_confidence, yolov3_threshold, LABELS, COLORS)
        
        if retval:
            blacklist = False
            whitelist = True

            if not blacklist:
                for object,confidence,box in retval:
                    logger.info("Have detected a %s with confidence %.2f at %s", object, confidence, box) 
            
            timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S-%f")
            source_path = motion_config['temp_motion_directory'] + "/" + camera_name +".jpg"
            dest_path = motion_config['detected_motion_directory'] + "/" + camera_name + "/" + timestamp +".jpg"

            if whitelist:
                os.rename(source_path, dest_path)

    except:
        logger.warning("Invalid data string")

    time.sleep(.001)




