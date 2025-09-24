import configparser
import sys
import logging
import time
import numpy as np
import cv2
from yolo_od_utils import yolo_object_detection
from datetime import datetime
import os
import paho.mqtt.client as mqtt
from PIL import Image

def read_config(config_file):
    global config
    config = configparser.ConfigParser()
    config.read(config_file)

read_config(sys.argv[1]) 

cameras = dict(config['cameras_detection']) 

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

if config.has_section("mqtt"):
    mqtt_client = mqtt.Client()
    mqtt_config = config['mqtt']
    mqtt_client.username_pw_set(mqtt_config["mqtt_username"], mqtt_config["mqtt_password"])
    try:
        mqtt_client.connect(mqtt_config["mqtt_ip_address"], int(mqtt_config["mqtt_port"]))
        mqtt_client.loop_start() 
    except:
        logger.error("Cannot connect to MQTT server at %s:%s", mqtt_config["mqtt_ip_address"], mqtt_config["mqtt_port"])


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
    
    for name,uri in cameras.items():

        image_path = motion_config['temp_motion_directory'] + "/" + name

        files = os.listdir(image_path)

        if len(files) > 1:
            logger.warning("There should only ever be one file in %s, please manually intervene", image_path)
        elif len(files) == 1:

            image_uri = image_path + "/" + files[0]

            # give OS time to write image from other process, really need IPC !
            time.sleep(0.1)

            try:
                image = Image.open(image_uri).convert("RGB") 
                image = np.asarray(image)
                image_width, image_height, image_depth = image.shape
                x,y,width,height,ignore = files[0].split('-')
                retval = yolo_object_detection(image_uri, net, yolov3_confidence, yolov3_threshold, LABELS, COLORS)
                motion_box = [int(x), int(y), int(x) + int(width), int(y) + int(height)]
            
                something_in_whitelist = []

                if retval:

                    blacklist = inference_config['blacklist']
                    whitelist = inference_config['whitelist']

                    for object,confidence,box in retval:
                        
                        in_blacklist = False
                        
                        if object in blacklist:
                            in_blacklist = True
                        if object in whitelist:
                            something_in_whitelist.append([object, confidence,box,motion_box])
                        if not in_blacklist:
                            logger.debug("%s with confidence %.2f at %s, trigger %s", object, confidence, box, motion_box) 

                    highest_confidence_object = {}         
                    
                    if len(something_in_whitelist) > 0:
                    
                        highest_confidence_found = 0.0 
                        highest_confidence_index = 0     
                        for index,object  in enumerate(something_in_whitelist):
                            if object[1] > highest_confidence_found:
                                highest_confidence_found = object[1]
                                highest_confidence_index = index

                        highest_confidence_object = something_in_whitelist[highest_confidence_index] 

                    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S-%f")
                    source_path = image_uri
                    dest_path = motion_config['detected_motion_directory'] + "/" + name + "/" + timestamp +".jpg"

                    if len(highest_confidence_object) > 0:
                        if config.has_section("mqtt"):
                            mqtt_config = config['mqtt']

                            x = round(((box[0] + box[2])/2)/image_width,2)
                            y = round(((box[1] + box[3])/2)/image_height,2)

                            message = name + ":" + highest_confidence_object[0] + ":" + str(x) + " " + str(y)

                            mqtt_client.publish(mqtt_config["mqtt_topic"], message) 
                                
                        logger.info("%s at %s highest confidence %.3f in whitelist at %s, motion trigger %s",
                                    name,
                                    highest_confidence_object[0],
                                    highest_confidence_object[1],
                                    highest_confidence_object[2],
                                    highest_confidence_object[3],)
                        os.rename(source_path, dest_path)
                    else:
                        logger.debug("Removing %s as not in whitelist", image_uri)
                        os.remove(image_uri)
                else:
                    logger.debug("Removing %s as no inference", image_uri)
                    os.remove(image_uri)

            except Exception as e:
                logger.warning(e)

    time.sleep(0.001)    




