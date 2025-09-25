#!/usr/bin/python3
import cv2
import numpy as np;
import configparser
import sys
import logging
from threading import Thread,Event
import time
import os
from pathlib import Path
import shutil
from PIL import Image
from yolo_od_utils import yolo_object_detection
from datetime import datetime
import paho.mqtt.client as mqtt

class VideoStreamWidget(object):
    def __init__(self, name, uri, motion_config):
        self.name = name
        self.motion_config = motion_config

        self.last_frame = None
        self.gray = None
        self.mask = None
        self.image_pixels = None
        self.status = None
        self.frame = None
        self.uri=uri

        self.capture = cv2.VideoCapture(uri)

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            self.status, self.frame = self.capture.read()
            
            if self.status:
                if float(self.motion_config['image_rescaling_factor']) != 1.0:
                    self.frame = cv2.resize(self.frame, (0, 0), 
                        fx = float(motion_config['image_rescaling_factor']), fy = float(self.motion_config['image_rescaling_factor']), interpolation = cv2.INTER_LINEAR)
                if self.last_frame is None:
                    self.last_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                    self.last_frame = cv2.GaussianBlur(self.last_frame, (int(self.motion_config['gaussian_kernel_size']),int(self.motion_config['gaussian_kernel_size'])), 0)
                    width,height = self.last_frame.shape
                    self.image_pixels = width * height
                    masks_directory = self.motion_config['masks_directory']
                    mask_template_name = f'{masks_directory}/{self.name}/mask.jpg'
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
                        candidate_mask_template_name = f'{masks_directory}/{self.name}/mask.candidate.jpg'
                        cv2.imwrite(candidate_mask_template_name, self.frame)

                else:    

                    self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

                    if self.mask is not None:
                        self.gray = cv2.bitwise_and(self.gray,self.mask)

                    self.gray = cv2.GaussianBlur(self.gray, (int(self.motion_config['gaussian_kernel_size']),int(self.motion_config['gaussian_kernel_size'])), 0)
                    frameDelta = cv2.absdiff(self.last_frame, self.gray)
                    self.last_frame = self.gray
                    thresh = cv2.threshold(frameDelta, int(self.motion_config['pixel_delta_threshold']), 255, cv2.THRESH_BINARY)[1]
                    kernel = np.ones((int(self.motion_config['delta_frame_opening']),int(self.motion_config['delta_frame_opening'])),np.uint8)
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
                    contours = contours[0] if len(contours) == 2 else contours[1]
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)

                    if int(self.motion_config['draw_contour_debug']):
                        for contour in contours:
                            x,y,w,h = cv2.boundingRect(contour)
                            # Colour contours grey
                            cv2.rectangle(self.frame,(x,y),(x+w,y+h),(125,125,125),1)

                    biggest_contour = None
                    combined_contour = None
                    for contour in contours[:int(self.motion_config['motion_contours_to_consider']) ]:
                        x1,y1,w1,h1 = cv2.boundingRect(contour)
                        if  biggest_contour == None:
                            biggest_contour = [x1,y1,w1,h1]
                        else:
                            x2,y2,w2,h2 = biggest_contour
                            if ((abs (((x2 + w2)/2) - ((x1 + w1)/2)) < int(self.motion_config['contour_combine_distance'])) 
                                and (abs (((y2 + h2)/2) - ((y1 + h1)/2)) < int(self.motion_config['contour_combine_distance']))):
                                if combined_contour == None:
                                    # no combined contour yet, keep separate from biggest_contour ! 
                                    combined_contour = [min(x2,x1),min(y2,y1),max(w2,w1),max(h2,h1)]
                                else:
                                    x1,y1,w1,h1 = combined_contour
                                    combined_contour = [min(x2,x1),min(y2,y1),max(w2,w1),max(h2,h1)]
                            else:
                                # contours not overlappign within limits
                                pass
                    if combined_contour != None:
                        x,y,w,h = combined_contour
                        # colour combined countours white
                        if int(self.motion_config['draw_contour_debug']):
                            cv2.rectangle(self.frame,(x,y),(x+w,y+h),(255,255,255),1)
                        if (((w * h) / self.image_pixels) * 100) > float(self.motion_config['minimum_motion_screen_percent']):
                            logger.debug("%s : Potentially significant motion at %d,%d:%d,%d", self.name, x,y,w,h)
                            # colour significant objects red
                            if int(self.motion_config['draw_potentially_significant_motion_debug']):
                                cv2.rectangle(self.frame,(x,y),(x+w,y+h),(0,0,255),1)
                            logger.info("%s : Significant motion at %d,%d:%d,%d", self.name, x,y,w,h)
                            message_text = str(x) + "-" + str(y) + "-" + str(w) + "-" + str(h) +"-"
                            uri = motion_config['temp_motion_directory'] + "/" + self.name + "/" + message_text + ".jpg"
                                    
                            if writer_flag[self.name].is_set() == False:
                                writer_flag[self.name].set()
                                writer_message[self.name] = message_text
                                writer_uri[self.name] = uri
                                cv2.imwrite(uri, self.frame)
                            else:
                                logger.info("Blocked passing a frame %s for %s", uri, self.name)  
    
    def show_frame(self):
        try:
            cv2.imshow(self.name, self.frame)
            key = cv2.waitKey(1)
        except:
            pass

def read_config(config_file):
    global config
    config = configparser.ConfigParser()
    config.read(config_file)

read_config(sys.argv[1]) 

cameras = dict(config['cameras_detection']) 
	
motion_config=dict(config['motion']) 

general_config=dict(config['general']) 

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

logger.info("OpenCVCam started")

classes = None

logger.debug("DNN started")

streams = {}

writer_flag = {}

writer_message = {}

writer_uri = {}

Path(motion_config['masks_directory']).mkdir(exist_ok=True)
Path(motion_config['temp_motion_directory']).mkdir(exist_ok=True)
Path(motion_config['detected_motion_directory']).mkdir(exist_ok=True)
for name,uri in cameras.items():
    streams[name] = VideoStreamWidget(name, uri, motion_config)
    Path(motion_config['masks_directory'] + "/" + name).mkdir(exist_ok=True)
    Path(motion_config['temp_motion_directory'] + "/" + name).mkdir(exist_ok=True)
    shutil.rmtree(motion_config['temp_motion_directory'] + "/" + name)
    Path(motion_config['temp_motion_directory'] + "/" + name).mkdir(exist_ok=True)
    Path(motion_config['detected_motion_directory'] + "/" + name).mkdir(exist_ok=True)
    writer_flag[name] = Event() 

coco_names_file = inference_config['classes']
yolov3_weight_file = inference_config['weights']
yolov3_config_file = inference_config['config']
yolov3_confidence = float(inference_config['dnn_confidence'])
yolov3_threshold = float(inference_config['dnn_threshold'])

LABELS = open(coco_names_file).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
net = cv2.dnn.readNetFromDarknet(yolov3_config_file, yolov3_weight_file)

if config.has_section("mqtt"):
    mqtt_client = mqtt.Client()
    mqtt_config = config['mqtt']
    mqtt_client.username_pw_set(mqtt_config["mqtt_username"], mqtt_config["mqtt_password"])
    try:
        mqtt_client.connect(mqtt_config["mqtt_ip_address"], int(mqtt_config["mqtt_port"]))
        mqtt_client.loop_start() 
    except:
        logger.error("Cannot connect to MQTT server at %s:%s", mqtt_config["mqtt_ip_address"], mqtt_config["mqtt_port"])

while True:
    start_time = time.time()
    for camera_name,uri in cameras.items():
        try:
            if int(general_config['display_video']) :
                streams[camera_name].show_frame()
        except AttributeError:
            pass
        
        if writer_flag[camera_name].is_set() == True:
            logger.info("Processing a frame for %s", camera_name) 
            time.sleep(0.1)
            image_uri = writer_uri[camera_name]
            image = Image.open(image_uri).convert("RGB") 
            image = np.asarray(image)
            image_width, image_height, image_depth = image.shape
            x,y,width,height,ignore = writer_message[camera_name].split('-')
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
                dest_path = motion_config['detected_motion_directory'] + "/" + camera_name + "/" + timestamp +".jpg"

                if len(highest_confidence_object) > 0:
                    if config.has_section("mqtt"):
                        mqtt_config = config['mqtt']

                        x = round(((box[0] + box[2])/2)/image_width,2)
                        y = round(((box[1] + box[3])/2)/image_height,2)

                        message = camera_name + ":" + highest_confidence_object[0] + ":" + str(x) + " " + str(y)

                        mqtt_client.publish(mqtt_config["mqtt_topic"], message) 
                                
                    logger.info("%s at %s highest confidence %.3f in whitelist at %s, motion trigger %s",
                                camera_name,
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

            logger.info("Procesesed a frame for %s", camera_name) 
            writer_flag[camera_name].clear()

    # sleep well within framerate over all cameras (single thread)
    time.sleep(1/float(motion_config['fps'])/10.0)
    end_time = time.time()
    delta_time = end_time - start_time
    target_time = 1.0/float(motion_config['fps'])
    if(delta_time > target_time):
       logger.info("Not real time: main thread execution in %.3f not %.3f seconds", delta_time, target_time)
    