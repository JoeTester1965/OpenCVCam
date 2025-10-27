#!/usr/bin/python3
import cv2
import numpy as np;
import configparser
import sys
import logging
from threading import Thread,Event
import queue
import time
import os
from pathlib import Path
import shutil
from PIL import Image
from helpers import opencv_yolo_detection
from helpers import hailo_yolo_detection
from datetime import datetime
import paho.mqtt.client as mqtt
from multiprocessing import shared_memory
import stat
import degirum as dg
import math

class VideoStreamWidget(object):
    def __init__(self, name, uri, motion_config):
        self.name = name
        self.motion_config = motion_config

        self.camera_up = False
        self.last_frame = None
        self.gray = None
        self.mask = None
        self.image_pixels = None
        self.status = None
        self.frame = None
        self.uri=uri

        self.bring_up_camera()

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
    
    def bring_up_camera(self):

        logger.info("%s : Bringing up camera", self.name)
    
        while self.camera_up == False:
            try:
                self.capture = cv2.VideoCapture(self.uri)
                try:
                    self.status, self.frame = self.capture.read() 
                    self.camera_up = True
                    logger.info("%s : VideoCapture up and running", self.name)
                except cv2.error as e:
                    self.camera_up = False
                    logger.error("%s : VideoCapture read error : %s", self.name, e)
                    try:
                        self.capture.release()
                    except:
                        pass
                    self.sleep(float(general_config['restart_camera_timeout_seconds']))

            except cv2.error as e:
                self.camera_up = False
                logger.error("%s : VideoCapture open error : %s", self.name, e)
                self.sleep(60)

    def update(self):
        while True:
            try:
                self.status, self.frame = self.capture.read()
            except cv2.error as e:
                    self.camera_up = False
                    logger.error("%s : VideoCapture read error : %s", self.name, e)
                    self.capture.release()
                    self.bring_up_camera()
            
            if self.status:
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
                    
                    logger.info("%s : image object size %d", self.name, self.frame.size)

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
                            cv2.rectangle(self.frame,(x,y),(x+w,y+h),(125,125,125),3)

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
                            cv2.rectangle(self.frame,(x,y),(x+w,y+h),(255,255,255),3)
                        if (((w * h) / self.image_pixels) * 100) > float(self.motion_config['minimum_motion_screen_percent']):
                             # colour significant objects red
                            if int(self.motion_config['draw_potentially_significant_motion_debug']):
                                cv2.rectangle(self.frame,(x,y),(x+w,y+h),(0,0,255),3)
                            logger.debug("%s : Significant motion at %d,%d:%d,%d", self.name, x,y,w,h)
                                    
                            if writer_flag[self.name].is_set() == False:
                                writer_flag[self.name].set()
                                writer_queue[self.name].put([x,y,w,h])
                                writer_shared_memory[self.name] = self.frame
                                logger.debug("%s : Not blocked processing significant motion at %d,%d:%d,%d", self.name, x,y,w,h) 
                            else:
                                logger.debug("%s : Blocked processing significant motion at %d,%d:%d,%d", self.name, x,y,w,h)  
    
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

if not os.path.isfile(sys.argv[1]):
    print("Need a config file please")
    sys.exit()

read_config(sys.argv[1]) 
 
cameras_config = dict(config['cameras_detection']) 
	
motion_config=dict(config['motion']) 

general_config=dict(config['general']) 

recorded_video_config=dict(config['recorded_video']) 

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

writer_queue = {}

writer_shared_memory = {}

Path(motion_config['masks_directory']).mkdir(exist_ok=True)
Path(general_config['media_directory']).mkdir(exist_ok=True)
Path(general_config['media_directory'] + "/inference").mkdir(exist_ok=True)
Path(general_config['media_directory'] + "/motion").mkdir(exist_ok=True)
Path(general_config['media_directory'] + "/video").mkdir(exist_ok=True)

cron_hourly_file =  "cron_hourly.sh"

with open(cron_hourly_file, "w") as f:
    f.write("#!/bin/sh\n")

for name,uri in cameras_config.items():
    writer_shared_memory[name] = shared_memory.SharedMemory(create=True, size= int(motion_config['max_image_object_size']))
    streams[name] = VideoStreamWidget(name, uri, motion_config)
    Path(motion_config['masks_directory'] + "/" + name).mkdir(exist_ok=True)
    Path(general_config['media_directory'] + "/inference/" + name).mkdir(exist_ok=True)
    Path(general_config['media_directory'] + "/motion/" + name).mkdir(exist_ok=True)
    
    writer_flag[name] = Event() 
    writer_queue[name] = queue.Queue()
    
for name,uri in recorded_video_config.items():
    Path(general_config['media_directory'] + "/video/" + name).mkdir(exist_ok=True)
    with open(cron_hourly_file, "a") as f:
        line = "nohup ffmpeg -i '" + uri + "' -vcodec copy -t 3540 -y " + general_config['media_directory'] + "/video/" + name + "/$(date +\%Y\%m\%d\%H).mp4 > /dev/null 2>&1 < /dev/null &\n"
        f.write(line)

with open(cron_hourly_file, "a") as f:
    line = "find " + general_config['media_directory'] + " -mtime " + general_config['days_media_stored'] + " -delete"
    f.write(line)

st = os.stat(cron_hourly_file)
os.chmod(cron_hourly_file, st.st_mode | stat.S_IEXEC)

if general_config['inference_type'] == 'inference-opencv':
    inference_config=dict(config['inference-opencv']) 
    coco_names_file = inference_config['classes']
    yolov3_weight_file = inference_config['weights']
    yolov3_config_file = inference_config['config']
    yolov3_confidence = float(inference_config['yolo_confidence'])
    yolov3_iou_threshold = float(inference_config['yolo_iou_threshold'])
    LABELS = open(coco_names_file).read().strip().split("\n")
    net = cv2.dnn.readNetFromDarknet(yolov3_config_file, yolov3_weight_file)
    model_width = int(inference_config['model_width'])
    model_height = int(inference_config['model_height'])

if general_config['inference_type'] == 'inference-degirum-hailo':
    inference_config=dict(config['inference-degirum-hailo'])
    device_type = inference_config['device_type']
    inference_host_address = inference_config['inference_host_address']
    zoo_url = inference_config['zoo_url']
    token = inference_config['token']
    model_name = inference_config['model_name']
    model_width = int(inference_config['model_width'])
    model_height = int(inference_config['model_height'])
    model_confidence = float(inference_config['model_confidence'])
    
    try:
        model = dg.load_model(  model_name=model_name,
                                inference_host_address=inference_host_address,
                                zoo_url=zoo_url,
                                device_type=device_type)
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        sys.exit(1)

if config.has_section("mqtt"):
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_config = config['mqtt']
    mqtt_client.username_pw_set(mqtt_config["mqtt_username"], mqtt_config["mqtt_password"])
    try:
        mqtt_client.connect(mqtt_config["mqtt_ip_address"], int(mqtt_config["mqtt_port"]))
        mqtt_client.loop_start() 
    except:
        logger.error("Cannot connect to MQTT server at %s:%s", mqtt_config["mqtt_ip_address"], mqtt_config["mqtt_port"])

while True:
    start_time = time.time()
    for camera_name,uri in cameras_config.items():
        try:
            if int(general_config['display_video']) :
                streams[camera_name].show_frame()
        except AttributeError:
            pass
        
        if writer_flag[camera_name].is_set() == True:
            logger.debug("Processing a frame for %s", camera_name) 

            timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S-%f")
            
            x,y,width,height = writer_queue[camera_name].get()
            image = writer_shared_memory[camera_name]

            image_width, image_height, image_depth = image.shape
            motion_box = [int(x), int(y), int(x) + int(width), int(y) + int(height)]

            if (int(motion_config['draw_contour_debug']) == True) or (int(motion_config['draw_potentially_significant_motion_debug']) == True):
                logger.info("%s : motion detection at %s", camera_name, motion_box)
                dest_path = general_config['media_directory'] + "/motion/" + camera_name + "/" + timestamp +".jpg"
                cv2.imwrite(dest_path, image)
                writer_flag[camera_name].clear()
            
            retval = None

            draw_inference_boxes = int(general_config['draw_inference_boxes']) 

            if general_config['inference_type'] == 'inference-opencv':        
                retval = opencv_yolo_detection(image, net, yolov3_confidence, yolov3_iou_threshold, LABELS, model_width, model_height)

            if general_config['inference_type'] == 'inference-degirum-hailo':
                retval = hailo_yolo_detection(image, model, model_confidence)
                
            something_in_whitelist = []

            if retval:

                blacklist = inference_config['blacklist']
                whitelist = inference_config['whitelist']

                for object,confidence,box in retval:
                        
                    in_blacklist = False
                        
                    if object in blacklist:
                        in_blacklist = True
                    if object in whitelist:
                        something_in_whitelist.append([object,confidence,box,motion_box])
                    if not in_blacklist:
                        logger.debug("%s : %s confidence %.2f at %s, trigger %s", camera_name, object, confidence, box, motion_box) 

                    #
                    # Then draw boxes here and not in helper file and change opencv_yolo_detection above
                    #
                    if draw_inference_boxes:
                        pt1_x = box[0]
                        pt1_y = box[1]
                        pt2_x = box[2]
                        pt2_y = box[3]

                        cv2.rectangle(image, (pt1_x, pt1_y), (pt2_x, pt2_y), (255,0,0), 2)
                        text = str(object) + ":" + str(round(confidence,2))                        
                        (t_w, t_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, thickness=1)
                        text_offset_x = 7
                        text_offset_y = 7
                        (text_box_x1, text_box_y1) = (pt1_x, pt1_y - (t_h + text_offset_y))
                        (test_box_x2, text_box_y2) = ((pt1_x + t_w + text_offset_x), pt1_y)
                        cv2.rectangle(image, (text_box_x1, text_box_y1), (test_box_x2, text_box_y2), (100,100,100), cv2.FILLED)
                        cv2.putText(image, text, (pt1_x + text_offset_x, pt1_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

                highest_confidence_object = {}         
                    
                if len(something_in_whitelist) > 0:
                    
                    highest_confidence_found = 0.0 
                    highest_confidence_index = 0     
                    for index,object  in enumerate(something_in_whitelist):
                        if object[1] > highest_confidence_found:
                            highest_confidence_found = object[1]
                            highest_confidence_index = index

                    highest_confidence_object = something_in_whitelist[highest_confidence_index] 

                if len(highest_confidence_object) > 0:
                    if config.has_section("mqtt"):
                        mqtt_config = config['mqtt']

                        x = round(((box[0] + box[2])/2)/image_width,2)
                        y = round(((box[1] + box[3])/2)/image_height,2)

                        message = camera_name + ":" + highest_confidence_object[0] + ":" + str(x) + " " + str(y)

                        mqtt_client.publish(mqtt_config["mqtt_topic"], message) 
                                
                    logger.info("%s : %s confidence %.3f in whitelist at %s, motion trigger %s",
                                camera_name,
                                highest_confidence_object[0],
                                highest_confidence_object[1],
                                highest_confidence_object[2],
                                highest_confidence_object[3],)
                    dest_path = general_config['media_directory'] + "/inference/" + camera_name + "/" + timestamp +".jpg"
                    cv2.imwrite(dest_path, image)
            
            logger.debug("Procesesed a frame for %s", camera_name) 
            writer_flag[camera_name].clear()

    time.sleep(0.0001)
    end_time = time.time()
    delta_time = end_time - start_time
    target_time = 1.0/float(motion_config['fps'])
    if(delta_time > target_time):
       logger.info("Not real time: main thread execution in %.3f not %.3f seconds", delta_time, target_time)
    