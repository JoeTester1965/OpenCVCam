#!/usr/bin/python3
import cv2
import numpy as np;
import configparser
import sys
import logging
from threading import Thread
import time
import os
import socket

class TimeoutCheck:

	def __init__(self, seconds_to_expire):
		self._start_time = time.perf_counter()
		self._seconds_to_expire = seconds_to_expire

	def reset(self):
		if self._start_time is None:
			return None
		self._start_time = time.perf_counter()

	def expired(self):
		if self._start_time is None:
			return None
		elapsed_time = time.perf_counter() - self._start_time
		if elapsed_time > self._seconds_to_expire:
			self._start_time = time.perf_counter()
			return True
		else:
			return False

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

        self.object_detection_timer = TimeoutCheck(float(motion_config['object_detection_timer']))
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

                    if int(self.motion_config['display_contour_debug']):
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
                        # colour combined countours blue
                        if int(self.motion_config['display_contour_debug']):
                            cv2.rectangle(self.frame,(x,y),(x+w,y+h),(255,0,0),3)
                        if (((w * h) / self.image_pixels) * 100) > float(self.motion_config['minimum_motion_screen_percent']):
                            logger.debug("%s : Potentially significant object at %d,%d:%d,%d", self.name, x,y,w,h)
                            # colour significant contours red
                            if int(self.motion_config['display_potentially_significant_object_debug']):
                                cv2.rectangle(self.frame,(x,y),(x+w,y+h),(0,255,255),3)
                            if(self.object_detection_timer.expired()):
                                # IPC event for accelerated object detection
                                logger.info("%s : Significant object at %d,%d:%d,%d", self.name, x,y,w,h)
                                filename = motion_config['temp_motion_directory'] + "/" + self.name + ".jpg"
                                cv2.imwrite(filename, self.frame)
                                ipc_message = self.name + "," + str(x) + "," + str(y) + "," + str(w) + "," + str(h)
                                ipc_socket.sendto(bytes(ipc_message, "utf-8"), (ipc_ip, int(ipc_port)))
                                pass

            time.sleep(1/float(motion_config['fps'])/float(general_config['sleep_ratio']))
    
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

cameras = dict(config['cameras']) 
	
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

logger.info("OpenCVCam started")

classes = None

logger.debug("DNN started")

streams = {}

ipc_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

ipc_port = int(general_config['ipc_port'])
ipc_ip = general_config['ipc_ip']

for name,uri in cameras.items():
    streams[name] = VideoStreamWidget(name, uri, motion_config)

while True:
    start_time = time.time()
    for name,uri in cameras.items():
        try:
            # code below not consistent with if int(dnn_config['display_object_detect_debug']): above!!
            if int(motion_config['display_contour_debug']) or int(motion_config['display_potentially_significant_object_debug']) or int(general_config['display_raw_video']) :
                streams[name].show_frame()
        except AttributeError:
            pass

    # sleep well within framerate over all cameras (single thread)
    time.sleep(1/float(motion_config['fps'])/float(general_config['sleep_ratio']))
    end_time = time.time()
    delta_time = end_time - start_time
    target_time = 1.0/float(motion_config['fps'])
    if(delta_time > target_time):
       logger.info("Not real time: execution in %.3f not %.3f seconds", delta_time, target_time)
    