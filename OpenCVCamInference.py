import configparser
import socket
import sys
import logging
import time

def read_config(config_file):
    global config
    config = configparser.ConfigParser()
    config.read(sys.argv[1])

read_config(sys.argv[1]) 

general_config=dict(config['general']) 

motion_config=dict(config['motion']) 

inference_config=dict(config['inference']) 

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

while True:
    data, addr = ipc_socket.recvfrom(1024)
    data_string = data.decode('utf-8')
    logger.debug("Recieved data_string %s from %s:%d", data_string, addr[0], addr[1])
    try:
        filename,x,y,width,height = data_string.split(',')
        path = motion_config['temp_motion_directory'] + "/" + filename + ".jpg"
        pass
    except:
        logger.warning("Invalid data string")

    time.sleep(.001)




