# OpenCVCam

Record multiple cameras and events using ffmpeg and opencv with motion and optionally hailo AI accelerator for object detection.

My Pi5 and Hailo-8L is sailing along with five cameras operational at home.

# Installation

Tested on Debian 1:6.12.34-1+rpt1~bookworm (2025-06-26) aarch64 GNU/Linux

First install  DegirumSDK for hailo if needed at https://github.com/DeGirum/hailo_examples

And while still in python venv used for above

```console
pip3 install paho-mqtt
```

Then install rest of what is needed here for camera processing and opencv inference

```console
sudo apt-get update 
sudo apt-get upgrade
sudo apt-get install python3-paho-mqtt python3-pill python3-opencv
cd yoloo
wget https://data.pjreddie.com/files/yolov3.weights
cd ..
chmod u+x *.sh
```

# Output directory structure

| key | notes |
| --------- |:-------|
| inference  | Contains images where have beem inference hits (cat, person etc) from each camera. |
| mask  | Motion detection masks for each camera. copy to mask.jpg then colour areas white to keep and black to mask out  |
| motion  | Contains images where motion has been detected from each camera. |
| video  | Video recorded by cron_hourly.sh srcipt which is created when running if does not already exist. |

# Configuration file config.txt

Adapt this to your own needs

*general*
| Name | Notes |
| --------- |:-------|
| Self explanatory | See example file |

*cameras_detection*
| Name | Notes |
| --------- |:-------|
| Self explanatory  | See example file |

*recorded_video*
| Name | Notes |
| --------- |:-------|
| Self explanatory | See example file |

*motion*
| Name | Notes |
| --------- |:-------|
| Mostly self explanatory up to max_image_object_size | See example file, leave rest alone, took time to tune. |
| max_image_object_size | IMPORTANT : set this to maximum image object size seen in logfile output when cameras starting up (-; |

*inference-opencv*
| Name | Notes |
| --------- |:-------|
| Mostly self explanatory | Blacklisted detections are totally ignored and whitelisted detections can be saved to file via save_inference_whitelist_images and also a message sent to MQTT server |

*inference-degirum-hailo*
| Name | Notes |
| --------- |:-------|
| Mostly self explanatory  | Ditto as above |

*mqtt*
| Name | Notes |
| --------- |:-------|
| Self explanatory | See example file |

Will notify of highest confidence whitelisted event in a camera frame, with center of object x and y as a fraction of frame size e.g.:

```console
test_video:person:0.59 0.62
```

# Starting and stopping

```console
bash ./start.sh
```

```console
bash ./stop.sh
```

Note: If using hailo, you will need to edit *start.sh* and add **source** to hailo env e.g.

```console
#!/bin/bash

source ../hailo_examples/degirum_env/bin/activate

nohup bash ./stop.sh &> /dev/null

nohup python3 OpenCvCam.py config.txt  2>/dev/null &
```

# Recording video

If cron_hourly.sh does not exist in your installation folder, then OpenCVCam will create for you.

Put an entry in crontab to run this e.g.

```console
@reboot sleep 300 && cd /home/pi/Documents/OpenCVCam && ./start.sh
0 * * * * cd /home/pi/Documents/OpenCVCam && ./cron_hourly.sh
```

# Notes

If only using this for hailo AI accelerator then disable hailo sharing service for better performance:

```console
sudo service hailort stop
sudo systemctl mask hailort
```





