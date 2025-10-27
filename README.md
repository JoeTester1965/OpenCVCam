# OpenCVCam

Monitor multiple cameras using opencv and optionally hailo AI accelerator

# Installation

Tested on Debian 1:6.12.34-1+rpt1~bookworm (2025-06-26) aarch64 GNU/Linux

First install  DegirumSDK for hailo if needed at https://github.com/DeGirum/hailo_examples

And while still in python venv used for above

```console
pip3 install paho-mqtt
```

Then install rest of what is needed here for camera processing and opencv inferenece

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

| Key | Notes |
| inference  | Contains images where have beem inference hits (cat, person etc) from each camera. |
| mask  | Motion detection masks for each camera. copy to mask.jpg then colour areas white to keep and black to mask out  |
| motion  | Contains images where motion has been detected from each camera. |
| video  | Video recorded by cron_hourly.sh srcipt which is created when running if does not already exist. |

# Configuration file config.txt

Adapt this to your own needs

*general*
| Name | Notes |
| TBD  | TBD |
| TBD  | TBD |
| TBD  | TBD |

*cameras_detection*

*recorded_video*

# Starting and stopping

```console
bash ./start.sh
```

```console
bash ./stop.sh
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
```





