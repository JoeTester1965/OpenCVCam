
*** To Do ***

# OpenCVCam

Based on [CudaCamz](https://github.com/JoeTester1965/CudaCamz)

# Installation

Tested on Debian 1:6.12.34-1+rpt1~bookworm (2025-06-26) aarch64 GNU/Linux

# Notes for later

only creates new cron file if not exists

if inferenece type none saves images from motion detect

disabled service, single process only but quicke n pi appartently: sudo systemctl stop hailort


```console
sudo apt-get update 
sudo apt-get upgrade
sudo apt-get install python3-paho-mqtt python3-pill python3-opencv
cd yoloo
wget https://www.kaggle.com/datasets/shivam316/yolov3-weights
cd ..
chmod u+x *.sh
./start.sh
```

Hailo

install https://github.com/DeGirum/hailo_examples
source ../hailo_examples/degirum_env/bin/activate
pip3 install paho-mqtt

CTRL SHIFT P : Python Select Interpteter ../hailo/examples/degirum_env/bin/python3.11    

# Configuration file

Edit the [config file](DrumDetector.ini) to adjust the following

| Key | Notes |
| TBD  | TBD |
| TBD  | TBD |
| TBD  | TBD |

# Starting and stopping

```console
bash start.sh
```

```console
bash stop.sh
```

# Example output

TBD

***TBD,TBD***


```console
TBD
```

# Notes

TBD

Enjoy!



