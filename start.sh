#!/bin/bash

source ../hailo_examples/degirum_env/bin/activate

nohup bash ./stop.sh &> /dev/null

nohup python3 OpenCvCam.py config.txt  2>/dev/null &