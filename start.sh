#!/bin/bash

nohup bash ./stop.sh &> /dev/null

CONFIG_FILE=".\config.txt.pc"

nohup python3 OpenCvCam.py $CONFIG_FILE  2>/dev/null &

nohup python3 OpenCvInference.py $CONFIG_FILE  2>/dev/null &

