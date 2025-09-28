#!/bin/bash

nohup bash ./stop.sh &> /dev/null

nohup python3 OpenCvCam.py config.txt  2>/dev/null &