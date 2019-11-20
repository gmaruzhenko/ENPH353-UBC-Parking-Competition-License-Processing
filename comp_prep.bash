#!/bin/bash
source ../devel/setup.bash #sets up path to controller workspace
args=("$@")
export ROS_MASTER_URI=10.42.0.${args[0]}:11311
export ROS_IP=10.42.0.${args[1]}
