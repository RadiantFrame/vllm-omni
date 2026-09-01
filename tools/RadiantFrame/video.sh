#!/bin/bash

# 分辨率
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,nb_frames,r_frame_rate,duration \
  -of default=nokey=0:noprint_wrappers=1 video.mp4