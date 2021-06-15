# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 20:29:55 2020

@author: ro
"""

ffmpeg -i output0 -i output1 -i output2 -i output3 -filter_complex "[0:v][1:v]hstack[top]; [2:v][3:v]hstack[bottom]; [top][bottom]vstack,format=yuv420p[v]; [0:a][1:a][2:a][3:a]amerge=inputs=4[a]" -map "[v]" -map "[a]" -ac 2 output.mp4

ffmpeg -i output-0.mp4 -i output-1.mp4 -i output-2.mp4 -i output-3.mp4 -filter_complex "[0:v][1:v]hstack[top]; [2:v][3:v]hstack[bottom]; [top][bottom]vstack,format=yuv420p[v]" -map "[v]"  -ac 2 output.mp4

ffmpeg -framerate 24 -i hodo%03d.png -pix_fmt yuv420p -vf pad="width=iw:height=ih+1:x=0:y=0:color=white" output.mp4