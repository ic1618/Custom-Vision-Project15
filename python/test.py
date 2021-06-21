import os
import sys
from numpy.lib.npyio import save
import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection import ObjectDetection
import pandas as pd
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import figure
import time
import json
import cv2
import gc

import scipy.misc

file_path = 'sounds'

mylist = os.listdir(file_path)
num_detected = 0
overlap = 6 * 1000
twelve = 12 * 1000

# fig, ax = plt.subplots()
# plt.figure(figsize=(3, 3))
fig = figure.Figure()
ax = fig.subplots(1)

for i in range(len(mylist)):

    print(len(mylist))
    print(i)
    print("Loading soundfile")

    start = time.time()
    clip = AudioSegment.from_wav(file_path+"/"+mylist[i])
    end = time.time()

    print("Took",int(end-start),"seconds to load file")

    current_s = 0

    for j in range(int(clip.duration_seconds/(overlap/1000))):
        # start = time.time()
        fig = figure.Figure()
        ax = fig.subplots(1)

        if((current_s+twelve)/1000 > clip.duration_seconds):
            print("Breaking, clip not in range")
            break

        short = clip[current_s:current_s+twelve]

        short.export("cache/short.wav", format="wav")

        y, sr = librosa.load("cache/short.wav")

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=1500)
        
        S_dB = librosa.power_to_db(S, ref=np.max)

        
        img = librosa.display.specshow(S_dB, sr=sr,
                                fmax=1500, ax=ax)
        
        save_to = 'cache/images/'+"detected_"+mylist[i]+"_"+str((current_s/1000))+'.png'



        start = time.time() 
        #print(S_dB.shape)
        fig.savefig(save_to)
        plt.cla()
        #plt.clf()
        plt.close(fig)
      
        print(time.time()-start)

        os.remove("cache/short.wav")


        current_s = current_s+overlap