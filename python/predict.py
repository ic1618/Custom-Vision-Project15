# The steps implemented in the object detection sample code:
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
"""Sample prediction script for TensorFlow SavedModel"""
import os
import sys
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
import PIL


MODEL_FILENAME = 'saved_model.pb'
LABELS_FILENAME = 'labels.txt'


class TFObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow SavedModel"""

    def __init__(self, model_filename, labels):
        super(TFObjectDetection, self).__init__(labels)
        model = tf.saved_model.load(os.path.dirname(model_filename),None)
        self.serve = model.signatures['serving_default']

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis, :, :, (2, 1, 0)]  # RGB -> BGR
        inputs = tf.convert_to_tensor(inputs)
        outputs = self.serve(inputs)
        return np.array(outputs['outputs'][0])


def main(file_path):
    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [l.strip() for l in f.readlines()]

    od_model = TFObjectDetection(MODEL_FILENAME, labels)

    detected = {
        "confidence": [],
        "filename": [],
        "offset (s)": [],
        "start": [],
        "end": [],
    }

    mylist = os.listdir(file_path)
    num_detected = 0
    pos_detected = 0
    overlap = 6 * 1000
    twelve = 12 * 1000

    # fig, ax = plt.subplots()
    # plt.figure(figsize=(3, 3))
    fig = figure.Figure()
    ax = fig.subplots(1)

    for i in range(len(mylist)):

        #print(len(mylist))
        #print(i)
        print("Loading soundfile")

        start = time.time()
        clip = AudioSegment.from_wav(file_path+"/"+mylist[i])
        end = time.time()

        #print("Took",int(end-start),"seconds to load file")



        current_s = 0

        for j in range(int(clip.duration_seconds/(overlap/1000))):
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


            #start = time.time()
            #print(S_dB.shape)
            fig.savefig(save_to)
            plt.cla()
            #plt.clf()
            plt.close(fig)
                #print(time.time()-start)

            os.remove("cache/short.wav")

            image = Image.open(save_to)

            #start = time.time()
            predictions = od_model.predict_image(image)
            hit = False


            for k in predictions:
                if(k['probability']>=0.60):
                    hit = True
                    num_detected+=1
                    print(num_detected)
                    print("GUNSHOT: ",current_s)
                    print(k)
                    detected['filename'].append(mylist[i])
                    detected['offset (s)'].append(current_s)
                    detected['confidence'].append(k['probability'])
                    detected['start'].append(str(current_s+12000*(k['boundingBox']['left'])))
                    detected['end'].append(str(current_s+12000*(k['boundingBox']['left']+k['boundingBox']['width'])))

                    image = cv2.imread(save_to)
                    top_left = (int(k['boundingBox']['left']*640),int(k['boundingBox']['top']*480))
                    bottom_right = (int((k['boundingBox']['left']+k['boundingBox']['width'])*640),int((k['boundingBox']['top']+k['boundingBox']['height'])*480))
                    print(top_left,bottom_right)
                    colour = (255,0,0)
                    thickness = 2
                    image = cv2.rectangle(image, bottom_right, top_left, colour,thickness)
                    cv2.imwrite(save_to, image)

                if(k['probability']>=0.5 and k['probability']<=0.60):
                    pos_detected+=1
                    print(k['probability'], " POSSIBLE AT ",current_s)
                #end = time.time()
                #print("Time to detect each image: {}".format(end - start))


            if not hit:
                os.remove(save_to)






            current_s = current_s+twelve
    print("Total detected:",num_detected)
    print("Possible detected:",pos_detected)

    with open("detected.json","w") as store:
        json.dump(detected, store, indent=3)
        
    df = pd.read_json("detected.json")
    df.to_csv("detected_csv.csv")
    
if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('USAGE: {} sound_path'.format(sys.argv[0]))
    else:
        main(sys.argv[1])
