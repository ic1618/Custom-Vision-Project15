# custom-vision-model
 
Custom Vision is a part of the Azure Cognitive Services provided by Microsoft which is used for training different machine learning models in order to perform object detction. This service provides us with an efficient method of labelling the sound shapes("the objects") for training models that can detect gunshot sound shapes in a spectogram.

First, we have to upload a set of images which will represent our dataset and tag them creating a bounding box around the sound shape("around the object"). Once the labelling is done, the dataset has to be trained. The model will be trained in the cloud so no coding is needed in the training process. The only feature that we have to adjust when we train the model is the time budget spent. A larger time budget allowed means better learning. However, when the model cannot be improved anymore, the training will stop even though there is still some time left.

The service provides us with some charts for model's precision and recall. By adjusting the probability threshold and the overlap threshold of our model we can see how the precision and recall evolves. This is helpful when trying to find the optimal probability threshold for detecting the gunshots.

The final model can be exported as a zip file on the local environment for personal use. The zip file contains two python files for object detection, a pb file that contains the model, a json file with metadata properties and some txt files. Some other files were added besides the standard package provided. The sound files are preprocessed inside the predict.py file and it extracts the audio files from the sounds folder and it exports labelled pictures with the identified gunshots in cache/images folder. The detected.json includes all the gunshots detected.
