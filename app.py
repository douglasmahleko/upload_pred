import streamlit as st
from flask import Flask, request, render_template
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import decode_predictions
from keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
import cv2


trans = TransfereLearning()
uploaded_file = st.file_uploader("Choose a file")
if st.button("predictionResults"):
   if uploaded_file is not None:
      pred = trans.predicyFramesFromVideo(uploaded_file)
      for val in pred:
         st.success(val)
      val = st.text_input("search for predictions")
      if val:
         dt = trans.search(val)
         st.write(dt)
         st.write(trans.search(val))
      st.write('unique predicted values are \t', trans.unique_val())

class TransfereLearning:
    def __init__(self):
        self.new_pics = []
        self.pred = []
        
    def predicyFramesFromVideo(self, file):
        model = InceptionV3()
        pics = self.load_video(file)
        for image in pics:
            image = img_to_array(image)
            image = image.reshape(1, 299,299,3)
            image = preprocess_input(image)
            self.new_pics.append(image)
        predictions = []
        for image in self.new_pics:
            yhat = model.predict(image)
            predictions.append(yhat)
        encoded_predictions = []
        for yhat in predictions:
            label = decode_predictions(yhat)
            label = label[0][0]
            encoded_predictions.append(label)
        for label in encoded_predictions:
            self.pred.append(label[1])
        return self.pred
    
    def crop_center_square(self, frame):
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

    def load_video(self, file, max_frames=0, resize=(299,299)):
        cap = cv2.VideoCapture(file)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self.crop_center_square(frame)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)

                if len(frames) == max_frames:
                    break
        finally:
            cap.release()
        return np.array(frames)
    
    def search(self, search):
        count = None
        exist = False
        for i in range(len(self.pred)):
            if self.pred[i] == search:
                count = i
                exist = True
                break
        if count != None:
            plt.imshow((self.new_pics[count].reshape(299,299,3)))
        return exist
    
    def unique_val(self):
        unique_pred = []
        for val in self.pred:
            if val not in unique_pred:
                unique_pred.append(val)
        return unique_pred


if __name__ == '__main__':
    main()