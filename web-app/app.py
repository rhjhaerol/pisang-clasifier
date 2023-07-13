from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from keras.models import Sequential
from keras.models import load_model
from keras.layers.convolutional import Conv2D
from tensorflow.keras.layers import MaxPool2D
from keras.layers import Dense,Flatten, Dropout, BatchNormalization
import cv2
import numpy as np


modelcnn = Sequential()

# 1st layer CNN
modelcnn.add(Conv2D(filters=32, kernel_size=5, activation='relu', input_shape=[150,150,3]))
modelcnn.add(MaxPool2D(pool_size=2,padding='same'))
modelcnn.add(BatchNormalization())
modelcnn.add(Dropout(0.2))

# 2nd layer CNN
modelcnn.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
modelcnn.add(MaxPool2D(pool_size=2,padding='same'))
modelcnn.add(BatchNormalization())
modelcnn.add(Dropout(0.2))

# 3rd layer CNN
modelcnn.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
modelcnn.add(MaxPool2D(pool_size=2,padding='same'))
modelcnn.add(BatchNormalization())
modelcnn.add(Dropout(0.2))

# 4th layer CNN
modelcnn.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
modelcnn.add(MaxPool2D(pool_size=2,padding='same'))
modelcnn.add(BatchNormalization())
modelcnn.add(Dropout(0.2))

modelcnn.add(Flatten())
modelcnn.add(Dense(515,activation='relu'))
modelcnn.add(Dense(2,activation='softmax'))

modelcnn.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model = load_model('model/modelcnn.h5')

app = Flask(__name__)
app.static_folder = 'static'

class_names = ['Pisang Kulit Luka', 'Pisang Kulit Tidak Luka']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "static/upload/" + imagefile.filename
    imagefile.save(image_path)

    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (150, 150))
    img = np.expand_dims(img_resized, axis=0)
    images = np.vstack([img])

    pred = model.predict(images, batch_size=10)
    hasil = class_names[np.argmax(pred)]

    return render_template('index.html', filename=imagefile.filename, result=hasil, img="static/upload/"+imagefile.filename)

if __name__ == "__main__":
    app.run()