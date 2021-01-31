#imports libraries needed for classification and hosting
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, render_template, Response
from flask_ngrok import run_with_ngrok
from PIL import Image
import cv2

#defines the neural network class
class neural_net(nn.Module):

    #initializes all the layers
    def __init__(self):
        super().__init__()
        
        #dropout for regularization
        self.conv_dropout = nn.Dropout(0.4)
        self.mid_dropout = nn.Dropout(0.5)
        self.lin_dropout = nn.Dropout(0.6)
        
        #convolutional blocks
        self.conv1 = nn.Conv2d(1, 64, 3, padding = 1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.b_norm1 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding = 1)
        self.b_norm2 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding = 1)
        self.b_norm3 = nn.BatchNorm2d(256)
        
        #final dense layers
        self.linear1 = nn.Linear(9216, 128)
        self.linear2 = nn.Linear(128, 7)
        
    #forward propagation through the neural network
    def forward(self, x):
        pred = F.relu(self.conv1(x.reshape(-1,1,48,48)))
        pred = F.relu(self.conv2(pred))
        pred = self.b_norm1(pred)
        pred = F.max_pool2d(pred, 2)
        pred = self.conv_dropout(pred)
        
        pred = F.relu(self.conv3(pred))
        pred = F.relu(self.conv4(pred))
        pred = self.b_norm2(pred)
        pred = F.max_pool2d(pred, 2)
        pred = self.conv_dropout(pred)
        
        pred = F.relu(self.conv5(pred))
        pred = F.relu(self.conv6(pred))
        pred = self.b_norm3(pred)
        pred = F.max_pool2d(pred, 2)
        pred = self.mid_dropout(pred)
        
        pred = pred.reshape(-1, 9216)
        
        pred = F.relu(self.linear1(pred))
        pred = self.lin_dropout(pred)
        
        pred = self.linear2(pred)

        return pred
        
        
#defines neural network object
emotion_net = neural_net()

#uploads weights previously trained on kaggle
state_dict = torch.load("C:\RJoshi\Midyear Project\emotion_net7.pth")

#loads the weights to the neural network and turns the neural network to evaluation mode
emotion_net.load_state_dict(state_dict)
emotion_net.eval()

#classification dict for the classificaiton
class_dict = {0: 'fearful',
 1: 'disgusted',
 2: 'angry',
 3: 'neutral',
 4: 'sad',
 5: 'surprised',
 6: 'happy'}

#starts running the application both locally and hosting it with ngrok
app = Flask(__name__)
run_with_ngrok(app)

#creates the home page
@app.route('/', methods = ["GET", "POST"])
def home():
    global image

    #if the user has uploaded an image, run
    if request.method == "POST":

        #gets uploaded image and classifies it
        uploaded_file = request.files['file']
        image = Image.open(uploaded_file).resize((48, 48))
        image = image.convert('L')
        emotion_arr = emotion_net(torch.Tensor(np.array(image).reshape(1,1,48,48)))
        order_of_max = np.argsort(emotion_arr.detach().numpy().flatten())[::-1]
        
        #displays results of classification softmax array
        return f'''<!doctype html>
    <html>
      <head>
        <title>File Upload</title>
        <style>
            body {background-color; rgb(200, 200, 150); text-align: center;}
            h1 {font-family; "Noto Sans Kr", sans-serif; text-align: center;}
            p {color: green; text-align: center}
        </style>
      </head>
      <body>
        <h1>Results</h1>
        <p>The probability you are <b>{class_dict[order_of_max[0]]} is predicted at {round(F.softmax(emotion_arr).detach().numpy().flatten()[order_of_max[0]]/F.softmax(emotion_arr).detach().numpy().sum()*100, 2)}%</b> confidence</p>
        <p>You probability you are <b>{class_dict[order_of_max[1]]} is predicted at {round(F.softmax(emotion_arr).detach().numpy().flatten()[order_of_max[1]]/F.softmax(emotion_arr).detach().numpy().sum()*100, 2)}%</b> confidence</p>
        <p>You probability you are <b>{class_dict[order_of_max[2]]} is predicted at {round(F.softmax(emotion_arr).detach().numpy().flatten()[order_of_max[2]]/F.softmax(emotion_arr).detach().numpy().sum()*100, 2)}%</b> confidence</p>
        <p>You probability you are <b>{class_dict[order_of_max[3]]} is predicted at {round(F.softmax(emotion_arr).detach().numpy().flatten()[order_of_max[3]]/F.softmax(emotion_arr).detach().numpy().sum()*100, 2)}%</b> confidence</p>
        <p>You probability you are <b>{class_dict[order_of_max[4]]} is predicted at {round(F.softmax(emotion_arr).detach().numpy().flatten()[order_of_max[4]]/F.softmax(emotion_arr).detach().numpy().sum()*100, 2)}%</b> confidence</p>
        <p>You probability you are <b>{class_dict[order_of_max[5]]} is predicted at {round(F.softmax(emotion_arr).detach().numpy().flatten()[order_of_max[5]]/F.softmax(emotion_arr).detach().numpy().sum()*100, 2)}%</b> confidence</p>
        <p>You probability you are <b>{class_dict[order_of_max[6]]} is predicted at {round(F.softmax(emotion_arr).detach().numpy().flatten()[order_of_max[6]]/F.softmax(emotion_arr).detach().numpy().sum()*100, 2)}%</b> confidence</p>

      </body>
    </html>'''

    #renders the home template before the user uploads an image
    return render_template("index.html")

#generates a frame for video streaming
def gen():

  #turns on webcam
  video_capture = cv2.VideoCapture(0)

  #defines opencv's built in facial detection
  faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

  #while the app is running
  while True:
      # Capture frame-by-frame
      
      #reads the next frame
      ret, frame = video_capture.read()
      
      #converts frame to grayscale and detects faces
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      faces = faceCascade.detectMultiScale(gray, 1.3, 2)

      # Draw a rectangle around the faces and writes the classification (emotion)
      for (x, y, w, h) in faces:
          cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
          crop_img = gray[y:y+h, x:x+w]
          reshaped_image = Image.fromarray(np.array(crop_img)).resize((48,48))
          tensor_image = torch.Tensor(np.array(reshaped_image).reshape(1,1,48,48))
          emotion = class_dict[int(np.argmax(emotion_net(tensor_image).detach().numpy(), axis = 1))]
          cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

      #converts the frame to jpg, then to bytes and then returns the bytes
      frame = cv2.imencode('.jpg', frame)[1].tobytes()
      yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#creates a route called video feed where the response updates the frame on the website (to send to the home page in templates)
@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/about')
def about():
  return render_template('about.html')


  
app.run()