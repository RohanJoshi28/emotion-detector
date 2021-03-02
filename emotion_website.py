#imports libraries needed for classification and hosting
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, render_template, Response, url_for
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

#defines generator architecture for generating images of faces 
class Generator(nn.Module):
    #Initializing the weight matricies of the generator
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 5776)

        self.convT1 = nn.ConvTranspose2d(4, 32, 3)       
        self.convT2 = nn.ConvTranspose2d(32, 16, 3)
        self.bnorm1 = nn.BatchNorm2d(16)
        self.convT3 = nn.ConvTranspose2d(16, 8, 3)
        self.convT4 = nn.ConvTranspose2d(8, 4, 3)
        self.bnorm2 = nn.BatchNorm2d(4)
        self.convT5 = nn.ConvTranspose2d(4, 1, 3)
        
    #generating image from random noise
    def forward(self, x):
        pred = F.leaky_relu(self.fc1(x))
        pred = F.leaky_relu(self.fc2(pred))
        pred = F.leaky_relu(self.fc3(pred))
        
        pred = pred.reshape(-1, 4, 38, 38)
        
        pred = F.leaky_relu(self.convT1(pred))
        pred = F.leaky_relu(self.bnorm1(self.convT2(pred)))
        pred = F.leaky_relu(self.convT3(pred))
        pred = F.leaky_relu(self.bnorm2(self.convT4(pred)))
        pred = torch.sigmoid(self.convT5(pred))
        
        return pred
        
#initializes neural network object
emotion_net = neural_net()

#initializes generator object
generator = Generator()

#uploads emotion net weights previously trained on kaggle
state_dict = torch.load("emotion_net7.pth")

#loads the weights to the emotion net neural network and turns the neural network to evaluation mode for prediction
emotion_net.load_state_dict(state_dict)
emotion_net.eval()

#uploads generator weights previously trained on kaggle
generator_state_dict = torch.load("generator.pth")

#loads the generator weights to the generator neural network and turns the generator to evaluation mode 
generator.load_state_dict(generator_state_dict)
generator.eval()

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
#remove cache for images (so that generator can generate new images every time)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
run_with_ngrok(app)

#also helps with removing the cache of images
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

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

      </head>
      <body style = "background-color: #FFCCBC; text-align: center;">
        <h1 style = "font-family: serif; text-align: center;">Results</h1>
        <p style = "font-size: 20px; font-family: serif;">The probability you are {class_dict[order_of_max[0]]} is predicted at {round(F.softmax(emotion_arr).detach().numpy().flatten()[order_of_max[0]]/F.softmax(emotion_arr).detach().numpy().sum()*100, 2)}% confidence</p>
        <br>
        <p style = "font-size: 20px; font-family: serif;">The probability you are {class_dict[order_of_max[1]]} is predicted at {round(F.softmax(emotion_arr).detach().numpy().flatten()[order_of_max[1]]/F.softmax(emotion_arr).detach().numpy().sum()*100, 2)}% confidence</p>
        <br>
        <p style = "font-size: 20px; font-family: serif;">The probability you are {class_dict[order_of_max[2]]} is predicted at {round(F.softmax(emotion_arr).detach().numpy().flatten()[order_of_max[2]]/F.softmax(emotion_arr).detach().numpy().sum()*100, 2)}%  confidence</p>
        <br>
        <p style = "font-size: 20px; font-family: serif;">The probability you are {class_dict[order_of_max[3]]} is predicted at {round(F.softmax(emotion_arr).detach().numpy().flatten()[order_of_max[3]]/F.softmax(emotion_arr).detach().numpy().sum()*100, 2)}%  confidence</p>
        <br>
        <p style = "font-size: 20px; font-family: serif;">The probability you are {class_dict[order_of_max[4]]} is predicted at {round(F.softmax(emotion_arr).detach().numpy().flatten()[order_of_max[4]]/F.softmax(emotion_arr).detach().numpy().sum()*100, 2)}%  confidence</p>
        <br>
        <p style = "font-size: 20px; font-family: serif;">The probability you are {class_dict[order_of_max[5]]} is predicted at {round(F.softmax(emotion_arr).detach().numpy().flatten()[order_of_max[5]]/F.softmax(emotion_arr).detach().numpy().sum()*100, 2)}%  confidence</p>
        <br>
        <p style = "font-size: 20px; font-family: serif;">The probability you are {class_dict[order_of_max[6]]} is predicted at {round(F.softmax(emotion_arr).detach().numpy().flatten()[order_of_max[6]]/F.softmax(emotion_arr).detach().numpy().sum()*100, 2)}%  confidence</p>
      </body>
    </html>'''

    #renders the home template before the user uploads an image
    return render_template("index.html")

#generates a frame for video streaming
def gen():
  #turns on webcam
  video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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

#page for generating images
@app.route('/generating', methods = ['GET', 'POST'])
def generating():
  if request.method == "POST":
    #generate noise to input into the generator
    noise = torch.randn(1, 512)

    #input noise into generator
    image = generator(noise).reshape(48,48).detach().numpy()*255

    #convert image to RGB
    image = Image.fromarray(image).convert('RGB')

    #save image as png
    image.save('./static/generated_image.png', 'PNG')
    return render_template('generating.html', image = True)

  return render_template('generating.html', image = False)

#about page to show what the project is about
@app.route('/about')
def about():
  return render_template('about.html')

#creators page to show who made the website
@app.route('/creators')
def creators():
  return render_template('creators.html')

#if the file running is this file, then run the app
if __name__ == "__main__":
  app.run()