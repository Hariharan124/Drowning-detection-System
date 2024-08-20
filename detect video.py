
# Importing the libraries
from tensorflow.keras.layers import *
from tensorflow.keras.models import * 
from tensorflow.keras.preprocessing import image
import cv2
import base64
import urllib.request
import array
import numpy as np
import win32com.client as wincl
speak = wincl.Dispatch("SAPI.SpVoice")
from time import sleep
from pygame import mixer

mixer.init()
sound = mixer.Sound('alarm.wav')
model1 = load_model("trained_model.h5")
model1.summary()
flg=0
flg2=0
flg3=0
flg4=0
flg5=0
flg6=0
def exp(a):
    if(a==0):
        out = "Drowning"
        return out
    
    if(a==1):
        out = "Non Drowning"
        return out
    
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('video.mp4')
 
cap.set(3, 640) # set video width
cap.set(4, 480) # set video height
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
 
    # Display the resulting frame
    #faces=face_cascade.detectMultiScale(frame,1.5,8,1060)#try to tune this 6.5 and 17 parameter to get good result 
    #for(x,y,w,h) in faces:
    #    frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    #X_test=[]
    #X_test.append(frame)
    #X_test = np.array(X_test)
    cv2.imwrite("3.jpg",frame)
    img = image.load_img('3.jpg',target_size=(224,224))
    img = image.img_to_array(img, dtype='uint8')
    img = np.expand_dims(img,axis=0)   ### flattening
    
    ypred = model1.predict(img)
    ypred= ypred.argmax();
    
    font = cv2.FONT_HERSHEY_SIMPLEX       
    # Use putText() method for 
    # inserting text on video
    v=exp(ypred)
    #cv2.putText(img, v, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
    #cv2.imshow('video', img) 
    cv2.putText(
                frame, 
                str(v), 
                (100,100), 
                font, 
                1, 
                (255,255,255), 
                2
               )
    cv2.imshow('Frame',frame)
    if(v=="Drowning"):
        flg=flg+1;
        if(flg==30):
            flg=0;
            speak.Speak("Alert Drowning Detected")
            sound.play();
            url2 = "https://iotprojects20232.000webhostapp.com/drowning/update.php?st=0"
            print(url2)
            response = urllib.request.urlopen(url2).read()
            arr=response
            arr=str(arr)
            print(arr)
            
            id=""
    else:
        sound.stop();
        flg2+=1;
        if(flg2==50):
            flg2=0
            speak.Speak("Alert Drowning Detected")
            sound.play();
            url2 = "https://iotprojects20232.000webhostapp.com/drowning/update.php?st=1"
            print(url2)
            response = urllib.request.urlopen(url2).read()
            arr=response
            arr=str(arr)
            print(arr)
        id=""
            
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()
