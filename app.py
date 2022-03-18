
import pandas as pd
import os
from os  import getcwd
import pickle
from flask import Flask, render_template, request
from fer import Video
from fer import FER
import sys
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')




app = Flask(__name__)

directory = getcwd()


# This function structures the HTML code for displaying the table on website
def html_code_table(vid_df,table_name,file_name,side):
    table_style = '<table style="border: 2px solid; float: ' + side + '; width: 40%;">'
    table_head = '<caption style="text-align: center; caption-side: top; font-size: 140%; font-weight: bold; color:black;"><strong>' + table_name + '</strong></caption>'
    table_head_row = '<tr><th>Product Name</th><th>Price (in Rs.)</th></tr>'
    
    html_code = table_style + table_head + table_head_row
    
    for i in range(len(vid_df.index)):
        row = '<tr><td>' + str(vid_df['Human Emotions'][i]) + '</td><td>' + str(vid_df['Emotion Value from the Video'][i]) + '</td></tr>'
        html_code = html_code + row
        
    html_code = html_code + '</table>'
    
    file_path = os.path.join(directory,'templates/')
    
    hs = open(file_path + file_name + '.html', 'w')
    hs.write(html_code)
    
    #print(html_code)

    
    

    

@app.route('/')
def upload():
   return render_template('upload.html')




	
@app.route('/uploaded', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      name = f.filename
      f.save('static/'+name)

   img = cv2.imread('static/'+name) 
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   faces = face_cascade.detectMultiScale(gray, 1.3, 5)

   for (x,y,w,h) in faces:
      cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

      roi_gray = gray[y:y+h, x:x+w]

      roi_gray = cv2.resize(roi_gray, (100, 100))

      face = roi_gray[:,:,np.newaxis]   ## converting image data into proper shape for tensorflow model

      face = np.expand_dims(face, axis=0) 
    
      expression = loaded_model.predict(face)[0]

      if expression[0] == 1:
         cv2.putText(img,'Anger/Sadness',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,)

      elif expression[1] == 1:
         cv2.putText(img,'Neutral',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,)

      elif expression[2] == 1:
         cv2.putText(img,'Happiness',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,) 
      break
   
   cv2.imwrite('static/'+name,img) 

   print('expression has been determined')
   
   def emotion_detection():

# Put in the location of the video file that has to be processed
    location_videofile = cv2.imread('static/'+name)

# Build the Face detection detector
    face_detector = FER(mtcnn=True)
# Input the video for processing
    input_video = Video(location_videofile)

# The Analyze() function will run analysis on every frame of the input video. 
# It will create a rectangular box around every image and show the emotion values next to that.
# Finally, the method will publish a new video that will have a box around the face of the human with live emotion values.
    processing_data = input_video.analyze(face_detector, display=False)

# We will now convert the analysed information into a dataframe.
# This will help us import the data as a .CSV file to perform analysis over it later
    vid_df = input_video.to_pandas(processing_data)
    vid_df = input_video.get_first_face(vid_df)
    vid_df = input_video.get_emotions(vid_df)


# We will now work on the dataframe to extract which emotion was prominent in the video
    angry = sum(vid_df.angry)
    disgust = sum(vid_df.disgust)
    fear = sum(vid_df.fear)
    happy = sum(vid_df.happy)
    sad = sum(vid_df.sad)
    surprise = sum(vid_df.surprise)
    neutral = sum(vid_df.neutral)

    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emotions_values = [angry, disgust, fear, happy, sad, surprise, neutral]

    score_comparisons = pd.DataFrame(emotions, columns = ['Human Emotions'])
    score_comparisons['Emotion Value from the Video'] = emotions_values

    html_code_table(score_comparisons ,'Emotion detection table','emotion_detection_table','left')
    

   return render_template('output.html',filename = name)



		
if __name__ == '__main__':
   app.run(debug = False)