from flask import Flask, render_template, request
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

   return render_template('output.html',filename = name)



		
if __name__ == '__main__':
   app.run(debug = False)