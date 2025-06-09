import os
import face_recognition
from sklearn.neighbors import KNeighborsClassifier
import cv2
class FaceAuthenticator:
    def __init__(self,folder_name='/Users/learnx/Desktop/project4',threshold=0.6):
        self.folder_name=folder_name
        self.threshold=threshold
        self.model=KNeighborsClassifier(n_neighbors=1)
        self.train_model()
        self.trained=False
    def train_model(self): 
        x=[]
        y=[]
        try:
              for preson_name in os.listdir(self.folder_name):
                  sub_fold_path=os.path.join(self.folder_name,preson_name)
                  print(preson_name)
                  if not os.path.isdir(sub_fold_path):
                     print("come")
                     continue  
                  for image_path in os.listdir(sub_fold_path):
                     print(image_path)
                     image_path=os.path.join(sub_fold_path,image_path)
                     print(image_path)
                     if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue

                     image=face_recognition.load_image_file(image_path)
                     location=face_recognition.face_locations(image)
                     encoding=face_recognition.face_encodings(image,location)
                     print(encoding)
                     if encoding:
                        x.append([encoding[0]]) 
                        y.append(preson_name) 
                     else:
                      print("image not found")
        except Exception as e:
         print(f"[Error] While reading known faces: {e}")  
         if x and y:       
          self.model.fit(x,y)
          self.trained=True
          
         else: 
             print("not found")
    def face_rec(self,frame):
        if not self.trained :
            print("not train")
        
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame_rgb_location=face_recognition.face_locations(frame_rgb)
        frame_rgb_encoding=face_recognition.face_encodings(frame_rgb,frame_rgb_location)
        
        for (top,right,bottom,left),encoding in zip(frame_rgb_location,frame_rgb_encoding):
            clst_=self.model.kneighbors([encoding])[0][0]
            if clst_[0][0]<self.threshold:
                name=self.model.predict([encoding])[0]
            else:
                name="not verified"
            cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2) 
            cv2.putText(frame,name,(left,top - 10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
        return frame       
if  __name__=="__main__" :
    recognizer= FaceAuthenticator()  
    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)   
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480) 
             
    while True:
        ret,image=cap.read()
        if not ret:
            print("not web cam open")
        else:
            image_final = recognizer.face_rec(image)   
            cv2.imshow("face recognition",image_final)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()
