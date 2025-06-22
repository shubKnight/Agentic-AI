import os
from deepface import DeepFace
from sklearn.neighbors import KNeighborsClassifier
import cv2
class FaceAuthenticator:
    def __init__(self,folder_name=r'c:\Users\Dell\OneDrive\Desktop\known_face',threshold=1.1):
        self.folder_name=folder_name
        self.threshold=threshold
        self.model=KNeighborsClassifier(n_neighbors=1)
        self.trained=False
        self.train_model()
       
    def train_model(self): 
        x=[]
        y=[]
        try:
              for preson_name in os.listdir(self.folder_name):
                  sub_fold_path=os.path.join(self.folder_name,preson_name)
                  print(preson_name)
                  if not os.path.isdir(sub_fold_path):
                     
                     continue  
                  for image_path in os.listdir(sub_fold_path):
                     
                     image_path=os.path.join(sub_fold_path,image_path)
                     encoding=DeepFace.represent(image_path,enforce_detection=False)
                     
                    
                     
                     if encoding:
                        x.append(encoding[0]["embedding"]) 
                        y.append(preson_name) 
                        
                     else:
                      print("image not found")
              self.model.fit(x,y)
              self.trained=True 
              print(x,y)
        except Exception as e:
         print(f"[Error] While reading known faces: {e}")      
              
               
               
        
    def face_rec(self,frame):
        if not self.trained :
            print("not train")
        
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
        frame_rgb_encoding=DeepFace.represent(frame_rgb,enforce_detection=False)
        
        for face in frame_rgb_encoding:
            enbadding=face["embedding"]
            face_area=face["facial_area"]
            clst_=self.model.kneighbors([enbadding])[0]
            if clst_[0][0]<self.threshold:
                name=self.model.predict([enbadding])[0]
                print( clst_[0][0])
                
            else:
                name="not verified"
                print( clst_[0][0])
            x,y,w,h=face_area['x'],face_area['y'],face_area['w'],face_area['h']
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) 
            cv2.putText(frame,name,(x,y - 10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
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
