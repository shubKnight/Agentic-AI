import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import euclidean
import random
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tqdm import trange
from keras import config
import tensorflow as tf
class AbsoluteDifference(Layer):
    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])


class FaceAuthenticator:
  
    def __init__(self,train_folder,folder_name1,thershold):
        self.train_folder=train_folder
        self.folder_name1=folder_name1
        self.thershold=thershold 
        self.model=self.snn_model()
        self.embadding_modal=self.embadding()
        
       
        
        self.optimizer = tf.keras.optimizers.Adam()
    def normalize(self, embedding):
        norm = np.linalg.norm(embedding)
        if norm == 0:
         return embedding
        return embedding / norm
    def embadding(self):
       inp=Input(shape=(100,100,1))
       c1=Conv2D(64,(10,10),activation="relu",padding="same")(inp)
       m1=MaxPooling2D((2,2))(c1)
       c2=Conv2D(128,(7,7),activation="relu",padding="same")(m1)
       m2=MaxPooling2D((2,2))(c2)
       c3=Conv2D(128,(4,4),activation="relu",padding="same")(m2)
       m3=MaxPooling2D((2,2))(c3)
       c4=Conv2D(256,(4,4),activation="relu",padding="same")(m3)
       f1=Flatten()(c4)
       D=Dense(128,activation="relu")(f1)
       return Model(inp,D)
    def snn_model(self):
        base=self.embadding()
        inp1=Input(shape=(100,100,1))
        inp2=Input(shape=(100,100,1))
        emb1=base(inp1)
        emb2 =base(inp2)
        L1 =L1 = AbsoluteDifference()([emb1, emb2])

        D=Dense(1,activation="sigmoid")(L1)
  
        return Model([inp1,inp2],D)
    def process_img_array(self, img_array):
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (100, 100))
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)  # shape: (100, 100, 1)
        img = np.expand_dims(img, axis=0)   # shape: (1, 100, 100, 1)
        return img
    def process_image(self,path):
        img =cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[ERROR] Could not read image: {path}")
            return np.zeros((100, 100, 1), dtype=np.float32)
        img=cv2.resize(img,(100,100))
        img=img/255
        img=np.expand_dims(img,axis=-1)  
        #img=np.expand_dims(img,axis=0)  
        return img.astype("float32")  
    
    def create_pairs(self, num_triplets=2000):
        persons = [p for p in os.listdir(self.train_folder) if os.path.isdir(os.path.join(self.train_folder, p))]
        pairs = []
        lable=[]
        for _ in range(num_triplets):
            anchor_person = random.choice(persons)
            anchor_imgs = os.listdir(os.path.join(self.train_folder, anchor_person))
            if len(anchor_imgs) < 2: continue
            anchor_img1, positive_img2 = random.sample(anchor_imgs, 2)
            a = os.path.join(self.train_folder, anchor_person, anchor_img1)
            p = os.path.join(self.train_folder, anchor_person, positive_img2)
            pairs.append((a,p))
            lable.append(1)
            other_person=[p for p in persons if p!=anchor_person]
            if not other_person:
                  continue
            negative_person = random.choice(other_person)
            negative_person_folder=os.path.join(self.train_folder,negative_person)
            negative_person_img=os.listdir(negative_person_folder)
            if len(negative_person_img)==0:
             continue
            n1=random.choice(negative_person_img)
            n=os.path.join(negative_person_folder,n1)
            
            pairs.append((a,n))
            lable.append(0)
        print("paires creat")
            
        return pairs,lable
    def make_dataset(self,pairs,lable):
        x1=[self.process_image(p[0]) for p in pairs]
        x2=[self.process_image(p[1]) for p in pairs]
        y=np.array(lable).astype("float32")
        dataset=tf.data.Dataset.from_tensor_slices(((x1,x2),y))
        return dataset.shuffle(1024).batch(batch_size=32).prefetch(1)
    
    
    def train_model(self):
        epochs=20
        a,b=self.create_pairs()
        print(a,b)
        self.model.summary()
        dataset=self.make_dataset(a,b)
        print(dataset)
        loss_fun= BinaryCrossentropy()
        opt=Adam(1e-4)
        for epoch in trange(epochs,desc="trainig loop"):
            total_loss=0
            num_batch=0
            print(f" Epoch {epoch+1}")
            

            for(img1,img2),y in dataset:
                
                with tf.GradientTape() as tape:
                    y_prd=self.model((img1,img2),training=True)
                    loss=loss_fun(y,y_prd)
                    print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}")
                   
                gradints=tape.gradient(loss,self.model.trainable_variables)
                opt.apply_gradients(zip(gradints, self.model.trainable_variables))
                total_loss=total_loss+loss.numpy()
                num_batch+=1
        avg_loss = total_loss / num_batch
        print(f"Loss: {avg_loss:.4f}")
        self.model.save("snn_face_verification_model.keras")
        print("Model saved as snn_face_verification_model.keras")  
        
    def load_trained_model(self):
        config.enable_unsafe_deserialization()
        self.model = load_model("snn_face_verification_model.keras", compile=False)
        print("Model loaded!")      
               
    def load_face(self):
        known_face=[]
        person_name=[]
        for person in os.listdir(self.folder_name1): 
            person_path=os.path.join(self.folder_name1,person) 
            for img_file in os.listdir(person_path):
                img_path=os.path.join(person_path,img_file)
                Img=self.process_image(img_path)
                
                known_face.append(Img)
                person_name.append(person)
        return known_face,person_name
    
    def face_veri(self, Img):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
        img_final = self.process_img_array(Img)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
        for (x, y, w, h) in faces:
           face = Img[y:y + h, x:x + w]
           img_final = self.process_img_array(face)
           known_faces, labels = self.load_face()

           best_prob = 0
           predicted_label = "Unknown"

           for c, label in zip(known_faces, labels):
              c = np.expand_dims(c, axis=0) if c.shape != (1, 100, 100, 1) else c
              input_img = img_final if img_final.shape == (1, 100, 100, 1) else np.expand_dims(img_final, axis=0)

              y_pred = self.model.predict([input_img, c])[0][0]

              if y_pred > best_prob:
                best_prob = y_pred
                predicted_label = label if y_pred > self.thershold else "Unknown"

        cv2.rectangle(Img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(Img, f"{predicted_label} ({best_prob:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0), 2)
        return Img

if __name__=="__main__" :
    model_path = "snn_face_verification_model.keras"
    f=FaceAuthenticator(r"c:\Users\Dell\OneDrive\Desktop\known_face",r"c:\Users\Dell\OneDrive\Desktop\known_face1",0.29 )
    if os.path.exists(model_path):
        os.remove(model_path)
        f.load_trained_mode()  
    else:
        f.train_model()    
    
    cap=cv2.VideoCapture(0) 
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)  
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)      
    while True:
        ret,img=cap.read()
        if not ret:
            print("not open webcam")
        else:
            img_final=f.face_veri(img)    
                   
        cv2.imshow("face verification",img_final)    
        if cv2.waitKey(1) & 0xFF ==ord('q') :
            break
cap.release() 
cv2.destroyAllWindows()
