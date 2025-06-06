import cv2
import face_recognition
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

known_image1 = face_recognition.load_image_file(r"/Users/learnx/Downloads/WhatsApp Image 2025-06-06 at 20.30.46.jpeg")
face_location1 = face_recognition.face_locations(known_image1, model='hog')
face_encoding1 = face_recognition.face_encodings(known_image1, face_location1)
known_image2 = face_recognition.load_image_file(r"/Users/learnx/Downloads/WhatsApp Image 2025-06-06 at 20.30.46 (1).jpeg")
face_location2 = face_recognition.face_locations(known_image2, model='hog')
face_encoding2 = face_recognition.face_encodings(known_image2, face_location2)
known_image3 = face_recognition.load_image_file(r"/Users/learnx/Downloads/WhatsApp Image 2025-06-06 at 20.30.46 (2).jpeg")
face_location3 = face_recognition.face_locations(known_image3, model='hog')
face_encoding3 = face_recognition.face_encodings(known_image3, face_location3)
known_image4 = face_recognition.load_image_file(r"/Users/learnx/Downloads/download.jpeg")
face_location4 = face_recognition.face_locations(known_image4, model='hog')
face_encoding4 = face_recognition.face_encodings(known_image4, face_location4)
known_image5 = face_recognition.load_image_file(r"/Users/learnx/Downloads/WhatsApp Image 2025-06-05 at 17.38.04.jpeg")
face_location5 = face_recognition.face_locations(known_image5, model='hog')
face_encoding5 = face_recognition.face_encodings(known_image5, face_location5)


#if not (face_encoding1 |face_encoding2|face_encoding3|face_location4|face_encoding5):
    # print("No face found in known image")
    

face_encoding1 = face_encoding1[0]
face_encoding2 = face_encoding2[0]
face_encoding3=face_encoding3[0]
face_encoding4=face_encoding4[0]
face_encoding5=face_encoding5[0]

x = [face_encoding1,face_encoding2,face_encoding3,face_encoding5,face_encoding4]
y = ['bill gates','ronaldo','trump','obama','khan']

model = KNeighborsClassifier(n_neighbors=1)
model.fit(x, y)
def face_rec(frame,coff_):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb_location = face_recognition.face_locations(frame_rgb, model='hog')
    frame_rgb_encoding = face_recognition.face_encodings(frame_rgb, frame_rgb_location)

    if not frame_rgb_encoding:
        print("No face found in webcam frame")
        return frame  
    test_encoding = frame_rgb_encoding[0]
    for (top, right, bottom, left), test_encoding in zip(frame_rgb_location, frame_rgb_encoding):
        clst_dist, _ = model.kneighbors([test_encoding], n_neighbors=1)
        if clst_dist[0][0] > coff_:
            name = "not verified"
        else:
            name = model.predict([test_encoding])[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, image = cap.read()
    if not ret:
        print("Cannot access webcam")
        break
    

    result = face_rec(image, 0.6)
    cv2.imshow("Face Recognition", result)  

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
