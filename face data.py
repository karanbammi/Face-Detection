import cv2
import numpy as np

#Init Camera
cap=cv2.VideoCapture(0)

# Load the haarcascade file
# Face detection

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip=0
face_data = []
dataset_path = './data/' #path where face data is going to saved

file_name = input('Enter the name of the person:  ')
while True:
    ret,frame = cap.read()

    if ret == False:
        continue
    
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    faces=sorted(faces,key=lambda f:f[2]*f[3]) #sort the face according to large area 

    face_section_list = []


    for face in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2) #rectangle arround the face

        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        face_section_list.append(face_section)


        skip+=1
        if skip%5==0:
            face_data.append(face_section)
            print(len(face_data))
    for im in face_section_list:
        cv2.imshow("Video Frame",frame)
        cv2.imshow("Face Section",face_section)



    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

np.save(dataset_path+file_name+'.npy',face_data) #saving the data in a file
print("Data succefully saved")

cap.release()
cv2.destroyAllWindows()