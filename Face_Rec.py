import cv2
import numpy as np
import face_recognition
import os
path = 'Images'


# from PIL import ImageGrab
images = []
classNames = []
List = os.listdir(path)
#print(List)

for cl in List:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
    #print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


knownEncodeList = findEncodings(images)
print("Encoding Complete")



cap = cv2.VideoCapture(0) # 'https://192.168.254.7:8080/video' for ip camera
while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(knownEncodeList, encodeFace)
        faceDis = face_recognition.face_distance(knownEncodeList, encodeFace)
        # print(faceDis)

        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX,0.6, (255, 255, 255), 1)

    # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    # cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('frame', img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("b"):
        break

# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()