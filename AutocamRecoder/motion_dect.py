import cv2
import time
from datetime import datetime
import pandas

first_frame = None
status_lis = [None,None]
tim_e =[]
df = pandas.DataFrame(columns = ["Start","End"])
video = cv2.VideoCapture(0)
while True:

    check,frame = video.read()
    status =  0

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray,(21,21),0)

    if first_frame is None:
        first_frame = gray
        continue
    delta_frame = cv2.absdiff(first_frame,gray2)
    delta_threshhold = cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
    delta_threshhold1 = cv2.dilate(delta_threshhold,None,iterations=2)

    (_,cnts,_) = cv2.findContours(delta_threshhold1.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for conter in cnts:
        if cv2.contourArea(conter)< 10000:
            continue
        status =1
        (x,y,w,h) = cv2.boundingRect(conter)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    status_lis.append(status)
    if status_lis[-1]==1 and status_lis[-2]==0:
            tim_e.append(datetime.now())
    if status_lis[-1]==0 and status_lis[-2]==1:
        tim_e.append(datetime.now())

    cv2.imshow("Gray Image",gray)
    cv2.imshow("Blurry image",delta_frame)
    cv2.imshow("Thresh Image",delta_threshhold1)
    cv2.imshow("Colored Image",frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        if status ==1:
            tim_e.append(datetime.now())
        break
print(tim_e)

for i in range(0,len(tim_e),2):
    df = df.append({"Start":tim_e[i],"End":tim_e[i+1]},ignore_index=True)

df.to_csv("Time.csv")
video.release()
cv2.destroyAllWindows()
