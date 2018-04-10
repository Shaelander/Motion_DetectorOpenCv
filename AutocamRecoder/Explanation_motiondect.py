import cv2
import time
from datetime import datetime
import pandas

first_frame = None
status_lis = [None,None]
tim_e =[]
df = pandas.DataFrame(columns = ["Start","End"])
#TO store the first numpy array so that first value or image dose not change.
video = cv2.VideoCapture(0)
while True:

    check,frame = video.read()
    statust =  0

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#This is used to blur the image it is used to remove noise and increase accureacy
#(21,21) is width and height last parameter is 0  means black and white  it is for standrd devation if it was 3 then it would be red green blue

    gray2 = cv2.GaussianBlur(gray,(21,21),0)

    if first_frame is None:
        first_frame = gray
        continue

#cv2.absdiff used here to compare two frames
    delta_frame = cv2.absdiff(first_frame,gray2)
#threshold used to detect motion 30 is the difference in pixels and 255 is the value you want to assign
#to pixel whose values are less than 30 and 1 is for gray2 above
    delta_threshhold = cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]

    #cv2.dilate is used to smoothen the image to remove the black holes from the thresh frame
    delta_threshhold1 = cv2.dilate(delta_threshhold,None,iterations=2)

    #contors we used .copy() here because we dont want to modify the delta_threshhold1
    #retrt exteranl used to draw conter on the image
    #CHAIN_APPROX_SIMPLE is an approximation method for retriving conters
    #below code is use for eg you have two white areas but they are distinct you will get two contors
    (_,cnts,_) = cv2.findContours(delta_threshhold1.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #now filter out conters to keep the area that has more than 1000 pixel and draw rectangle around them
    for conter in cnts:
        if cv2.contourArea(conter)< 10000:
            continue
        statust =1
        (x,y,w,h) = cv2.boundingRect(conter)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    status_lis.append(statust)
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
        if statust ==1:
            tim_e.append(datetime.now())
        break
print(tim_e)

for i in range(0,len(tim_e),2):
    df = df.append({"Start":tim_e[i],"End":tim_e[i+1]},ignore_index=True)

df.to_csv("Time.csv")

video.release()
cv2.destroyAllWindows()
