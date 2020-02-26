import cv2
import numpy as np

video=cv2.VideoCapture("video.mp4")
insan_bul=cv2.CascadeClassifier("haarcascade_fullbody.xml")

while True:
    ret,kare=video.read()
    gri_ton=cv2.cvtColor(kare,cv2.COLOR_BGR2GRAY)
    
    beden=insan_bul.detectMultiScale(gri_ton,1.3,3)
        
    for (x,y,w,h) in beden:
        cv2.rectangle(kare,(x,y),(x+w,y+h),(255,0,0),3)
        
    cv2.imshow("Kare",kare)
    
    if cv2.waitKey(5) & 0xFF==ord('q'):
        break
video.release()
cv2.destroyAllWindows()

        
    
