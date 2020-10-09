import imutils
import cv2
import time

cam = cv2.VideoCapture("http://192.168.0.104:8080/vedio") # put the Ipwebcam address here to use mobile phone camera

time.sleep(1)
firstframe=None
area_screen = 500

while True:
    _,img = cam.read() 
    #img=cv2.flip(img,-1) # used this part to flip image if you are getting upside down image from integrated webcamera 
    output='Normal'
    img = imutils.resize(img,width=500)

    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gaussian_img=cv2.GaussianBlur(gray_img,(21,21),0)

    if firstframe is None:
        firstframe = gaussian_img
        continue
    # to find difference between "blur" and "black and white image"
    img_diff = cv2.absdiff(firstframe,gray_img)

    #set threshold to get fine result and dilate is used to remove noise from threshold image
    threshold_img=cv2.threshold(img_diff,159,255,cv2.THRESH_BINARY)[1] # <- change second parameter according to your camera
    threshold_img = cv2.dilate(threshold_img,None,iterations=2)
    
    # find and grab contours
    cnts= cv2.findContours(threshold_img.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)

    for a in cnts:
        if cv2.contourArea(a) < area_screen:
            continue
      
        (x,y,w,h) = cv2.boundingRect(a)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        output ='moving object detected'    
    print(output)    
    # to put output on screen
    cv2.putText(img,output,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    # to show final result
    cv2.imshow('Motion_Detector_camera',img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cam.release()   
cv2.destroyAllWindows()