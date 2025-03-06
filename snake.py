import cv2
import pyautogui
import numpy as np
import dxcam
import time

green = np.array([48,159,138])
greenupper = np.array([48,159,138])

blue_lower = np.array([110,0,0])
blue_upper = np.array([115,255,255])

head_lower = np.array([110,170,200])
head_upper = np.array([115,175,246])

camera = dxcam.create()
camera.start(target_fps=120) 
while True:
    img = camera.get_latest_frame() # np.array(pyautogui.screenshot())
    
    if img[0][0][0] is not None:
        mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_RGB2HSV), green, green)
        contours, h = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

        x,y,w,h = cv2.boundingRect([c for i,c in enumerate(contours) if h[0][i][3] != -1][0])
        
        #print(x,y,w,h)
        #cv2.imshow("ciao", cv2.resize(mask,(400,400)))
        
        mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_RGB2HSV), blue_lower, blue_upper)
        cv2.imshow("ciao",mask[y:y+h, x:x+w])


    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        exit()