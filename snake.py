import cv2
import pyautogui
import numpy as np
import dxcam
import time

pyautogui.PAUSE = 0

green_border = np.array([48,159,138])

blue_lower = np.array([110,0,0])
blue_upper = np.array([115,255,255])

head_lower = np.array([110,170,200])
head_upper = np.array([115,175,246])

camera = dxcam.create()
camera.start(target_fps=120)

img = camera.get_latest_frame()
    
while img[0][0][0] is None:
    img = camera.get_latest_frame()

mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_RGB2HSV), green_border, green_border)
contours, h_grid = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

x_grid,y_grid,w_grid,h_grid = cv2.boundingRect([c for i,c in enumerate(contours) if h_grid[0][i][3] != -1][0])
x_grid += 2
y_grid += 2
w_grid -= 4
h_grid -= 4

#find out grid size
light_lower = np.array([35, 0, 200])
light_upper = np.array([65, 165, 255])

line = img[y_grid : y_grid+1, x_grid  : x_grid+w_grid]
mask = cv2.inRange(cv2.cvtColor(line, cv2.COLOR_RGB2HSV), light_lower, light_upper)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
light = len(contours)
mask = 255 - mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
dark = len(contours)
grid_size_x = light + dark

line = img[y_grid : y_grid+h_grid, x_grid  : x_grid+1]
mask = cv2.inRange(cv2.cvtColor(line, cv2.COLOR_RGB2HSV), light_lower, light_upper)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
light = len(contours)
mask = 255 - mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
dark = len(contours)
grid_size_y = light + dark

print(grid_size_x, grid_size_y)

def square(grid,x,y):
    ys = len(grid)*y//grid_size_y
    ye = len(grid)*(y+1)//grid_size_y
    xs = len(grid[0])*x//grid_size_x
    xe = len(grid[0])*(x+1)//grid_size_x
    return grid[ys:ye, xs:xe]

def draw_rectangle(img,x,y,color,thickness):
    ys = len(img)*y//grid_size_y
    ye = len(img)*(y+1)//grid_size_y
    xs = len(img[0])*x//grid_size_x
    xe = len(img[0])*(x+1)//grid_size_x
    cv2.rectangle(img,(xs,ys),(xe,ye),color,thickness)

def cell_to_point(x, y):
    yr = int(len(img)*(y+0.5)//grid_size_y)
    xr = int(len(img[0])*(x+0.5)//grid_size_x)
    return (xr,yr)

while True:
    img = camera.get_latest_frame()[y_grid:y_grid+h_grid, x_grid:x_grid+w_grid]
    
    if img[0][0][0] is not None:
        hsvimg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        mask = cv2.inRange(hsvimg, head_lower, head_upper)

        sums = np.array([[np.sum(square(mask, x, y)) for x in range(grid_size_x)] for y in range(grid_size_y)])
        head_y, head_x = np.unravel_index(np.argmax(sums), sums.shape)
        draw_rectangle(img,head_x,head_y,(0,255,255),2)


        mask = cv2.inRange(hsvimg, blue_lower, blue_upper)
        curr_x = head_x
        curr_y = head_y
        direction = ""

        while True:
            print(curr_x, curr_y)
            sq = square(mask,curr_x,curr_y)
            up = np.sum(sq[0, :]) if direction != "down" else 0
            left = np.sum(sq[:, 0]) if direction != "right" else 0
            down = np.sum(sq[-1, :]) if direction != "up" else 0
            right = np.sum(sq[:, -1]) if direction != "left" else 0

            max_value = max(up, left, down, right)
            if max_value == 0:
                break

            if max_value == up:
                cv2.line(img, cell_to_point(curr_x,curr_y), cell_to_point(curr_x, curr_y-1),(255,255,255), 2)
                direction = "up"
                curr_y -= 1
            elif max_value == left:
                cv2.line(img, cell_to_point(curr_x,curr_y), cell_to_point(curr_x-1, curr_y),(255,255,255), 2)
                direction = "left"
                curr_x -= 1
            elif max_value == down:
                cv2.line(img, cell_to_point(curr_x,curr_y), cell_to_point(curr_x, curr_y+1),(255,255,255), 2)
                direction = "down"
                curr_y += 1
            elif max_value == right:
                cv2.line(img, cell_to_point(curr_x,curr_y), cell_to_point(curr_x+1, curr_y),(255,255,255), 2)
                direction = "right"
                curr_x += 1
            else:
                break



        for y in range(grid_size_y):
            for x in range(grid_size_x):
                s = square(mask,x,y)
                if np.sum(s) > 400000:
                    draw_rectangle(img,x,y,(0,0,255),1)
                    #if s[len(img)//grid_size_y//2, 0] != 0:
                    #    cv2.line(img, cell_to_point(x,y), cell_to_point(x-1, y),(255,255,255), 2)
        
        grid = [['s' if np.sum(square(mask,x,y)) > 340000 else ' ' for x in range(grid_size_x)] for y in range(grid_size_y)]



        """if(head_x == 9 and head_y == 4):
            pyautogui.press('w')
        if(head_x == 9 and head_y == 0):
            pyautogui.press('a')
        if(head_x == 0 and head_y == 0):
            pyautogui.press('s')
        if(head_x == 0 and head_y == 4):
            pyautogui.press('d')"""


        cv2.imshow("ciao", img)


    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        exit()