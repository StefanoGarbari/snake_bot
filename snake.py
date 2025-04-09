import cv2
import pyautogui
import numpy as np
import dxcam
import time

pyautogui.PAUSE = 0
BLUE_TO_CELL_RATIO =0.05

green_border = np.array([48,159,138])

blue_lower = np.array([110,0,0])
blue_upper = np.array([115,255,255])

head_lower = np.array([110,170,200])
head_upper = np.array([115,175,246])

red_lower = np.array([0,200,0])
red_upper = np.array([10,255,255])

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

def draw_line(img,x,y,dir,color,thickness):
    ys = len(img)*(y+0.5)//grid_size_y
    xs = len(img[0])*(x+0.5)//grid_size_x

    ye = ys
    xe = xs

    if grid[y][x]=='w':
        ye -= len(img)//grid_size_y
    if grid[y][x]=='a':
        xe -= len(img[0])//grid_size_x
    if grid[y][x]=='s':
        ye += len(img)//grid_size_y
    if grid[y][x]=='d':
        xe += len(img[0])//grid_size_x

    cv2.line(img,(int(xs),int(ys)),(int(xe),int(ye)),color,thickness)

def cell_to_point(x, y):
    yr = int(len(img)*(y+0.5)//grid_size_y)
    xr = int(len(img[0])*(x+0.5)//grid_size_x)
    return (xr,yr)

def path_to_apple(grid, head, apple):
    temp = [row[:] for row in grid]
    to_visit = []
    to_visit.append(apple)
    temp[apple[0]][apple[1]] = 'x'
    temp[head[0]][head[1]] = ' '

    index = 0
    while index < len(to_visit): #to_visit[index] != head:
        y,x = to_visit[index]
        if x != 0 and temp[y][x-1] == ' ':
            temp[y][x-1] = 'd'
            to_visit.append((y,x-1))
        if y != 0 and temp[y-1][x] == ' ':
            temp[y-1][x] = 's'
            to_visit.append((y-1,x))
        if x != grid_size_x-1 and temp[y][x+1] == ' ':
            temp[y][x+1] = 'a'
            to_visit.append((y,x+1))
        if y != grid_size_y-1 and temp[y+1][x] == ' ':
            temp[y+1][x] = 'w'
            to_visit.append((y+1,x))
        index += 1
    
    y,x = head
    while (y,x) != apple:
        grid[y][x] = temp[y][x]
        if grid[y][x] == 'w':
            y -= 1
        elif grid[y][x] == 'a':
            x -= 1
        elif grid[y][x] == 's':
            y += 1
        elif grid[y][x] == 'd':
            x += 1
        else:
            return
    




THRESHOLD = len(img)//grid_size_y * len(img[0])//grid_size_x * 255* BLUE_TO_CELL_RATIO

direction = 'd'
grid = [['x' if np.sum(square(mask,x,y)) > THRESHOLD else ' ' for x in range(grid_size_x)] for y in range(grid_size_y)]

while True:
    img = camera.get_latest_frame()[y_grid:y_grid+h_grid, x_grid:x_grid+w_grid]
    
    if img[0][0][0] is not None:
        hsvimg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        mask = cv2.inRange(hsvimg, head_lower, head_upper)
        sums = np.array([[np.sum(square(mask, x, y)) for x in range(grid_size_x)] for y in range(grid_size_y)])
        head_y, head_x = np.unravel_index(np.argmax(sums), sums.shape)
        draw_rectangle(img,head_x,head_y,(0,255,255),2)

        mask = cv2.inRange(hsvimg,red_lower, red_upper)
        sums = np.array([[np.sum(square(mask, x, y)) for x in range(grid_size_x)] for y in range(grid_size_y)])
        apple_y, apple_x = np.unravel_index(np.argmax(sums), sums.shape)
        draw_rectangle(img,apple_x,apple_y,(255,0,0),2)
        
        mask = cv2.inRange(hsvimg, blue_lower, blue_upper)
        #body = cv2.bitwise_or(hsvimg, hsvimg, mask=mask)
       
        for y in range(grid_size_y):
            for x in range(grid_size_x):
                s = square(mask,x,y)
                if np.sum(s) > THRESHOLD:
                    draw_rectangle(img,x,y,(0,0,255),1)

                if grid[y][x] in ('w','a','s','d'):
                    draw_line(img,x,y,grid[y][x],(255,255,255),2d)
                    

        #grid = [['x' if np.sum(square(mask,x,y)) > THRESHOLD else ' ' for x in range(grid_size_x)] for y in range(grid_size_y)]

        #print(grid)
        #path_to_apple(grid, (head_y, head_x), (apple_y, apple_x))
        #print(grid)
        #exit()
        
        if direction != grid[head_y][head_x]:
            if grid[head_y][head_x] == ' ' or grid[head_y][head_x] == 'x':
                print("FUORI TRACCIATO")
                print(head_x, head_y)
                grid = [['x' if np.sum(square(mask,x,y)) > THRESHOLD else ' ' for x in range(grid_size_x)] for y in range(grid_size_y)]
                path_to_apple(grid, (head_y, head_x), (apple_y, apple_x))
                #exit()
            #print(grid)
            direction = grid[head_y][head_x]
            pyautogui.press(direction)

        
        cv2.imshow("ciao", img)


    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        for raw in grid:
            for cell in raw:
                print(cell, end=' ')
            print()
        exit()