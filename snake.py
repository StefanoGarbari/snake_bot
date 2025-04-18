import cv2
import pyautogui
import numpy as np
import dxcam
from time import sleep

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

hsvimg = cv2.cvtColor(img[y_grid : y_grid+h_grid, x_grid  : x_grid+w_grid], cv2.COLOR_RGB2HSV)

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

    if dir=='w':
        ye -= len(img)//grid_size_y
    if dir=='a':
        xe -= len(img[0])//grid_size_x
    if dir=='s':
        ye += len(img)//grid_size_y
    if dir=='d':
        xe += len(img[0])//grid_size_x

    cv2.line(img,(int(xs),int(ys)),(int(xe),int(ye)),color,thickness)

def cell_to_point(x, y):
    yr = int(len(img)*(y+0.5)//grid_size_y)
    xr = int(len(img[0])*(x+0.5)//grid_size_x)
    return (xr,yr)


def find_apple(hsvimg):
    mask = cv2.inRange(hsvimg, red_lower, red_upper)
    sums = np.array([[np.sum(square(mask, x, y)) for x in range(grid_size_x)] for y in range(grid_size_y)])
    y, x = np.unravel_index(np.argmax(sums), sums.shape)
    return (y,x)

def find_path(snake, apple):
    rtn = [[' ' for col in range(grid_size_x)] for row in range(grid_size_y)]
    if len(snake) == 0:
        return rtn

    grid = [[0 for col in range(grid_size_x)] for row in range(grid_size_y)]
    for i,s in enumerate(snake):
        grid[s[0]][s[1]] = len(snake) - i
    
    grid[snake[0][0]][snake[0][1]] = 0
    
    to_visit = []
    to_visit.append(snake[0])

    first = True

    index = 0
    while index < len(to_visit) and to_visit[index] != apple:
        y,x = to_visit[index]
        val = grid[y][x] - 1
        first = False
        if x != 0 and grid[y][x-1] >= 0 and grid[y][x-1] + val <= 0:
            grid[y][x-1] = val
            to_visit.append((y,x-1))
        if y != 0 and grid[y-1][x] >= 0 and grid[y-1][x] + val <= 0:
            grid[y-1][x] = val
            to_visit.append((y-1,x))
        if x != grid_size_x-1 and grid[y][x+1] >= 0 and grid[y][x+1] + val <= 0:
            grid[y][x+1] = val
            to_visit.append((y,x+1))
        if y != grid_size_y-1 and grid[y+1][x] >= 0 and grid[y+1][x] + val <= 0:
            grid[y+1][x] = val
            to_visit.append((y+1,x))
        
        index += 1

    grid[snake[0][0]][snake[0][1]] = 0
    
    if index >= len(to_visit):
        return rtn
    
    y,x = apple
    while (y,x) != snake[0]:
        if grid[y][x] == -1:
            if snake[0][1] == x-1:
                rtn[y][x-1] = 'd'
                x -= 1
            if snake[0][1] == x+1:
                rtn[y][x+1] = 'a'
                x += 1
            if snake[0][0] == y-1:
                rtn[y-1][x] = 's'
                y -= 1
            if snake[0][0] == y+1:
                rtn[y+1][x] = 'w'
                y += 1
        else:
            if x != 0 and grid[y][x-1] == grid[y][x] + 1:
                rtn[y][x-1] = 'd'
                x -= 1
            elif y != 0 and grid[y-1][x] == grid[y][x] + 1:
                rtn[y-1][x] = 's'
                y -= 1
            elif x != grid_size_x-1 and grid[y][x+1] == grid[y][x] + 1:
                rtn[y][x+1] = 'a'
                x += 1
            elif y != grid_size_y-1 and grid[y+1][x] == grid[y][x] + 1:
                rtn[y+1][x] = 'w'
                y += 1
            else:
                print("apple: ", apple  )
                for row in rtn:
                    for cell in row:
                        print(cell, end='.')
                    print()
                print
                for row in grid:
                    for cell in row:
                        print(cell, end='.')
                    print()
                exit()
            
    
    return rtn


    


snake = []
BASE_LENGTH = 4
score = 0
apple = find_apple(hsvimg)
path = find_path(snake, apple)


direction = 'd'

while True:
    img = camera.get_latest_frame()[y_grid:y_grid+h_grid, x_grid:x_grid+w_grid]
    
    if img[0][0][0] is not None:
        hsvimg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        bgrimg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        mask = cv2.inRange(hsvimg, head_lower, head_upper)
        sums = np.array([[np.sum(square(mask, x, y)) for x in range(grid_size_x)] for y in range(grid_size_y)])
        temp_y, temp_x = np.unravel_index(np.argmax(sums), sums.shape)

        if len(snake) == 0 or (temp_y, temp_x) not in snake[:-1]:
            snake.insert(0, (temp_y, temp_x))

        if snake[0] == apple:
            score += 1
            apple = find_apple(hsvimg)
            path = find_path(snake, apple)
        
        if len(snake) > BASE_LENGTH + score:
            snake.pop()
        
        #print(score, len(snake), len(set(snake)), len(snake) == len(set(snake)))

        if path[snake[0][0]][snake[0][1]] == ' ':
            print("off track!")
            path = find_path(snake, apple)

        if direction != path[snake[0][0]][snake[0][1]]:
            direction = path[snake[0][0]][snake[0][1]]
            pyautogui.press(direction)
        

        #output
        draw_rectangle(bgrimg,snake[0][1],snake[0][0],(255,0,0),4)

        draw_rectangle(bgrimg,apple[1],apple[0],(50,0,255),2)

        for y in range(grid_size_y):
            for x in range(grid_size_x):
                if path[y][x] in ('w','a','s','d'):
                    draw_line(bgrimg,x,y,path[y][x],(0,100,0),2)
        
        for s in snake:
            draw_rectangle(bgrimg,s[1],s[0],(255,0,255),1)


        cv2.imshow("snake", bgrimg)


    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        for row in path:
            for cell in row:
                print(cell, end=' ')
            print()
        exit()