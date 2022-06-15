import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

# w: width of the building.
# h: height of the building.
w, h = [int(i) for i in input().split()] # game window or grid size
n = int(input())  # maximum number of turns before game over.
x0, y0 = [int(i) for i in input().split()] # the starting position

h_min, h_max = 0, h
w_min, w_max = 0, w

# game loop
while True:
    bomb_dir = input()  # the direction of the bombs from batman's current location (U, UR, R, DR, D, DL, L or UL)

    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr, flush=True)
    
    if bomb_dir == 'U':
        h_max = y0
        w_min, w_max = x0, x0
        y0 = y0 - math.ceil((h_max - h_min)/2)
        
    if bomb_dir == 'UR':
        h_max = y0
        w_min = x0
        x0, y0 = x0 + math.ceil((w_max - w_min)/2), y0 - math.ceil((h_max - h_min)/2)        
    
    if bomb_dir == 'R':
        h_min, h_max = y0, y0
        w_min = x0
        x0 = x0 + math.ceil((w_max - w_min)/2)

    if bomb_dir == 'DR':
        h_min = y0
        w_min = x0
        x0, y0 = x0 + math.ceil((w_max - w_min)/2), y0 + math.ceil((h_max - h_min)/2)        

    if bomb_dir == 'D':
        h_min = y0
        w_min, w_max = x0, x0
        y0 = y0 + math.ceil((h_max - h_min)/2)

    if bomb_dir == 'DL':
        h_min = y0
        w_max = x0
        x0, y0 = x0 - math.ceil((w_max - w_min)/2), y0 + math.ceil((h_max - h_min)/2)     

    if bomb_dir == 'L':
        h_min, h_max = y0, y0
        w_max = x0
        x0 = x0 - math.ceil((w_max - w_min)/2)

    if bomb_dir == 'UL':
        h_max = y0
        w_max = x0
        x0, y0 = x0 - math.ceil((w_max - w_min)/2), y0 - math.ceil((h_max - h_min)/2)   
    
    # the location of the next window Batman should jump to.
    print(str(x0) + " " + str(y0))

