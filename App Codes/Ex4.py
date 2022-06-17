import sys
import math
import numpy as np

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

land_x_list = []
land_y_list = []
cord_dist_list = []
gravity = 3.711

surface_n = int(input())  # the number of points used to draw the surface of Mars.
for i in range(surface_n):
    # land_x: X coordinate of a surface point. (0 to 6999)
    # land_y: Y coordinate of a surface point. By linking all the points together in a sequential fashion, you form the surface of Mars.
    land_x, land_y = [int(j) for j in input().split()]

    land_x_list.append(land_x)
    land_y_list.append(land_y)
    cord_dist_list.append([land_x, land_y])

# game loop

# list with values T/F indicating whether the surface is flat
def check_land(land_x_list, land_y_list):

    hor_dist_list = []
    ver_dist_list = []
    slope_list = []
    avg_hor_dist_list = []
    avg_ver_dist_list = []

    for i in range(len(land_y_list)-1):
        
        hor_dist = (land_x_list[i+1] - land_x_list[i])
        ver_dist = (land_y_list[i+1] - land_y_list[i])
        slope = ver_dist/hor_dist
        avg_hor_dist = hor_dist/2
        avg_ver_dist = ver_dist/2

        hor_dist_list.append(hor_dist)
        ver_dist_list.append(ver_dist)
        avg_hor_dist_list.append(avg_hor_dist)
        avg_ver_dist_list.append(avg_ver_dist)
        slope_list.append(slope)

    return hor_dist_list, ver_dist_list, slope_list, avg_hor_dist_list, avg_ver_dist_list

def optimal_flat(flat_ground_index, x, y, avg_hor_dist_list, avg_ver_dist_list):
    
    find_opt_flat_dist = []

    for i in range(len(flat_ground_index)):

        flat_dist = np.sqrt((y - avg_ver_dist_list[flat_ground_index[i]])**2 + (x - avg_hor_dist_list[flat_ground_index[i]])**2)

        find_opt_flat_dist.append(flat_dist)

    optimal_dist = min(find_opt_flat_dist)
    optimal_dist_index = find_opt_flat_dist.index(min(find_opt_flat_dist))

    return optimal_dist, optimal_dist_index

def shorter_side(l_cord, r_cord, x, y):

    l_dist = np.sqrt((x - l_cord[0])**2 + (y - l_cord[1])**2)
    r_dist = np.sqrt((x - r_cord[0])**2 + (y - r_cord[1])**2)

    return l_cord if l_dist < r_dist else r_cord

def measure_dist_travel(x,y, final_flat_cord, h_speed):
    """measure the distance traveled"""
    
    req_dist = abs(x - final_flat_cord[0]) # distance required to final_flat_cord
    req_time = req_dist/abs(h_speed) # time required to final_flat_cord

    if x - final_flat_cord[0] > 0:
        hor_dir = 'to L'
    else:
        hor_dir = 'to R'

    return req_dist, req_time, hor_dir
    
def cal_angular_accel(rotate, power, gravity):
    """calculate the angular acceleration"""
    ang_accel = np.sqrt(power**2 + gravity**2 -2*power*gravity*math.cos(rotate))

    return ang_accel

def cal_slope(pt1, pt2):
    """calculate the slope of two points"""
    slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])

    return slope

def land_cord(x,y, cord_dist_list, slope_list, h_speed, land_x_list, land_y_list, final_flat_cord):
    """determine your current altitude difference"""
    
    final_flat_cord_index = cord_dist_list.index(final_flat_cord)
    final_flat_slope_index = slope_list.index(0)

    # kink case
    if x in land_x_list:
        lander_index_first, lander_index_second = land_x_list.index(x), land_x_list.index(x)
        
        # edge slope
        if lander_index_first == 0:
            lander_slope = cal_slope(cord_dist_list[lander_index_first], cord_dist_list[lander_index_first+1])
            
        elif lander_index_first == surface_n - 1:
            lander_slope = cal_slope(cord_dist_list[lander_index_first-1], cord_dist_list[lander_index_first])
            
        # kink slope
        else:
            lander_slope = cal_slope(cord_dist_list[lander_index_first-1], cord_dist_list[lander_index_first])

    else:
        lander_index_first = 0

        for i in range(len(land_x_list)-1):      
            if land_x_list[i] <= x and land_x_list[i+1] >= x:
                lander_index_first = i

        lander_index_second = lander_index_first + 1
        lander_slope = cal_slope(cord_dist_list[lander_index_first], cord_dist_list[lander_index_second])


    if h_speed < 0: # moving left   
            
        x1 = x # first x position
        x2 =   # need to compute
        y1 =   land_y_list[lander_index_first] # first y position
        y2 = lander_slope*(x2 - x1) + y1

    else: # moving right


    return y2

def control_movement(final_flat_cord, x, y):

    rotate_output_final = ''
    power_output_final = ''

    # move right
    if final_flat_cord[0] - x > 0:

        rotate_output_final = "-45"
        power_output_final = "4"
    # descend
    else:
        rotate_output_final = "+10"
        power_output_final = "0"

    # move left
    '''if final_flat_cord[0] < x and final_flat_cord[1] < y: 
        rotate_output = '+45'
        power_output = '0'''
    
    # ascend
    
    return rotate_output_final, power_output_final

while True:
    # h_speed: the horizontal speed (in m/s), can be negative.
    # v_speed: the vertical speed (in m/s), can be negative.
    # fuel: the quantity of remaining fuel in liters.
    # rotate: the rotation angle in degrees (-90 to 90).
    # power: the thrust power (0 to 4).
    x, y, h_speed, v_speed, fuel, rotate, power = [int(i) for i in input().split()]

    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr, flush=True)

    hor_dist_list, ver_dist_list, slope_list, avg_hor_dist_list, avg_ver_dist_list = check_land(land_x_list, land_y_list)
    
    # find the index with flat ground such that the distance is at least 1000m wide
    flat_ground_index = [i for i, value in enumerate(slope_list) if value ==0]

    # optimal flat ground
    optimal_dist, optimal_dist_index = optimal_flat(flat_ground_index, x, y, avg_hor_dist_list, avg_ver_dist_list)
    final_optimal_index = flat_ground_index[optimal_dist_index]

    # go from the shorter side
    final_flat_cord = shorter_side(l_cord = cord_dist_list[final_optimal_index], r_cord = cord_dist_list[final_optimal_index+1], x=x, y=y)

    req_dist, req_time, hor_dir = measure_dist_travel(x, y, final_flat_cord, h_speed)

    rotate_output, power_output = control_movement(final_flat_cord=final_flat_cord, x=x, y=y)

    # rotate power. rotate is the desired rotation angle. power is the desired thrust power.
    print(str(rotate_output) + " " + str(power_output))
