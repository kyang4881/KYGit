import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.
# ---
# Hint: You can use the debug stream to print initialTX and initialTY, if Thor seems not follow your orders.

# light_x: the X position of the light of power
# light_y: the Y position of the light of power
# initial_tx: Thor's starting X position
# initial_ty: Thor's starting Y position
light_x, light_y, initial_tx, initial_ty = [int(i) for i in input().split()]

thor_x = initial_tx
thor_y = initial_ty
 

# game loop
while True:
    remaining_turns = int(input())  # The remaining amount of turns Thor can move. Do not remove this line.

    # Write an action using print
    print("Debug messages...", file=sys.stderr, flush=True)

    direction = ""

    if thor_y > light_y: 
        direction = 'N'
        thor_y -= 1

    if thor_y < light_y:
        direction = 'S'
        thor_y += 1

    if thor_x < light_x:
        direction = direction + 'E'
        thor_x += 1

    if thor_x > light_x:
        direction = direction + 'W'
        thor_x -=1

    # A single line providing the move to be made: N NE E SE S SW W or NW
    print(direction)

 https://www.codingame.com/ide/puzzle/power-of-thor-episode-1
