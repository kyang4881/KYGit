import sys
import math

# The while loop represents the game.
# Each iteration represents a turn of the game
# where you are given inputs (the heights of the mountains)
# and where you have to print an output (the index of the mountain to fire on)
# The inputs you are given are automatically updated according to your last actions.



# game loop
while True:

    mountain_h_list = []
    
    for i in range(8):
        mountain_h = int(input())  # represents the height of one mountain.

        mountain_h_list.append(mountain_h)
    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr, flush=True)
        
    shoot = mountain_h_list.index(max(mountain_h_list))
    # The index of the mountain to fire on.
    print(shoot)


# shoot the highest mountain

# given 8 mountains each turn
# shoot the highest mountain by returning the index by the end of the turn

https://www.codingame.com/ide/puzzle/the-descent
