# https://www.codingame.com/ide/puzzle/the-descent Paweł Szyszkowski s18184

import sys
import math

# The while loop represents the game.
# Each iteration represents a turn of the game
# where you are given inputs (the heights of the mountains)
# and where you have to print an output (the index of the mountain to fire on)
# The inputs you are given are automatically updated according to your last actions.


# game loop
while True:
    mountains = [];
    for i in range(8):
        mountain_h = int(input())  # represents the height of one mountain.

    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr, flush=True)

    # The index of the mountain to fire on.
        mountains.append(mountain_h);
    print(mountains.index(max(mountains)))
	
	
	
	
# https://www.codingame.com/ide/puzzle/power-of-thor-episode-1 Paweł Szyszkowski s18184

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

# game loop
while True:
    direction = ""
    remaining_turns = int(input())  # The remaining amount of turns Thor can move. Do not remove this line.

    # Write an action using print
    # To debug: print("Debug messages...", file=sys.stderr, flush=True)


    # A single line providing the move to be made: N NE E SE S SW W or NW
    if light_y > initial_ty:
        initial_ty += 1
        direction += "S"
    elif light_y < initial_ty:
        direction += "N"
        initial_ty -= 1
    if light_x > initial_tx:        
        direction += "E"
        initial_tx += 1
    elif light_x < initial_tx:
        direction += "W"
        initial_tx -= 1    
    print(direction)
	
	
# https://www.codingame.com/ide/puzzle/horse-racing-duals Paweł Szyszkowski s18184	

import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

n = int(input())
array = []
minValue = None
for i in range(n):
    pi = int(input())
    array.append(pi)
array.sort()
for i in range(len(array)-1):
    # print(array[i], file=sys.stderr, flush=True)
    if minValue == None:
        minValue = abs(array[i] - array[i+1])
    elif abs(array[i] - array[i+1]) < minValue:
        minValue = abs(array[i] - array[i+1])
# Write an answer using print
# To debug: print("Debug messages...", file=sys.stderr, flush=True)
print(minValue)
