import sys
import math
import string

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

l = int(input())
h = int(input())
t = input().upper()

num_char = l*27
a_to_z = list((string.ascii_lowercase).upper())
a_to_z.append('?')

char_list = []

for i in range(h):
    row = input()
    char_list.append(list(row))
    #print(list(row))
    #print('char_list')


# Write an answer using print
# To debug: print("Debug messages...", file=sys.stderr, flush=True)
'''
print(l)
print(h)
print(t)
print(row)
'''
index_list = [] # a list of the character's index

for char in list(t):
    if char in a_to_z:
        index_list.append(a_to_z.index(char))
    else:
        index_list.append(26)

combined_char = []

for i in range(h):
    for j in range(len(index_list)):
        combined_char.extend(''.join(char_list[i][index_list[j]*l : index_list[j]*l+l]))

n = int(len(combined_char)/h)
incr=0

for i in range(h):
    print(''.join(combined_char[0+n*incr:n*(incr+1)]))
    incr += 1

# l = width of letter
# h = height of letter
# t = line of text given, with N characters


'''
-T = upper case output
-non alpha => '?'

'''
