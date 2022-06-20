import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

n = int(input())  # Number of elements which make up the association table.
q = int(input())  # Number Q of file names to be analyzed.

# dictionary containing file ext as keys and MIME as values 
file_dict = {}
filename = []

for i in range(n):
    # ext: file extension
    # mt: MIME type.
    ext, mt = input().split()
    file_dict[ext.lower()] = mt

for i in range(q):
    fname = input()  # One file name per line.
    filename.append(fname.lower())

# Write an answer using print
# To debug: print("Debug messages...", file=sys.stderr, flush=True)
'''
print('file_dict --------------------------------')
print(file_dict)
print('    ')
print('filename_ext ------------------------------')
print(filename)
'''

#[print(file_dict[name.split(".")[-1]]) if "." in name and name.split(".")[-1] in file_dict.keys() else print("UNKNOWN") for name in filename]

for name in filename:
    if "." in name and name.split(".")[-1] in file_dict.keys():
        print(file_dict[name.split(".")[-1]])
    else:
        print("UNKNOWN")


# For each of the Q filenames, display on a line the corresponding MIME type. If there is no corresponding type, then display UNKNOWN.
#print(print_this)
