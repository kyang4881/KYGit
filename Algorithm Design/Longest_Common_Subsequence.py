import sys

def common_ele(a, b):
    common_ele_lst = []
    n1, n2 = len(a), len(b)
    for i in range(1, n1+1):
        for j in range(1, n2+1):
            if a[i-1] == b[j-1]:
                common_ele_lst.append(a[i-1])
    return common_ele_lst

def LCMS(a,b):
    input_eles = common_ele(a, b)
    common_eles = [int(i, base=16) for i in input_eles]
    n = len(common_eles)
    lst1 = [1 for i in range(n+1)]
    for i in range(1, n):
        for j in range(0, i):
            if ((common_eles[i] > common_eles[j]) and (lst1[i] < lst1[j] +1)):
                lst1[i] = lst1[j] + 1
    lst2 = [1 for i in range(n+1)]
    for i in reversed(range(n-1)): 
        for j in reversed(range(i-1 ,n)):
            if(common_eles[i] > common_eles[j] and lst2[i] < lst2[j] + 1):
                lst2[i] = lst2[j] + 1 
    longest = lst1[0] + lst2[0] - 1
    for i in range(1, n):
        longest = max((lst1[i] + lst2[i]-1), longest)
    return longest

num_pair = int(sys.stdin.readline())
for _ in range(num_pair):
    a = sys.stdin.readline().split()
    b = sys.stdin.readline().split()
    print(LCMS(a, b))
