import sys

def assemble(budget, num_part, price, warranty):
    mem = {}
    def dp_fct(i, j, parts_left, min_warranty):
        if (i, j, parts_left, min_warranty) in mem:
            return mem[(i, j, parts_left, min_warranty)]
        if j == 0 and parts_left == 0:
            return min_warranty
        if i == len(price) or j <= 0 or parts_left <= 0:
            return 0
        result = 0
        for k in range(len(price[i])):
            cost = price[i][k]
            warranty_length = warranty[i][k]
            if j >= cost and warranty_length >= min_warranty:
                result = max(result, dp_fct(i+1, j-cost, parts_left-1, min_warranty))
        mem[(i, j, parts_left, min_warranty)] = result
        return result

    lower, upper = min(min(w) for w in warranty), max(max(w) for w in warranty)
    while lower < upper:
        mid = (lower + upper + 1) // 2
        if dp_fct(0, budget, num_part, mid) > 0:
            lower = mid
        else:
            upper = mid - 1
    return dp_fct(0, budget, num_part, lower)

num_case = int(sys.stdin.readline())
for _ in range(num_case):
    b, p = [int(s) for s in sys.stdin.readline().split()]
    price, warranty = [], []
    for i in range(p):
        a = [int(s) for s in sys.stdin.readline().split()]
        price.append(a[1:a[0]+1])
        warranty.append(a[a[0]+1:2*a[0]+1])
    print(assemble(b, p, price, warranty))
