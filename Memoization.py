import sys

factorial_mem = {}

def factorial(n):
    """
    Calculate and store the factorial values in a dictionary
    """
    global factorial_mem
    if n not in factorial_mem:
        if n == 0 or n == 1:
            factorial_mem[n] = 1
        else:
            factorial_mem[n] = n * factorial(n-1)
    return factorial_mem[n]

derangement_mem = {0: 1, 1: 0}

def derangement(q):
    """
    Calculate and store the derangement values in a dictionary
    """
    global derangement_mem
    if q not in derangement_mem:
        if q == 0 or q == 1:
            return
        else:
            derangement_mem[q] = (q-1)*(derangement(q-1) + derangement(q-2))
    return derangement_mem[q]

def arrangement(n, p, q):
    """
    Return the maximum number of rides
    """
    global total_cnt
    global r
    global rstatic
    r = n - (p + q)
    rstatic = n - (p + q)
    total_cnt = 0
    calculate_cnt(n, p, q)
    return total_cnt
    
def calculate_cnt(n, p, q):
    """
    Calculate the maximum number of rides
    """
    global total_cnt
    global r
    global rstatic
    if r == 0:
        derang_q = derangement(q)                                                       # !q
        facto_q = factorial(q)                                                          # q!
        facto_nq = factorial(n-q)                                                       # (n-q)!
        facto_n = factorial(n)                                                          # n!
        n_C_nq = facto_n // (facto_nq * facto_q)                                        # n!//((n-q)!*(n-(n-q))!)
        total_cnt += derang_q * facto_q * facto_nq * n_C_nq
    else:
        derang_q = derangement(q + r)                                                   # !(q+r)
        facto_q = factorial(q + r)                                                      # (q+r)!
        facto_nq = factorial(n - (q + r))                                               # (n-(q+r))!
        facto_n = factorial(n)                                                          # n!
        n_C_nq = facto_n // (facto_nq * facto_q)                                        # n!//((n-(q+r))!*(n-(n-(q+r)))!)
        rstatic_C_r = factorial(rstatic) // (factorial(r) * factorial(rstatic - r))     # rstatic!//(r!*(r_static - r)!)
        total_cnt += derang_q * facto_q * facto_nq * n_C_nq * rstatic_C_r               
        r -= 1
        calculate_cnt(n, p, q)        
    
num_line = int(sys.stdin.readline())
for _ in range(num_line):
    a = [int(s) for s in sys.stdin.readline().split()]
    n, p, q = a[0], a[1], a[2]
    print(arrangement(n, p, q))
