import sys
import heapq

def project_selection(c, k):
    """
    Calculate and return the maximized capital amount
    """
    projects = []
    for n in range(len(cr)): 
        projects.append([cr[n][0], [cr[n][1], cr[n][1] - cr[n][0]]])
    
    projects.sort()
    affordable_projects = []
    n_proj = 0
    
    while n_proj < k:
        if len(projects) > 0 and projects[0][0] <= c:
            heapq.heappush(affordable_projects, -projects[0][1][1])
            heapq.heappop(projects) 
        elif len(affordable_projects) > 0:
            c += -(heapq.heappop(affordable_projects))
            n_proj += 1
        else:
            return 'impossible'
            break      
    return c
  
a = [int(s) for s in sys.stdin.readline().split()]
cr = [[int(t) for t in s.split(':')] for s in sys.stdin.readline().split()]
for _ in range(a[1]):
    b = [int(s) for s in sys.stdin.readline().split()]
    c, k = b[0], b[1]
    print(project_selection(c, k))
