import sys
import heapq

def cheapest_plan(fc, sc, tg):
    adj_list = [[(j, adj[i][j]) for j in range(len(adj[i])) if adj[i][j] != 0] for i in range(len(adj))]
    n = len(adj_list)
    dist = [[float('inf') for _ in range(fc+1)] for _ in range(n)]
    dist[sc][0] = 0
    heap = [(0, sc, 0)]

    while heap:
        d, u, f = heapq.heappop(heap)
        if d > dist[u][f]:
            continue
        for v, w in adj_list[u]:
            for i in range(fc, 3*fc//4 -1, -1):
                if i >= w and i >= f and dist[u][f] + (i - f) * fuel_price[u] < dist[v][i - w] and i-w > 0:
                    dist[v][i - w] = dist[u][f] + (i - f) * fuel_price[u]
                    heapq.heappush(heap, (dist[v][i - w], v, i - w))
    return min(dist[tg]) if min(dist[tg]) != float("inf") else 'impossible'

n, m = [int(s) for s in sys.stdin.readline().split()]
fuel_price = [int(s) for s in sys.stdin.readline().split()]
adj = [[0] * n for _ in range(n)]
for _ in range(m):
    a = [int(s) for s in sys.stdin.readline().split()]
    adj[a[0]][a[1]] = adj[a[1]][a[0]] = a[2]
q = int(sys.stdin.readline())
for _ in range(q):
    fc, sc, tg = [int(s) for s in sys.stdin.readline().split()]
    print(cheapest_plan(fc, sc, tg))
