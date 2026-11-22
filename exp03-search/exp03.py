import numpy as np, random
from collections import deque
import matplotlib.pyplot as plt
import os
import os as _os
FIGS=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "figs"); os.makedirs(FIGS, exist_ok=True)

# carve maze
random.seed(0)
H,W = 23,35
maze = np.ones((H,W), dtype=int)
def carve(r,c):
    maze[r,c]=0
    dirs=[(-2,0),(2,0),(0,-2),(0,2)]
    random.shuffle(dirs)
    for dr,dc in dirs:
        nr,nc=r+dr,c+dc
        if 0<=nr<H and 0<=nc<W and maze[nr,nc]==1:
            maze[r+dr//2,c+dc//2]=0
            carve(nr,nc)
carve(1,1)
# ensure goal open
maze[H-2,W-2]=0
START,GOAL=(1,1),(H-2,W-2)

def neighbours(cell):
    r,c=cell
    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
        nr,nc=r+dr,c+dc
        if 0<=nr<H and 0<=nc<W:
            if maze[nr,nc]==0:
                yield (nr,nc)

def search(start,goal,mode='bfs'):
    frontier=deque([start])
    parent={start:None}
    order={}
    while frontier:
        node=frontier.popleft() if mode=='bfs' else frontier.pop()
        if node in order: continue
        order[node]=len(order)
        if node==goal: break
        for nxt in neighbours(node):
            if nxt not in parent:
                parent[nxt]=node
                frontier.append(nxt)
    path,cur=[],goal
    if goal in parent:
        while cur is not None:
            path.append(cur); cur=parent[cur]
    return order,path[::-1]

for mode in ('bfs','dfs'):
    order,path=search(START,GOAL,mode)
    print(f'{mode.upper()}: {len(order)} expanded, path length {len(path)}')

# validate
def validate(path):
    ok=True
    for i in range(len(path)-1):
        a,b=path[i],path[i+1]
        if abs(a[0]-b[0])+abs(a[1]-b[1])!=1: ok=False
        if maze[b[0],b[1]]==1: ok=False
    return ok
for mode in ('bfs','dfs'):
    order,path=search(START,GOAL,mode)
    print(mode, 'valid:', validate(path))

# figure
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,mode in zip(axes,['bfs','dfs']):
    order,path=search(START,GOAL,mode)
    grid=np.full(maze.shape,np.nan)
    for (r,c),k in order.items():
        grid[r,c]=k
    ax.imshow(maze,cmap='binary')
    ax.imshow(grid,cmap='Greys_r',alpha=0.75)
    rr=[p[0] for p in path]; cc=[p[1] for p in path]
    ax.plot(cc,rr,lw=1.5,color='k')
    ax.set_title(f'{mode.upper()}: {len(order)} expanded, path {len(path)}')
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig(f'{FIGS}/fig4_1_maze.png',dpi=150); plt.close()
print('open cells:', int((maze==0).sum()))
print('done exp3')
