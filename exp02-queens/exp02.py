import time, random
from itertools import permutations
import matplotlib.pyplot as plt
import os
import os as _os
FIGS=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "figs"); os.makedirs(FIGS, exist_ok=True)

def conflicts(board):
    n=len(board); bad=0
    for i in range(n):
        for j in range(i+1,n):
            if board[i]==board[j] or abs(board[i]-board[j])==abs(i-j):
                bad+=1
    return bad

t0=time.perf_counter()
sols=[p for p in permutations(range(8)) if conflicts(p)==0]
print(len(sols),'solutions in', round(time.perf_counter()-t0,4),'s')
print('first:', sols[0])

def is_safe(board,col,row):
    for c in range(col):
        if board[c]==row or abs(board[c]-row)==abs(c-col):
            return False
    return True

def solve(n=8):
    sols,nodes=[], [0]
    def place(board,col):
        if col==n:
            sols.append(tuple(board)); return
        for row in range(n):
            nodes[0]+=1
            if is_safe(board,col,row):
                board[col]=row
                place(board,col+1)
                board[col]=-1
    place([-1]*n,0)
    return sols,nodes[0]

sols,nodes=solve(8)
print(len(sols),'solutions,',nodes,'nodes visited')

# scaling table
rows=[]
for n in range(4,11):
    t0=time.perf_counter(); s,nd=solve(n); tb=time.perf_counter()-t0
    t0=time.perf_counter()
    _=[p for p in permutations(range(n)) if conflicts(p)==0]
    tg=time.perf_counter()-t0
    rows.append((n,len(s),nd,tb,tg))
    print(f'{n:>3} {len(s):>10} {nd:>10} {tb:>10.4f}s {tg:>10.4f}s')

# plot run time
ns=[r[0] for r in rows]
tb=[r[3] for r in rows]
tg=[r[4] for r in rows]
plt.figure(figsize=(7,4))
plt.semilogy(ns,tg,'o-',label='generate and test')
plt.semilogy(ns,tb,'s-',label='backtracking')
plt.xlabel('board size n'); plt.ylabel('run time (s, log scale)'); plt.legend()
plt.title('Eight queens: run time vs board size')
plt.tight_layout(); plt.savefig(f'{FIGS}/fig3_1_scaling.png',dpi=150); plt.close()

# board plot first solution
board=sols[0]
plt.figure(figsize=(4,4))
for c in range(8):
    plt.plot(board[c], c, 'ks', ms=12)
plt.gca().invert_yaxis()
plt.xticks(range(8)); plt.yticks(range(8))
plt.xlabel('row'); plt.ylabel('column')
plt.title('First solution, n=8')
plt.grid(True,ls=':',alpha=0.5)
plt.tight_layout(); plt.savefig(f'{FIGS}/fig3_2_board.png',dpi=150); plt.close()
print('done exp2')
