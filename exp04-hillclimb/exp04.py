import random, numpy as np, matplotlib.pyplot as plt, os
import os as _os
FIGS=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "figs"); os.makedirs(FIGS, exist_ok=True)

# Example 1 landscape
xs=np.linspace(0,10,600)
fx=np.sin(xs)+0.6*np.sin(2.7*xs)+0.35*np.sin(5.1*xs)-0.05*(xs-5)**2
plt.figure(figsize=(9,3))
plt.plot(xs,fx,lw=1.1,color='0.1')
plt.plot(xs[np.argmax(fx)],fx.max(),'o',ms=7,mfc='none',mec='k')
plt.xlabel('x'); plt.ylabel('f(x)')
plt.title('Landscape with multiple local maxima')
plt.tight_layout(); plt.savefig(f'{FIGS}/fig5_1_landscape.png',dpi=150); plt.close()

def conflicts(board):
    n=len(board); bad=0
    for i in range(n):
        for j in range(i+1,n):
            if board[i]==board[j] or abs(board[i]-board[j])==abs(i-j):
                bad+=1
    return bad

def hill_climb(n=8, seed=0):
    r=random.Random(seed)
    board=[r.randrange(n) for _ in range(n)]
    while True:
        current=conflicts(board)
        if current==0: return True
        best,move=current,None
        for c in range(n):
            old=board[c]
            for row in range(n):
                if row==old: continue
                board[c]=row
                v=conflicts(board)
                if v<best: best,move=v,(c,row)
            board[c]=old
        if move is None: return False
        board[move[0]]=move[1]

def hill_climb_restarts(n=8,restarts=10,seed=0):
    for k in range(restarts):
        if hill_climb(n,seed=seed*1000+k):
            return True
    return False

trials=200
plain=sum(hill_climb(8,seed=s) for s in range(trials))/trials
print(f'plain hill climbing: {100*plain:.1f}% of {trials} random starts')
res={}
for k in (1,2,5,10,20):
    ok=sum(hill_climb_restarts(8,k,seed=1000+s) for s in range(trials))/trials
    res[k]=ok
    print(f'{k:>3} restarts: {100*ok:5.1f}%  predicted {100*(1-(1-plain)**k):5.1f}%')

# success rate plot
ks=list(res.keys())
plt.figure(figsize=(7,4))
plt.plot(ks,[100*res[k] for k in ks],'o-',label='measured')
plt.plot(ks,[100*(1-(1-plain)**k) for k in ks],'s--',label='predicted 1-(1-p)^k')
plt.xlabel('number of random restarts'); plt.ylabel('success rate (%)')
plt.title('Hill climbing: success rate vs restarts'); plt.legend()
plt.tight_layout(); plt.savefig(f'{FIGS}/fig5_2_restarts.png',dpi=150); plt.close()
print('done exp4')
