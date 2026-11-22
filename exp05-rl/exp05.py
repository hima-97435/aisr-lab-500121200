import numpy as np, matplotlib.pyplot as plt, os
import os as _os
FIGS=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "figs"); os.makedirs(FIGS, exist_ok=True)

H,W=5,7
WALLS={(1,2),(2,2),(3,2),(1,4),(2,4)}
GOAL,PIT,START=(0,W-1),(2,W-1),(H-1,0)
A=[(-1,0),(1,0),(0,-1),(0,1)]
def step(s,a):
    r,c=s; dr,dc=A[a]; nr,nc=r+dr,c+dc
    if not (0<=nr<H and 0<=nc<W) or (nr,nc) in WALLS:
        nr,nc=r,c
    ns=(nr,nc)
    if ns==GOAL: return ns,+10.0,True
    if ns==PIT: return ns,-10.0,True
    return ns,-0.1,False

def train(eps,episodes=600,alpha=0.5,gamma=0.95,seed=0):
    rng=np.random.default_rng(seed)
    Q=np.zeros((H,W,4))
    curve=[]
    for _ in range(episodes):
        s,total=START,0.0
        for _ in range(120):
            if rng.random()<eps: a=int(rng.integers(4))
            else: a=int(np.argmax(Q[s[0],s[1]]))
            ns,r,done=step(s,a)
            total+=r
            Q[s[0],s[1],a]+=alpha*(r+gamma*np.max(Q[ns[0],ns[1]])*(not done)-Q[s[0],s[1],a])
            s=ns
            if done: break
        curve.append(total)
    return Q,np.array(curve)

def smooth(v,k=25): return np.convolve(v,np.ones(k)/k,mode='valid')

plt.figure(figsize=(9,3))
for eps in (0.05,0.20,0.50):
    Q,curve=train(eps)
    plt.plot(smooth(curve),lw=1.1,label=f'epsilon = {eps}')
    print(f'eps={eps}: mean return over last 100 = {curve[-100:].mean():.2f}')
plt.xlabel('episode'); plt.ylabel('return per episode (25-ep mean)'); plt.legend()
plt.title('Q-learning: return vs episode for three epsilon'); plt.tight_layout()
plt.savefig(f'{FIGS}/fig6_1_learning.png',dpi=150); plt.close()

Q,_=train(0.20)
policy=np.argmax(Q,axis=2)
# print policy grid
arrows=['^','v','<','>']
for r in range(H):
    print(' '.join(arrows[policy[r,c]] if (r,c) not in WALLS else '#' for c in range(W)))
print('done exp5')
