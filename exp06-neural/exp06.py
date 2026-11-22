import numpy as np, matplotlib.pyplot as plt, os
import os as _os
FIGS=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "figs"); os.makedirs(FIGS, exist_ok=True)

X=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=float)
y=np.array([[0],[1],[1],[0]],dtype=float)
def sigmoid(z): return 1/(1+np.exp(-z))

def train_xor(lr,epochs=20000,seed=1,hidden=4):
    rng=np.random.default_rng(seed)
    W1=rng.normal(0,1,(2,hidden)); b1=np.zeros((1,hidden))
    W2=rng.normal(0,1,(hidden,1)); b2=np.zeros((1,1))
    loss=[]
    for _ in range(epochs):
        h=sigmoid(X@W1+b1); o=sigmoid(h@W2+b2)
        loss.append(float(np.mean((o-y)**2)))
        do=(o-y)*o*(1-o)
        dW2=h.T@do; db2=do.sum(0,keepdims=True)
        dh=(do@W2.T)*h*(1-h)
        dW1=X.T@dh; db1=dh.sum(0,keepdims=True)
        W2-=lr*dW2; b2-=lr*db2
        W1-=lr*dW1; b1-=lr*db1
    return (W1,b1,W2,b2),np.array(loss)

# loss curves
plt.figure(figsize=(9,3))
for lr in (0.05,0.5,2.0):
    _,l=train_xor(lr)
    plt.semilogy(l,lw=1.0,label=f'learning rate = {lr}')
    print(f'lr={lr}: final MSE = {l[-1]:.2e}')
plt.xlabel('epoch'); plt.ylabel('mean squared error'); plt.legend()
plt.title('XOR training loss vs epoch'); plt.tight_layout()
plt.savefig(f'{FIGS}/fig7_1_loss.png',dpi=150); plt.close()

# trained params for boundary
params,loss=train_xor(2.0)
W1,b1,W2,b2=params
print('final loss:',loss[-1])
preds=sigmoid(sigmoid(X@W1+b1)@W2+b2).ravel().round(3)
print('predictions:',preds)

# boundary
gx,gy=np.meshgrid(np.linspace(-0.25,1.25,300),np.linspace(-0.25,1.25,300))
G=np.c_[gx.ravel(),gy.ravel()]
Z=sigmoid(sigmoid(G@W1+b1)@W2+b2).reshape(gx.shape)
plt.figure(figsize=(5,4))
plt.contourf(gx,gy,Z,levels=np.linspace(0,1,11),cmap='Greys')
plt.contour(gx,gy,Z,levels=[0.5],colors='k')
plt.scatter(X[:,0],X[:,1],c=y.ravel(),edgecolors='k',cmap='bwr',s=60)
plt.xlabel('x1'); plt.ylabel('x2'); plt.title('Learned decision boundary')
plt.tight_layout(); plt.savefig(f'{FIGS}/fig7_2_boundary.png',dpi=150); plt.close()

# try lr=10
_,l10=train_xor(10.0)
print('lr=10 final MSE:', l10[-1])
print('done exp6')
