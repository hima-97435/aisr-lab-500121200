import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import os as _os
FIGS = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "figs")
os.makedirs(FIGS, exist_ok=True)

# Example 1: building a signal and reading its spectrum
fs, T = 500.0, 2.0
t = np.arange(0, T, 1/fs)
N = t.size
rng = np.random.default_rng(0)
x = (1.00*np.sin(2*np.pi*5*t)
    + 0.50*np.sin(2*np.pi*20*t)
    + 0.25*np.sin(2*np.pi*50*t)
    + 0.20*rng.standard_normal(N))
X = np.fft.rfft(x)
f = np.fft.rfftfreq(N, d=1/fs)
amp = np.abs(X)*2.0/N
amp[0] /= 2.0
print('N =', N, 'resolution df =', fs/N, 'Hz', 'Nyquist =', fs/2, 'Hz')
print('peaks at:', np.round(f[amp > 0.1], 2))
print('peak amplitudes:', np.round(amp[amp > 0.1], 2))

# Figure 2.1: time domain first 0.5s
plt.figure(figsize=(8,3))
plt.plot(t[t<0.5], x[t<0.5], lw=0.8, color='0.1')
plt.xlabel('time (s)'); plt.ylabel('amplitude'); plt.title('Constructed signal, first 0.5 s')
plt.tight_layout(); plt.savefig(f'{FIGS}/fig2_1_time.png', dpi=150); plt.close()

# Figure 2.2: spectrum
plt.figure(figsize=(8,3))
plt.plot(f, amp, lw=0.8, color='0.1')
plt.xlabel('frequency (Hz)'); plt.ylabel('amplitude')
plt.title('Amplitude spectrum')
plt.tight_layout(); plt.savefig(f'{FIGS}/fig2_2_spectrum.png', dpi=150); plt.close()

# Example 2: aliasing
fs_a = 8.0
t_a = np.arange(0, 1.0, 1/fs_a)
x_a = np.sin(2*np.pi*5*t_a)
X_a = np.fft.rfft(x_a)
f_a = np.fft.rfftfreq(t_a.size, d=1/fs_a)
print('Nyquist =', fs_a/2, 'Hz')
print('aliased peak at:', f_a[np.argmax(np.abs(X_a))], 'Hz')

# Figure 2.3
t_fine = np.linspace(0,1,500)
plt.figure(figsize=(8,3))
plt.plot(t_fine, np.sin(2*np.pi*5*t_fine), 'k--', lw=0.8, label='true 5 Hz')
plt.plot(t_fine, np.sin(2*np.pi*3*t_fine), 'r-', lw=0.8, label='apparent 3 Hz')
plt.plot(t_a, x_a, 'bo', ms=4, label='samples at 8 Hz')
plt.xlabel('time (s)'); plt.ylabel('amplitude'); plt.legend(fontsize=8)
plt.title('Aliasing: 5 Hz sampled at 8 Hz')
plt.tight_layout(); plt.savefig(f'{FIGS}/fig2_3_aliasing.png', dpi=150); plt.close()

# Example 3: pendulum
g, L = 9.81, 1.0
def pendulum(t, y, b):
    theta, omega = y
    return [omega, -b*omega - (g/L)*np.sin(theta)]
def simulate(theta0_deg, b, T=20.0, n=4000):
    te = np.linspace(0, T, n)
    sol = solve_ivp(pendulum, [0, T], [np.deg2rad(theta0_deg), 0.0], args=(b,), t_eval=te, rtol=1e-9, atol=1e-12)
    return sol.t, sol.y[0], sol.y[1]
T0 = 2*np.pi*np.sqrt(L/g)
print('T0 =', round(T0,4), 's')
for a in (5, 95, 175):
    tt, th, _ = simulate(a, 0.0, T=60.0, n=12000)
    fs2 = 1/(tt[1]-tt[0])
    A = np.abs(np.fft.rfft(th))
    fr = np.fft.rfftfreq(th.size, 1/fs2)
    pk = fr[1:][np.argmax(A[1:])]
    print(f'{a:>4} deg -> period {1/pk:.3f} s')

# Figure 2.4: damping
plt.figure(figsize=(8,3))
for b,label in ((0.0,'undamped b=0'),(0.3,'light b=0.3'),(2.0,'heavy b=2.0')):
    tt, th, _ = simulate(40, b, T=20.0, n=4000)
    plt.plot(tt, np.rad2deg(th), lw=0.9, label=label)
plt.xlabel('time (s)'); plt.ylabel('angle (deg)'); plt.legend(fontsize=8)
plt.title('Pendulum angle vs time for three damping values')
plt.tight_layout(); plt.savefig(f'{FIGS}/fig2_4_damping.png', dpi=150); plt.close()

# Figure 2.5: phase portrait + period vs amplitude
fig, axes = plt.subplots(1,2,figsize=(10,4))
tt, th, om = simulate(40, 0.0, T=20.0, n=4000)
axes[0].plot(np.rad2deg(th), om, lw=0.9, label='b=0')
tt2, th2, om2 = simulate(40, 0.3, T=20.0, n=4000)
axes[0].plot(np.rad2deg(th2), om2, lw=0.9, label='b=0.3')
axes[0].set_xlabel('angle (deg)'); axes[0].set_ylabel('angular velocity (rad/s)'); axes[0].legend(fontsize=8)
axes[0].set_title('Phase portrait')

periods=[]
for a in [5,30,60,90,120,150,175]:
    tt, th, _ = simulate(a, 0.0, T=60.0, n=12000)
    fs2 = 1/(tt[1]-tt[0])
    A = np.abs(np.fft.rfft(th))
    fr = np.fft.rfftfreq(th.size, 1/fs2)
    pk = fr[1:][np.argmax(A[1:])]
    periods.append(1/pk)
axes[1].plot([5,30,60,90,120,150,175], periods, 'o-', ms=4, label='measured')
axes[1].axhline(T0, color='r', ls='--', label=f'small angle T0={T0:.3f}s')
axes[1].set_xlabel('initial amplitude (deg)'); axes[1].set_ylabel('period (s)'); axes[1].legend(fontsize=8)
axes[1].set_title('Period vs amplitude')
plt.tight_layout(); plt.savefig(f'{FIGS}/fig2_5_phase_period.png', dpi=150); plt.close()
print('periods:', [round(p,3) for p in periods])
print('done exp1')
