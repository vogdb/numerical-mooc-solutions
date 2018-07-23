import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams, animation

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16

# Du, Dv, F, k = 0.00016, 0.00008, 0.035, 0.065  # Bacteria 1
Du, Dv, F, k = 0.00019, 0.00005, 0.060, 0.062  # Fingerprint


def laplacian(f, dh):
    '''
    Laplacian of function `f`. `dy` is assumed to be as `dx`. Otherwise it would have been
    `(left + right - 2 * center) / dx ** 2 + (bottom + top - 2 * center) / dy ** 2`.
    :param f: two-dimensional array of values at some timestep.
    :param dh:
    :return:
    '''
    bottom = f[0:-2, 1:-1]
    left = f[1:-1, 0:-2]
    top = f[2:, 1:-1]
    right = f[1:-1, 2:]
    center = f[1:-1, 1:-1]
    return (bottom + left + top + right - 4 * center) / dh ** 2


def boundary_conditions(f):
    # Neumann BCs on all sides
    f[0, :] = f[1, :]
    f[:, 0] = f[:, 1]
    f[-1, :] = f[-2, :]
    f[:, -1] = f[:, -2]


def solve(U, V, dt, dh):
    for n in range(len(U) - 1):
        U[n + 1] = U[n].copy()
        V[n + 1] = V[n].copy()
        Un = U[n][1:-1, 1:-1]
        Vn = V[n][1:-1, 1:-1]
        ddU = laplacian(U[n], dh)
        ddV = laplacian(V[n], dh)
        U[n + 1][1:-1, 1:-1] = Un + dt * (Du * ddU - Un * Vn * Vn + F * (1 - Un))
        V[n + 1][1:-1, 1:-1] = Vn + dt * (Dv * ddV + Un * Vn * Vn - (F + k) * Vn)
        boundary_conditions(U[n + 1])
        boundary_conditions(V[n + 1])


def create_animation(U, V):
    speed = 50
    frame_num = int(len(U) / speed)

    def animate(i):
        ax_U.imshow(U[i * speed], cmap=plt.cm.RdBu)
        ax_V.imshow(V[i * speed], cmap=plt.cm.RdBu)

    fig = plt.figure(figsize=(8, 5))
    ax_U = plt.subplot(121)
    ax_U.set_axis_off()
    ax_U.set_title(r'U')
    ax_V = plt.subplot(122)
    ax_V.set_title(r'V')
    ax_V.set_axis_off()

    anim = animation.FuncAnimation(fig, animate, frames=frame_num, blit=False)
    anim.save('animation.mp4')


def main():
    runtime_start = time.time()

    n = 192
    dh = 5 / (n - 1)
    T = 8000
    dt = .9 * dh ** 2 / (4 * max(Du, Dv))
    nt = int(T / dt)

    U = np.zeros(nt, dtype=np.ndarray)
    V = np.zeros(nt, dtype=np.ndarray)

    # initial conditions
    UV_initial = np.load('./uv_initial.npz')
    U[0] = UV_initial['U']
    V[0] = UV_initial['V']

    solve(U, V, dt, dh)
    runtime_calc = time.time()
    print('Calculation is done in {:.2f} seconds'.format(runtime_calc - runtime_start))
    create_animation(U, V)
    runtime_end = time.time()
    print('Animation is done in {:.2f} seconds'.format(runtime_end - runtime_calc))


main()
