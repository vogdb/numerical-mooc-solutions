import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams  # ,cm
from matplotlib import animation

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 16


def laplacian(f, dx):
    '''
    Laplacian of function `f`. `dy` is assumed to be as `dx`. Otherwise it would have been
    `(left + right - 2 * center) / dx ** 2 + (bottom + top - 2 * center) / dy ** 2`.
    :param f: two-dimensional array of values at some timestep.
    :param dx:
    :return:
    '''
    bottom = f[0:-2, 1:-1]
    left = f[1:-1, 0:-2]
    top = f[2:, 1:-1]
    right = f[1:-1, 2:]
    center = f[1:-1, 1:-1]
    return (bottom + left + top + right - 4 * center) / dx ** 2


def boundary_conditions(f):
    # Enforce Neumann BCs
    f[-1, :] = f[-2, :]
    f[:, -1] = f[:, -2]


def solve(T, alpha, dt, dx):
    for n in range(len(T) - 1):
        T[n + 1] = T[n].copy()
        T[n + 1][1:-1, 1:-1] = T[n][1:-1, 1:-1] + alpha * dt * laplacian(T[n], dx)
        boundary_conditions(T[n + 1])
    return T


def create_animation(L, T, nx):
    def animate(i):
        # contour.set_data(x, y, T[i])
        contour = plt.contourf(x, y, T[i])
        plt.title(r't = %i' % i)
        return contour

    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, nx)
    x, y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8, 5))
    ax = plt.axes(xlim=(0, L), ylim=(0, L))
    # contour = ax.contourf([], [], [])[0]
    plt.xlabel(r'x')
    plt.ylabel(r'y')
    plt.colorbar()

    anim = animation.FuncAnimation(fig, animate, frames=len(T), blit=False)
    anim.save('animation.mp4')
    # plt.contourf(x, y, T[-1], 20, cmap=cm.viridis)
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    # plt.show()


def main():
    L = 1.0e-2
    nx = 21
    nt = 200
    dx = L / (nx - 1)
    T = np.zeros(nt, dtype=np.ndarray)

    # conditions at time 0
    T[0] = np.ones((nx, nx)) * 20
    T[0][0, :] = 100
    T[0][:, 0] = 100

    alpha = 1e-4
    sigma = 0.25
    dt = sigma * dx ** 2 / alpha
    solve(T, alpha, dt, dx)

    create_animation(L, T, nx)


main()
