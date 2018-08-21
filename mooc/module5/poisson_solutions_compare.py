import numpy as np
import time
import numba
from numba import jit


def p_analytical(X, Y, L):
    ''' Computes analytical solution to Poisson problem
    Parameters:
    ----------
    X: 2D array of float
        Mesh with x components
    Y: 2D array of float
        Mesh with y components
    L: float
        Size of domain
    Returns:
    -------
    Analytical solution
    '''
    return np.sin(X * np.pi / L) * np.cos(Y * np.pi / L)


def poisson_IG(nx, ny, xmax, xmin, ymax, ymin):
    '''Initialize the Poisson problem initial guess and other variables
    Parameters:
    ----------
    nx : int
        number of mesh points in x
    ny : int
        number of mesh points in y
    xmax: float
        maximum value of x in mesh
    xmin: float
        minimum value of x in mesh
    ymax: float
        maximum value of y in mesh
    ymin: float
        minimum value of y in mesh
    
    Returns:
    -------
    X  : 2D array of floats
        X-position of mesh
    Y  : 2D array of floats
        Y-position of mesh
    x  : 1D array of floats
        x range
    y  : 1D array of floats
        y range
    p_i: 2D array of floats
        initial guess of p
    b  : 2D array of floats
        forcing function
    dx : float
        mesh size in x direction
    dy : float
        mesh size in y direction
    '''

    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)

    # Mesh
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    # Source
    L = xmax - xmin
    b = -2 * (np.pi / L) ** 2 * np.sin(np.pi * X / L) * np.cos(np.pi * Y / L)

    # Initialize
    p_i = np.zeros((ny, nx))

    return X, Y, x, y, p_i, b, dx, dy, L


def constructMatrix(nx, ny):
    """ Generate implicit matrix for poisson equation with Dirichlet == 0 everywhere
        Assumes dx = dy

    Parameters:
    ----------
    nx   : int
        number of discretization points in x
    ny   : int
        number of discretization points in y
    sigma: float
        alpha*dt/dx

    Returns:
    -------
    A: 2D array of floats
        Matrix of implicit 2D heat equation
    """

    A = np.zeros(((nx - 2) * (ny - 2), (nx - 2) * (ny - 2)))

    row_number = 0  # row counter
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):

            # Corners
            if i == 1 and j == 1:
                A[row_number, row_number] = -4  # Set diagonal
                A[row_number, row_number + 1] = 1  # fetch i+1
                A[row_number, row_number + nx - 2] = 1  # fetch j+1

            elif i == nx - 2 and j == 1:
                A[row_number, row_number] = -4  # Set diagonal
                A[row_number, row_number - 1] = 1  # Fetch i-1
                A[row_number, row_number + nx - 2] = 1  # fetch j+1

            elif i == 1 and j == ny - 2:
                A[row_number, row_number] = -4  # Set diagonal
                A[row_number, row_number + 1] = 1  # fetch i+1
                A[row_number, row_number - (nx - 2)] = 1  # fetch j-1

            elif i == nx - 2 and j == ny - 2:
                A[row_number, row_number] = -4  # Set diagonal
                A[row_number, row_number - 1] = 1  # Fetch i-1
                A[row_number, row_number - (nx - 2)] = 1  # fetch j-1

            # Sides
            elif i == 1:  # Left boundary
                A[row_number, row_number] = -4  # Set diagonal
                A[row_number, row_number + 1] = 1  # fetch i+1
                A[row_number, row_number + nx - 2] = 1  # fetch j+1
                A[row_number, row_number - (nx - 2)] = 1  # fetch j-1

            elif i == nx - 2:  # Right boundary
                A[row_number, row_number] = -4  # Set diagonal
                A[row_number, row_number - 1] = 1  # Fetch i-1
                A[row_number, row_number + nx - 2] = 1  # fetch j+1
                A[row_number, row_number - (nx - 2)] = 1  # fetch j-1

            elif j == 1:  # Bottom boundary
                A[row_number, row_number] = -4  # Set diagonal
                A[row_number, row_number + 1] = 1  # fetch i+1
                A[row_number, row_number - 1] = 1  # fetch i-1
                A[row_number, row_number + nx - 2] = 1  # fetch j+1

            elif j == ny - 2:  # Top boundary
                A[row_number, row_number] = -4  # Set diagonal
                A[row_number, row_number + 1] = 1  # fetch i+1
                A[row_number, row_number - 1] = 1  # fetch i-1
                A[row_number, row_number - (nx - 2)] = 1  # fetch j-1

            # Interior points
            else:
                A[row_number, row_number] = -4  # Set diagonal
                A[row_number, row_number + 1] = 1  # fetch i+1
                A[row_number, row_number - 1] = 1  # fetch i-1
                A[row_number, row_number + nx - 2] = 1  # fetch j+1
                A[row_number, row_number - (nx - 2)] = 1  # fetch j-1

            row_number += 1  # Jump to next row of the matrix!

    return A


def generateRHS(nx, ny, sigma, p, p_bc):
    """ Generates right-hand side for poisson equation with Dirichlet everywhere
        Assumes dx=dy, Neumann BCs = 0, and constant Dirichlet BCs

        Paramenters:
        -----------
        nx   : int
            number of discretization points in x
        ny   : int
            number of discretization points in y
        sigma: float
            dx^2 * dy^2
        p    : array of float
            poisson imposed
        p_bc : float
            poisson imposed in Dirichlet BC

        Returns:
        -------
        RHS  : array of float
            Right hand side of 2D implicit heat equation
    """
    RHS = np.zeros((nx - 2) * (ny - 2))

    row_number = 0  # row counter
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):

            # Corners
            if i == 1 and j == 1:
                RHS[row_number] = p[j, i] * sigma + 2 * p_bc

            elif i == nx - 2 and j == 1:
                RHS[row_number] = p[j, i] * sigma + 2 * p_bc

            elif i == 1 and j == ny - 2:
                RHS[row_number] = p[j, i] * sigma + 2 * p_bc

            elif i == nx - 2 and j == ny - 2:
                RHS[row_number] = p[j, i] * sigma + 2 * p_bc

            # Sides
            elif i == 1:
                RHS[row_number] = p[j, i] * sigma + p_bc

            elif i == nx - 2:
                RHS[row_number] = p[j, i] * sigma + p_bc

            elif j == 1:
                RHS[row_number] = p[j, i] * sigma + p_bc

            elif j == ny - 2:
                RHS[row_number] = p[j, i] * sigma + p_bc

            # Interior points
            else:
                RHS[row_number] = p[j, i] * sigma

            row_number += 1  # Jump to next row!

    return RHS


def map_1Dto2D(nx, ny, p_1D, p_bc):
    """ Takes solution of linear system, stored in 1D, and puts them in a 2D array with the BCs

    Parameters:
    ----------
        nx  : int
            number of nodes in x direction
        ny  : int
            number of nodes in y direction
        p_1D: array of floats
            solution of linear system
        p_bc: float
            Dirichlet BC

    Returns:
    -------
        T: 2D array of float
            p stored in 2D array with BCs
    """
    p = np.zeros((ny, nx))

    row_number = 0
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            p[j, i] = p_1D[row_number]
            row_number += 1
    # Dirichlet BC
    p[0, :] = p_bc
    p[:, 0] = p_bc
    p[-1, :] = p_bc
    p[:, -1] = p_bc

    return p


@jit(nopython=True)
def poisson2d_SOR(p, pn, b, l2_target, omega):
    '''Solves the Laplace equation using SOR with a 5-point stencil

    Parameters:
    ----------
    p: 2D array of float
        Initial potential distribution
    b: 2D array of float
        right hand side of Poisson
    l2_target: float
        Stopping criterion
    omega: float
        Relaxation parameter

    Returns:
    -------
    p: 2D array of float
        Potential distribution after relaxation
    '''

    iterations = 0
    iter_diff = l2_target + 1  # initialize iter_diff to be larger than l2_target
    ny, nx = p.shape

    while iter_diff > l2_target:
        for j in range(ny):
            for i in range(nx):
                pn[j, i] = p[j, i]

        iter_diff = 0.0
        denominator = 0.0

        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                # p[j, i] = 0.25 * (p[j, i - 1] + p[j, i + 1] + p[j - 1, i] + p[j + 1, i] - b[j, i])
                p[j, i] = (1 - omega) * p[j, i] + omega * .25 * \
                          (p[j, i - 1] + p[j, i + 1] + p[j - 1, i] + p[j + 1, i] - b[j, i])

        for j in range(ny):
            for i in range(nx):
                iter_diff += (p[j, i] - pn[j, i]) ** 2
                denominator += (pn[j, i] * pn[j, i])

        iter_diff /= denominator
        iter_diff = iter_diff ** 0.5
        iterations += 1

    return p, iterations


def compare():
    nx = 41
    ny = 41
    xmin = 0
    xmax = 1
    ymin = -0.5
    ymax = 0.5

    X, Y, x, y, p_i, b, dx, dy, L = poisson_IG(nx, ny, xmax, xmin, ymax, ymin)
    p_an = p_analytical(X, Y, L)

    start_time = time.time()
    A = constructMatrix(nx, ny)
    p_bc = 0
    rhs = generateRHS(nx, ny, dx ** 2, b, p_bc)
    p_lin = np.linalg.solve(A, rhs)
    p_lin = map_1Dto2D(nx, ny, p_lin, p_bc)
    end_time = time.time()
    print('matrix calc time: {:.3f} ms'.format(1000 * (end_time - start_time)))

    start_time = time.time()
    p_i[int(ny/2), int(nx/2)] = 1e-5
    p_sor, iterations = poisson2d_SOR(p_i, np.empty_like(p_i), b * dx**2, 1e-8, 2. / (1 + np.pi / nx))
    end_time = time.time()
    print('SOR relaxation calc time: {:.3f} ms'.format(1000 * (end_time - start_time)))

    print(np.allclose(p_an, p_lin, atol=1e-3))
    print(np.allclose(p_an, p_sor, atol=1e-3))
    # print(np.hstack((p_an[:, 1][:, np.newaxis], p_lin[:, 1][:, np.newaxis])))
    # print(np.hstack((p_an[1, :][:, np.newaxis], p_lin[1, :][:, np.newaxis])))
    # print(np.hstack((p_an[:, 1][:, np.newaxis], p_sor[:, 1][:, np.newaxis])))
    # print(np.hstack((p_an[1, :][:, np.newaxis], p_sor[1, :][:, np.newaxis])))


compare()
