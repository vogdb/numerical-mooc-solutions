import numpy as np
from scipy.linalg import solve

from animation import create_animation


def constructMatrix(nx, ny, sigma):
    """ Generate implicit matrix for 2D heat equation with Dirichlet in bottom and right and Neumann in top and left
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

    def get_row_number(i, j):
        return (j - 1) * (nx - 2) + (i - 1)

    shape = ((nx - 2) * (ny - 2), (nx - 2) * (ny - 2))
    A = np.zeros(shape)
    # -1 * (i-1,j) + -1 * (i+1,j) + (1/sigma + 4) * (i,j) + -1 * (i,j-1) + -1 * (i,j+1)
    A_i_j = np.eye(shape[0]) * (1 / sigma + 4)  # main i,j
    A_im1_j = np.eye(shape[0], k=-1) * (-1)  # i-1,j
    A_ipl1_j = np.eye(shape[0], k=1) * (-1)  # i+1,j
    A_i_jm1 = np.eye(shape[0], k=-(nx - 2)) * (-1)  # i,j-1
    A_i_jpl1 = np.eye(shape[0], k=nx - 2) * (-1)  # i,j+1
    A += A_i_j + A_im1_j + A_ipl1_j + A_i_jm1 + A_i_jpl1

    # General boundary conditions. They are relevant either for Dirichle, Neumann and etc.
    # A contains [1, nx-2] and [1, ny-2] elements, so boundaries [:,0], [:,ny-1], [0,:], [nx-1,:]
    # are not presented in A.

    # Bottom [:,0] and Top [:,ny-1] boundaries. They are implicitly applied as they are not presented in A.
    # Bottom has negative indexes and Top has indexes outside of A.shape.
    # row_number = get_row_number(np.arange(1, nx - 1), 1)
    # A[row_number - 1, row_number] = 0
    # row_number = get_row_number(np.arange(1, nx - 1), ny - 1)
    # A[row_number + 1, row_number] = 0

    # Left [0,:] boundary. Now [1,:] considers its left neighbour as [0,:] which is not true.
    # It's left neighbour is [nx-2,:]. So, we must zero its participation.
    row_number = get_row_number(1, np.arange(1, ny - 1))
    A[row_number, row_number - 1] = 0
    # Right [nx-1,:] boundary. Now [nx-2,:] considers its right neighbour as [nx-1,:] which is not true.
    # It's right neighbour is [1,:]. So, we must zero its participation.
    row_number = get_row_number(nx - 2, np.arange(1, ny - 1 - 1))  # (ny - 1 - 1) for the last A[-1,-1]
    A[row_number, row_number + 1] = 0

    # right boundary Neumann condition
    # (nx-2)+1,j becomes (nx-2),j, which decreases by 1 original (nx-2),j
    row_number = get_row_number(nx - 2, np.arange(1, ny - 1))
    A[row_number, row_number] -= 1

    # top boundary Neumann condition
    row_number = get_row_number(np.arange(1, nx - 1), ny - 2)
    A[row_number, row_number] -= 1
    return A


def generateRHS(nx, ny, sigma, T, T_bc):
    """ Generates right-hand side for 2D implicit heat equation with Dirichlet in bottom and left and Neumann in top and right
        Assumes dx=dy, Neumann BCs = 0, and constant Dirichlet BCs

        Paramenters:
        -----------
        nx   : int
            number of discretization points in x
        ny   : int
            number of discretization points in y
        sigma: float
            alpha*dt/dx
        T    : array of float
            Temperature in current time step
        T_bc : float
            Temperature in Dirichlet BC

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
            if i == 1 and j == 1:  # Bottom left corner (Dirichlet down and left)
                RHS[row_number] = T[j, i] * 1 / sigma + 2 * T_bc

            elif i == nx - 2 and j == 1:  # Bottom right corner (Dirichlet down, Neumann right)
                RHS[row_number] = T[j, i] * 1 / sigma + T_bc

            elif i == 1 and j == ny - 2:  # Top left corner (Neumann up, Dirichlet left)
                RHS[row_number] = T[j, i] * 1 / sigma + T_bc

            elif i == nx - 2 and j == ny - 2:  # Top right corner (Neumann up and right)
                RHS[row_number] = T[j, i] * 1 / sigma

                # Sides
            elif i == 1:  # Left boundary (Dirichlet)
                RHS[row_number] = T[j, i] * 1 / sigma + T_bc

            elif i == nx - 2:  # Right boundary (Neumann)
                RHS[row_number] = T[j, i] * 1 / sigma

            elif j == 1:  # Bottom boundary (Dirichlet)
                RHS[row_number] = T[j, i] * 1 / sigma + T_bc

            elif j == ny - 2:  # Top boundary (Neumann)
                RHS[row_number] = T[j, i] * 1 / sigma

            # Interior points
            else:
                RHS[row_number] = T[j, i] * 1 / sigma

            row_number += 1  # Jump to next row!

    return RHS


def map_1Dto2D(nx, ny, T_1D, T_bc):
    """ Takes temperatures of solution of linear system, stored in 1D,
    and puts them in a 2D array with the BCs
    Valid for constant Dirichlet bottom and left, and Neumann with zero
    flux top and right

    Parameters:
    ----------
        nx  : int
            number of nodes in x direction
        ny  : int
            number of nodes in y direction
        T_1D: array of floats
            solution of linear system
        T_bc: float
            Dirichlet BC

    Returns:
    -------
        T: 2D array of float
            Temperature stored in 2D array with BCs
    """
    T = np.zeros((ny, nx))

    row_number = 0
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            T[j, i] = T_1D[row_number]
            row_number += 1
    # Dirichlet BC
    T[0, :] = T_bc
    T[:, 0] = T_bc
    # Neumann BC
    T[-1, :] = T[-2, :]
    T[:, -1] = T[:, -2]

    return T


def btcs_2D(T, A, nt, sigma, T_bc, nx, ny, dt):
    """ Advances diffusion equation in time with backward Euler
   
    Parameters:
    ----------
    T: 2D array of float
        initial temperature profile
    A: 2D array of float
        Matrix with discretized diffusion equation
    nt: int
        number of time steps
    sigma: float
        alpha*dt/dx^2
    T_bc : float 
        Dirichlet BC temperature
    nx   : int
        Discretization points in x
    ny   : int
        Discretization points in y
    dt   : float
        Time step size
        
    Returns:
    -------
    T: 2D array of floats
        temperature profile after nt time steps
    """

    j_mid = int((np.shape(T)[0]) / 2)
    i_mid = int((np.shape(T)[1]) / 2)
    T_record = []

    for t in range(nt):
        Tn = T.copy()
        T_record.append(Tn)
        b = generateRHS(nx, ny, sigma, Tn, T_bc)
        # Use np.linalg.solve
        T_interior = solve(A, b)
        T = map_1Dto2D(nx, ny, T_interior, T_bc)

        # Check if we reached T=70C
        if T[j_mid, i_mid] >= 70:
            print("Center of plate reached 70C at time {0:.2f}s, in time step {1:d}.".format(dt * t, t))
            break

    if T[j_mid, i_mid] < 70:
        print("Center has not reached 70C yet, it is only {0:.2f}C.".format(T[j_mid, i_mid]))

    T_record.append(T)
    return T, T_record


alpha = 1e-4

L = 1.0e-2
H = 1.0e-2

nx = 21
ny = 21
nt = 300

dx = L / (nx - 1)
dy = H / (ny - 1)

x = np.linspace(0, L, nx)
y = np.linspace(0, H, ny)

T_bc = 100

Ti = np.ones((ny, nx)) * 20
Ti[0, :] = T_bc
Ti[:, 0] = T_bc

sigma = 0.25
A = constructMatrix(nx, ny, sigma)

dt = sigma * min(dx, dy) ** 2 / alpha
T, T_record = btcs_2D(Ti.copy(), A, nt, sigma, T_bc, nx, ny, dt)

original = np.load('implicit_original.npz')
T_original, A_original = original['T'], original['A']
print(np.allclose(T_record, T_original))
print(np.array_equal(A, A_original))
# flip to show it as we see it in our calculation
create_animation('implicit', [np.flip(T, 0) for T in T_record], speed=2)
