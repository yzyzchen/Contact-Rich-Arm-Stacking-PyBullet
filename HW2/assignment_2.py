import numpy as np
import cvxpy as cp
np.set_printoptions(precision=3, suppress=True)
# Name: Yuzhou Chen
# Date: 10/03/2024

def get_contacts():
    """
        Return contact normals and locations as a matrix
        :return:
            - Contact Matrix R: <np.array> of size (2,3) containing the contact locations [r0 | r1 | r2]
            - Normal Matrix N: <np.array> of size (2,3) containing the contact locations [n0 | n1 | n2]
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE

    R = np.zeros((2, 3))  # TODO: Replace None with your result
    N = np.zeros((2, 3))  # TODO: Replace None with your result
    l = 1
    sqrt_2 = np.sqrt(2)/2

    n0 = np.array([-sqrt_2, -sqrt_2])
    n1 = np.array([1, 0])
    n2 = np.array([0,1])

    r0 = np.array([l/2 * sqrt_2, l/2 * sqrt_2])
    r1 = np.array([-l/2, 0])
    r2 = np.array([0, -l/2])

    N = np.column_stack((n0, n1, n2))
    R = np.column_stack((r0, r1, r2))
    # ------------------------------------------------
    return R, N


def calculate_grasp(R, N):
    """
        Return the grasp matrix as a function of contact locations and normals
        :param R: <np.array> locations of contact
        :param N: <np.array> contact normals
        :return: <np.array> of size (3,6) Grasp matrix for Fig. 1 containing [ J0 | J1 | J2]
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE

    G = np.zeros((3, 6))  # TODO: Replace with your result
    n0, n1, n2 = N[:, 0], N[:, 1], N[:, 2]
    r0, r1, r2 = R[:, 0], R[:, 1], R[:, 2]

    J0 = np.array([
        [n0[1], n0[0]],
        [-n0[0], n0[1]],
        [-r0[0]*n0[0] - r0[1]*n0[1], r0[0]*n0[1] - r0[1]*n0[0]]
    ])

    J1 = np.array([
        [n1[1], n1[0]],
        [-n1[0], n1[1]],
        [-r1[0]*n1[0] - r1[1]*n1[1], r1[0]*n1[1] - r1[1]*n1[0]]
    ])

    J2 = np.array([
        [n2[1], n2[0]],
        [-n2[0], n2[1]],
        [-r2[0]*n2[0] - r2[1]*n2[1], r2[0]*n2[1] - r2[1]*n2[0]]
    ])
    G = np.column_stack((J0, J1, J2))
    # ------------------------------------------------
    return G


def calculate_facet(mu):
    """
        Return friction cone representation in terms of facet normals
        :param mu: <float> coefficient of friction
        :return: <np.array> Facet normal matrix
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE

    F = np.zeros((6, 6))  # TODO: Replace with your result
    F_i = 1/(np.sqrt(1+mu**2)) * np.array([
                                        [1,mu],
                                        [-1, mu]
                                        ])
    F[0:2, 0:2] = F_i
    F[2:4, 2:4] = F_i
    F[4:6, 4:6] = F_i
    # ------------------------------------------------
    return F


def compute_grasp_rank(G):
    """
        Return boolean of if grasp has rank 3 or not
        :param G: <np.array> grasp matrix as a numpy array
        :return: <bool> boolean flag for if rank is 3 or not
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE
    rank = np.linalg.matrix_rank(G)
    if rank == 3:
        flag = True
    else:
        flag = False # TODO: Replace None with your result
    # ------------------------------------------------
    return flag


def compute_constraints(G, F):
    """
        Return grasp constraints as numpy arrays
        :param G: <np.array> grasp matrix as a numpy array
        :param F: <np.array> friction cone facet matrix as a numpy array
        :return: <np.array>x5 contact constraints
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE
    b = np.zeros(3)  # Make b a 1D array (shape (3,))
    A = np.block([[G, b.reshape(-1, 1)]])  # Grasp matrix combined with vector b reshaped as a column vector
    # Ensure eT and zero_block are of compatible dimensions
    eT = np.array([0, 1, 0, 1, 0, 1]).reshape(1, -1)  # Reshape to 1x6 row vector
    zero_block = np.array([[0]])  # Create a 1x1 block for concatenation
    P = np.block([
        [eT, zero_block],         
        [-F, np.ones((6, 1))],  
        [np.zeros((1, 6)), -1]
    ])
    q = np.block([3, np.zeros(7)]).flatten()  # Flatten q to make it 1D
    c = np.block([[np.zeros((6, 1))], [1]]).reshape(-1, 1)  # Ensure c is a column vector

    # b = np.zeros((3,1))     # TODO: Replace None with your result
    # A = np.block([[G,b]])   # TODO: Replace None with your result
    # eT = np.array([0, 1, 0, 1, 0, 1])
    # zero_block = np.array([0]) 
    # P = np.block([[eT,zero_block],
    #               [-F,1*np.ones((6,1))],
    #               [np.zeros(6,),-1]])    # TODO: Replace None with your result
    # q = np.block([3,np.zeros((7,))]).reshape(-1,1)   # TODO: Replace None with your result
    # c = np.block([[np.zeros((6,1))],[1]]).reshape(-1,1)   # TODO: Replace None with your result
    # ------------------------------------------------
    return A, b, P, q, c


def check_force_closure(A, b, P, q, c):
    """
        Solves Linear program given grasp constraints - DO NOT EDIT
        :return: d_star
    """
    # ------------------------------------------------
    # DO NOT EDIT THE CODE IN THIS FUNCTION
    x = cp.Variable(A.shape[1])
    # print(f"P @ x shape: {(P @ x).shape}")
    # print(f"q shape: {q.shape}")

    prob = cp.Problem(cp.Maximize(c.T@x),
                      [P @ x <= q, A @ x == b])
    prob.solve()
    d = prob.value
    print('Optimal value of d (d^*): {:3.2f}'.format(d))
    return d
    # ------------------------------------------------


if __name__ == "__main__":
    mu = 0.3
    R, N = get_contacts()
    G = calculate_grasp(R, N)
    F = calculate_facet(mu=mu)
    A, b, P, q, c = compute_constraints(G, F)
    d = check_force_closure(A, b, P, q, c)


