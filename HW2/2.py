import numpy as np
import cvxpy as cp
np.set_printoptions(precision=3, suppress=True)


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

    r0 = np.array([(l/2)*0.7071 , (l/2)*0.7071]) 
    r1 = np.array([-l/2, 0])  
    r2 = np.array([0, -l/2])  
    
    n0 = np.array([-0.7071, -0.7071])  
    n1 = np.array([1, 0])  
    n2 = np.array([0, 1])  

    R = np.column_stack((r0, r1, r2))
    N = np.column_stack((n0, n1, n2))
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
    # Extract individual contact locations and normals
    r0, r1, r2 = R[:, 0], R[:, 1], R[:, 2]
    n0, n1, n2 = N[:, 0], N[:, 1], N[:, 2]

    # Calculate the tangential and normal components of the grasp matrix columns
    J0_tangential = np.array([n0[1],
                             -n0[0],
                             -r0[0]*n0[0] - r0[1]*n0[1]])
    J0_normal = np.array([n0[0],
                             n0[1],
                             r0[0]*n0[1] - r0[1]*n0[0]])
    
    J1_tangential = np.array([n1[1],
                             -n1[0],
                             -r1[0]*n1[0] - r1[1]*n1[1]])
    J1_normal = np.array([n1[0],
                             n1[1],
                             r1[0]*n1[1] - r1[1]*n1[0]])

    J2_tangential = np.array([n2[1],
                             -n2[0],
                             -r2[0]*n2[0] - r2[1]*n2[1]])
    J2_normal = np.array([n2[0],
                             n2[1],
                             r2[0]*n2[1] - r2[1]*n2[0]])
    

    # Combine the tangential and normal components to form the grasp matrix
    G = np.column_stack((J0_tangential,J0_normal,J1_tangential,J1_normal,J2_tangential,J2_normal))

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
    k = 1/(np.sqrt(1+mu**2))
    M = np.array([[1,mu],[-1, mu]])
    F_i = k*M
    F = np.block([
        [F_i, np.zeros((2,4))],
        [np.zeros((2,2)),F_i,np.zeros((2,2))],
        [np.zeros((2,4)), F_i]
        ])
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
    #print(rank)
    # ------------------------------------------------
    if rank==3:
        flag=True
    else:
        flag = False
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

    A = np.block([[G,np.zeros((3,1))]])
    #print(np.shape(A))
    b = np.zeros((3,1))   
    #print(np.shape(b))
    et = np.array([0,1,0,1,0,1])
    #print(np.shape(b))
    nc = 3
    P = np.block([[et,0],
                  [-F,1*np.ones((6,1))],
                  [np.zeros(6,),-1]])  
    #print(np.shape(P))
    q = np.block([nc,np.zeros((7,))]).reshape(-1,1)
    #print(np.shape(q))
    c = np.block([[np.zeros((6,1))],[1]]).reshape(-1,1)
    #print(np.shape(q))
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
    print(f"P @ x shape: {(P @ x).shape}")
    print(f"q shape: {q.shape}")

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


