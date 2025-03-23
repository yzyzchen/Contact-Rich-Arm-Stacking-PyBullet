import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def assignment_3_render(traj):
    """
    Function designed to render the trajectory of the dropped block using matplotlib
    :param traj:
    :return: None
    """
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-0.1, 2))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    dt = 0.01

    def animate(i):
        # draws square given its configuration
        local_corners = np.array([[-0.2, -0.2, 0.2, 0.2, -0.2],# Multiply by 0.5
                                  [-0.2, 0.2, 0.2, -0.2, -0.2],
                                  [1, 1, 1, 1, 1]])
        H = np.array([[np.cos(traj[2, i]), -np.sin(traj[2, i]), traj[0, i]],
                      [np.sin(traj[2, i]), np.cos(traj[2, i]), traj[1, i]],
                      [0., 0., 1]])
        world_corners = H @ local_corners

        line.set_data(world_corners[0, :], world_corners[1, :])
        time_text.set_text(time_template % (i * dt))
        return line, time_text

    ani = animation.FuncAnimation(
        fig, animate, traj.shape[1], interval=dt * 3000, blit=True)
    plt.show()


def LCPSolve(M, q, pivtol=1e-8):
    """
    Function called to solve the Linear Complementary Problem Equations
    :param M:
    :param q:
    :param pivtol: smallest allowable pivot element
    :return:
    """
    rayTerm = False
    loopcount = 0
    if (q >= 0.).all():  # Test missing in Rob Dittmar's code
        # As w - Mz = q, if q >= 0 then w = q and z = 0
        w = q
        z = np.zeros_like(q)
        retcode = 0.
    else:
        dimen = M.shape[0]  # number of rows
        # Create initial tableau
        tableau = np.hstack([np.eye(dimen), -M, -np.ones((dimen, 1)), np.asarray(np.asmatrix(q).T)])
        # Let artificial variable enter the basis
        basis = list(range(dimen))  # basis contains a set of COLUMN indices in the tableau
        locat = np.argmin(tableau[:, 2 * dimen + 1])  # row of minimum element in column 2*dimen+1 (last of tableau)
        basis[locat] = 2 * dimen  # replace that choice with the row
        cand = locat + dimen
        pivot = tableau[locat, :] / tableau[locat, 2 * dimen]
        tableau -= tableau[:,
                   2 * dimen:2 * dimen + 1] * pivot  # from each column subtract the column 2*dimen, multiplied by pivot
        tableau[locat, :] = pivot  # set all elements of row locat to pivot
        # Perform complementary pivoting
        oldDivideErr = np.seterr(divide='ignore')['divide']  # suppress warnings or exceptions on zerodivide inside numpy
        while np.amax(basis) == 2 * dimen:
            loopcount += 1
            eMs = tableau[:, cand]  # Note: eMs is a view, not a copy! Do not assign to it...
            missmask = eMs <= 0.
            quots = tableau[:, 2 * dimen + 1] / eMs  # sometimes eMs elements are zero, but we suppressed warnings...
            quots[missmask] = np.Inf  # in any event, we set to +Inf elements of quots corresp. to eMs <= 0.
            locat = np.argmin(quots)
            if abs(eMs[locat]) > pivtol and not missmask.all():  # and if at least one element is not missing
                # reduce tableau
                pivot = tableau[locat, :] / tableau[locat, cand]
                tableau -= tableau[:, cand:cand + 1] * pivot
                tableau[locat, :] = pivot
                oldVar = basis[locat]
                # New variable enters the basis
                basis[locat] = cand
                # Select next candidate for entering the basis
                if oldVar >= dimen:
                    cand = oldVar - dimen
                else:
                    cand = oldVar + dimen
            else:
                rayTerm = True
                break
        np.seterr(divide=oldDivideErr)  # restore original handling of zerodivide in Numpy
        # Return solution to LCP
        vars = np.zeros(2 * dimen + 1)
        vars[basis] = tableau[:, 2 * dimen + 1]
        w = vars[:dimen]
        z = vars[dimen:2 * dimen]
        retcode = vars[2 * dimen]
    # end if (q >= 0.).all()

    if rayTerm:
        retcode = (2, retcode, loopcount)  # ray termination
    else:
        retcode = (1, retcode, loopcount)  # success
    return (w, z, retcode)