import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as la


def Make_B_List(invexpdtK, expVData):
    """Make list of off-diagonal matrices in matrix O. deriv. Eq. (20)

    Args:
        expVData, (N,L) matrix, each column reps a (N,N) diagonal mat

    Returns:
        python list of (N,N) matrices."""

    _, L = expVData.shape
    B_list = []
    for j in range(L):
        Bmat = invexpdtK * expVData[:, j]
        # note negative signs, except for bdy term B_{L-1}
        if j == L - 1:
            B_list.append(Bmat)
        else:
            B_list.append(-Bmat)
    return B_list


def Show_P_Cyclic_Full(B_list):
    """Given list of B matrices, make and plot the full (NL,NL) dimensional
    normalized block p-cyclic matrix O.

    Args:
        B_list: length L list of sub-diagonal and boundary blocks of p-cyclic matrix

    Returns:
        O_mat: (NL,NL) np array"""

    L = len(B_list)
    N, N = B_list[0].shape
    O_mat = np.zeros((N * L, N * L))

    for i in range(L):
        # diagonal: all identity
        O_mat[i * N : (i + 1) * N, i * N : (i + 1) * N] = np.identity(N)
        if i == L - 1:  # special boundary term
            O_mat[0:N, i * N : (i + 1) * N] = B_list[i]
        else:  # -1 diagonal
            O_mat[(i + 1) * N : (i + 2) * N, i * N : (i + 1) * N] = B_list[i]

    plt.figure()
    plt.imshow(O_mat)
    plt.colorbar()
    plt.title(f"Full P-cyclic matrix L*N = {L*N}")

    return O_mat


def Get_R_Full(R_dict, N, L):
    """Given dictionary of R entries, create full (NL,NL) R matrix, which
    has block upper bidiagonal form with an extra last block column

    Args:
        R_dict: size (3L-3) dictionary of R matrix block entries.
                key = tuple of entry coordinates
                value = (N,N) np array

    Returns:
        R_full: (NL,NL) upper triangular np array"""
    R_full = np.zeros((N * L, N * L))
    for k, v in R_dict.items():
        i = k[0]
        j = k[1]
        R_full[i * N : (i + 1) * N, j * N : (j + 1) * N] = v

    # plt.figure();plt.imshow(R_full);plt.colorbar()
    # plt.title(f"Full R matrix L*N = {L*N}")

    return R_full


def Show_R_Full(R_dict, N, L):
    """Given dictionary of R entries, create and show full (NL,NL) R matrix, which
    has block upper bidiagonal form with an extra last block column

    Args:
        R_dict: size (3L-3) dictionary of R matrix block entries.
                key = tuple of entry coordinates
                value = (N,N) np array

    Returns:
        R_full: (NL,NL) upper triangular np array"""
    R_full = np.zeros((N * L, N * L))
    for k, v in R_dict.items():
        i = k[0]
        j = k[1]
        R_full[i * N : (i + 1) * N, j * N : (j + 1) * N] = v

    plt.figure()
    plt.imshow(R_full)
    plt.colorbar()
    plt.title(f"Full R matrix L*N = {L*N}")

    return R_full


def Show_Q_Full(Q_list, N, L):
    """Given list of Q (2N,2N) matrices, create and show full (NL,NL) Q matrix

    Args:
        Q_list, length L-1 list of (2N,2N) orthogonal matrices

    Returns:
        Q_full: (NL,NL) orthogonal matrix"""

    Q_full = np.identity(N * L)

    for k in range(len(Q_list)):
        pad = np.eye(N * L)
        pad[k * N : (k + 2) * N, k * N : (k + 2) * N] = Q_list[k]
        Q_full = Q_full @ pad  # Q0 @ Q1 @ ... @ Q(L-2)

    plt.figure()
    plt.imshow(Q_full)
    plt.colorbar()
    plt.title(f"Full Q matrix L*N = {L*N}")

    return Q_full


def Show_Q_Inv_Full(Q_list, N, L):
    """Given list of Q (2N,2N) matrices, create and show full (NL,NL) Q matrix

    Args:
        Q_list, length L-1 list of (2N,2N) orthogonal matrices

    Returns:
        Q_inv_full: (NL,NL) orthogonal matrix"""
    Q_inv_full = np.identity(N * L)
    for k in range(len(Q_list)):
        pad = np.eye(N * L)
        pad[k * N : (k + 2) * N, k * N : (k + 2) * N] = Q_list[k]
        Q_inv_full = pad.T @ Q_inv_full  # not sure about mulciplication order

    plt.figure()
    plt.imshow(Q_inv_full)
    plt.colorbar()
    plt.title(f"Full Inverse Q matrix L*N = {L*N}")

    return Q_inv_full


def Block_Struct_Orth_Fact(B_list):
    """Block structured orthogonal factorization (BSOF), a block structured QR
    factorization algorithm introduced in Wright, S.J., 1992.
    Args:
        B_list: length L list of sub-diagonal and boundary blocks of p-cyclic matrix

    Returns:
        Q_list, length L-1 list of (2N,2N) orthogonal matrices
        R_dict, size (3L-3) dictionary of R matrix block entries.
                key = tuple of entry coordinates
                value = (N,N) np array

    TODO: Better return structure for R
          Set to 0 the very small elements of Q and R?
    """
    # input shapes
    L = len(B_list)
    N, N = B_list[0].shape

    # output accumulators
    Q_list = []
    R_dict = {}

    # initial step
    ###tempA = A_list[0]
    tempA = np.identity(N)
    tempB = B_list[-1]
    # first L-2 blocks
    for k in range(L - 2):
        # compute QR, set R diagonal element
        tall_mat = np.concatenate((tempA, B_list[k]), axis=0)
        Q, R = la.qr(tall_mat)
        # plt.figure();plt.imshow(R);plt.colorbar()
        Q_list.append(Q)
        R_dict[(k, k)] = R[:N, :N]  # take top part only
        # update tempA,tempB, set off diagonal R elements
        rhs = np.zeros((2 * N, 2 * N))
        #####rhs[N:,:N] = A_list[k+1]; rhs[:N,N:] = tempB;
        rhs[N:, :N] = np.identity(N)
        rhs[:N, N:] = tempB
        lhs = Q.T @ rhs
        tempA = lhs[N:, :N]
        tempB = lhs[N:, N:]
        R_dict[(k, k + 1)] = lhs[:N, :N]
        R_dict[(k, L - 1)] = lhs[:N, N:]
    # last QR block
    rhs_row1 = np.concatenate((tempA, tempB), axis=1)
    #####rhs_row2 = np.concatenate((B_list[-2], A_list[-1]), axis=1)
    rhs_row2 = np.concatenate((B_list[-2], np.identity(N)), axis=1)
    rhs = np.concatenate((rhs_row1, rhs_row2), axis=0)
    Q, R = la.qr(rhs)
    R_dict[(L - 2, L - 2)] = R[:N, :N]
    R_dict[(L - 2, L - 1)] = R[:N, N:]
    R_dict[(L - 1, L - 1)] = R[N:, N:]
    Q_list.append(Q)

    # check output shapes
    assert len(Q_list) == L - 1 and len(R_dict) == 3 * L - 3

    return Q_list, R_dict


def Invert_R_Row_BBS(R, N, L):
    """Get X = R^{-1} based on RX = I using Row Block Back Substitution.
    Not exactly same as paper description.

    Args:
        R: Full R matrix (NL,NL)
        N: Size of each block
        L: Number of blocks

    Returns:
        X = full R inverse, (NL,NL)
    """
    X = np.zeros((L * N, L * N))
    # last (3N,3N) block is full, directly solve
    X[(L - 3) * N :, (L - 3) * N :] = la.solve_triangular(
        R[(L - 3) * N :, (L - 3) * N :], np.identity(3 * N)
    )
    # Row block back substitution
    for i in range(L - 3):
        # print(f"row {i}")
        # remaining diagonal (i,i) block shape (N,N)
        X[i * N : (i + 1) * N, i * N : (i + 1) * N] = la.solve_triangular(
            R[i * N : (i + 1) * N, i * N : (i + 1) * N], np.identity(N)
        )

    # last block column, incomplete RHS
    X[: (L - 3) * N, (L - 1) * N :] = (
        -R[: (L - 3) * N, (L - 1) * N :] @ X[(L - 1) * N :, (L - 1) * N :]
    )
    for i in range(L - 4, -1, -1):
        # print(f"modify row {i} in place")
        # complete rhs
        X[i * N : (i + 1) * N, (i + 1) * N :] -= (
            R[i * N : (i + 1) * N, (i + 1) * N : (i + 2) * N]
            @ X[(i + 1) * N : (i + 2) * N, (i + 1) * N :]
        )
        # solve using X[i,i] = Rinv[i,i]
        X[i * N : (i + 1) * N, (i + 1) * N :] = (
            X[i * N : (i + 1) * N, i * N : (i + 1) * N]
            @ X[i * N : (i + 1) * N, (i + 1) * N :].copy()
        )
    return X


def Apply_Orth_Fact(Rinv, Q_list, N, L):
    """Produce R^{-1} @ Q^T, inverse of full p-cyclic matrix.
    Apply Q to pairs of columns of R^{-1}, back to front.
    TODO: use paper trick to reduce flops

    Args:
        Rinv, start with full R^{-1}, modified in place, end at R^{-1} @ Q^T
        Q_list, length L-1 list of (2N,2N) orthogonal matrices
        N: Size of each block
        L: Number of blocks

    Returns:
        None

    """
    for k in range(L - 2, -1, -1):
        Rinv[:, k * N : (k + 2) * N] = Rinv[:, k * N : (k + 2) * N] @ Q_list[k].T
    return None


def Get_Full_O_Inv(invexpdtK, expVData):
    """Get full O^{-1} which gives all unequal time GFs

    Args:
        expVData: (N,L) matrix, each col reps a diagonal mat

    Returns:
        (NL,NL) mat, full O^{-1}"""
    N, L = expVData.shape
    blist = Make_B_List(invexpdtK, expVData)
    # Block structured orthogonal factorization (BSOF)
    qlist, rdict = Block_Struct_Orth_Fact(blist)
    # X = R^{-1}
    rfull = Get_R_Full(rdict, N, L)
    xfull = Invert_R_Row_BBS(rfull, N, L)
    # Apply Q^T to get O^{-1} = R^{-1} @ Q^T, X modified in place
    Apply_Orth_Fact(xfull, qlist, N, L)

    return xfull


if __name__ == "__main__":
    print("test BSOFI real on small problem")
    N = 4
    L = 5
    invexpdtK = np.identity(N)
    expVuData = np.random.random((N, L))

    Gu_list = Make_B_List(invexpdtK, expVuData)
    qlist, rdict = Block_Struct_Orth_Fact(Gu_list)
    qfull = Show_Q_Full(qlist, N, L)
    qinvfull = Show_Q_Inv_Full(qlist, N, L)

    plt.figure()
    plt.imshow(qinvfull - qfull.T)
    plt.colorbar()
    plt.title("Q^{-1} - Q^T")

    rfull = Show_R_Full(rdict, N, L)
    ofull = Show_P_Cyclic_Full(Gu_list)

    plt.figure()
    plt.imshow(qfull @ rfull)
    plt.colorbar()
    plt.title("Reconstructed Full P-cyclic matrix")

    plt.figure()
    plt.imshow(qfull @ rfull - ofull)
    plt.colorbar()
    plt.title("QR - O")

    xfull = Invert_R_Row_BBS(rfull, N, L)
    plt.figure()
    plt.imshow(xfull)
    plt.colorbar()
    plt.title("Full R^{-1} matrix")

    Apply_Orth_Fact(xfull, qlist, N, L)
    plt.figure()
    plt.imshow(xfull)
    plt.colorbar()
    plt.title("Full R^{-1}Q^T matrix")

    plt.figure()
    plt.imshow(xfull @ ofull)
    plt.colorbar()
    plt.title("Check inversion correct")

    plt.show()
