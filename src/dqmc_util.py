import numpy as np
from scipy import linalg as la
from scipy.sparse import coo_matrix


def makeKMatrix(Nx: int, Ny: int, t: float = 1, tp: float = 0, mu: float = 0):
    """Generate kinetic matrix for 2D square Nx-by-Ny lattice, periodic bdy conditions.
    Args:
        t = NN hopping strength
        tp = NNN hopping stength
        mu = Chemical potential
        Nx, Ny = Lattice dimensions
    Returns:
        A shape (N,N) float numpy array, N = Nx*Ny.
        Matrix created is deliberately not sparse."""

    # Initialize vectors of row and column indices, matrix data
    i, j = np.empty(0, dtype=int), np.empty(0, dtype=int)
    data = np.empty(0, dtype=float)

    # Look at each lattice pt, find nearest neighbors
    for y in range(Ny):
        for x in range(Nx):
            # total index n
            n = x + Nx * y
            # neighbors, with periodic B.C.
            neighbors = [
                ((x + 1) % Nx, y),
                ((x - 1) % Nx, y),
                (x, (y + 1) % Ny),
                (x, (y - 1) % Ny),
            ]
            next_neighbors = [
                ((x + 1) % Nx, (y + 1) % Ny),
                ((x - 1) % Nx, (y + 1) % Ny),
                ((x + 1) % Nx, (y - 1) % Ny),
                ((x - 1) % Nx, (y - 1) % Ny),
            ]
            # get total indices of nearest neighbors
            for pt in neighbors:
                i = np.append(i, pt[0] + Nx * pt[1])
            data = np.append(data, (-t) * np.ones(len(neighbors), dtype=float))
            # get total indices of next-nearest neighbors
            for pt in next_neighbors:
                i = np.append(i, pt[0] + Nx * pt[1])
            data = np.append(data, (-tp) * np.ones(len(next_neighbors), dtype=float))
            # everything stays in n-th column
            j = np.append(
                j, n * np.ones(len(neighbors) + len(next_neighbors), dtype=int)
            )

            # chemical potential contribes only to diagonal elements
            i = np.append(i, n)
            j = np.append(j, n)
            data = np.append(data, -mu)

    # Construction
    kmatrix = coo_matrix((data, (i, j)), shape=(Nx * Ny, Nx * Ny)).toarray()
    return kmatrix


def makeExpLamSData(S, lmbd, spin):
    """Construct the length N vector exp(spin*lmbd*S[:,l]) for every single l = 0...L-1.
    Each length N vector stands for a diagonal matrix of size N-by-N.
    There are L total such matrices.
    For efficiency, store information for all L matrices in one N-by-L 2D array.
    Each column represents a different diagonal matrix expV

    Args:
        S = (N,L) int np array, current H-S field configuration
        spin = 1 or -1, representing up or down spin

    Returns:t
        (N,L) np array storing exp(V(l,spin)) data
    """

    assert spin == 1 or spin == -1, "Spin value must be +1 or -1"
    N, L = S.shape
    diags = np.zeros((N, L), dtype=float)
    for l in range(L):
        diags[:, l] = np.exp(lmbd * S[:, l] * spin)

    return diags


def makeGMatrix_QR(expVData, invexpdtK, l_shift=0, batch_size=10, check=False):
    """Calculate equal-time GF from scratch using Pivoted-QR to avoid
    numerical error accumulation. Method described in tutorial.

    Args:
        expVData: N-by-L 2D array, V matrix data (depends on current H-S config).
                  Each column is exp(spin*lmbd*S[:,l]).
        l_shift: Account for change in GF defn after l wraps.
                 Defaults to 0, no wrapping
        batch_size: Number of B matrices to directly multiply together.
                    Defaults to 10, a typical safe value
        check: bool, whether to check if batch_size is safe.
               Defaults to False in simulation

    Returns:
        A (N,N) np-array, equal-time GF matrix
        +1 or -1, sign of determinant of this GF matrix
    """

    # dimensions
    N, L = expVData.shape

    # account for l_shift wraps: Roll forward, so that we start with
    # the l-th column in zero-th column
    expVData = np.roll(expVData, -l_shift, axis=1)

    # divide into batches to reduce number of QR decomps we have to do
    num_batches = L // batch_size
    Bmats = []
    # form every batch by normal matrix multiplication
    for j in range(num_batches):
        # split along axis 1 (L-axis) into batches
        data_j = expVData[:, j * batch_size : (j + 1) * batch_size]
        # start with identity
        Bmat = np.identity(N, dtype=float)
        # form B[(j+1)*b]....B[j*b] by multiplying right to left
        for l in range(data_j.shape[1]):
            Bmat = (invexpdtK * data_j[:, l]) @ Bmat
        Bmats.append(Bmat)
    # if no even split, then one extra batch with fewer matrices
    if L % batch_size != 0:
        data_j = expVData[:, (num_batches * batch_size) :]
        Bmat = np.identity(N, dtype=float)
        for l in range(data_j.shape[1]):
            Bmat = (invexpdtK * data_j[:, l]) @ Bmat
        Bmats.append(Bmat)
        num_batches += 1  # account for extra batch

    # Now, go through Bmats and calculate matrix products
    Q, R, perm = la.qr(Bmats[0], pivoting=True)
    inv_perm = np.argsort(perm)  # inverse of permutation: P^{-1} = P^T.
    D_arr = np.diagonal(R)  # extract diagonal
    T = 1 / D_arr[:, None] * R[:, inv_perm]  # T = inv(D) @ R @ P^T
    # assert (Q * D_arr) @ T == Bmats[0]

    for l in range(1, num_batches):
        # l-th B matrix batch
        C = Bmats[l] @ Q * D_arr
        Q, R, perm = la.qr(C, pivoting=True)
        inv_perm = np.argsort(perm)
        D_arr = np.diagonal(R)
        T = ((1 / D_arr[:, None]) * R[:, inv_perm]) @ T
        # assert (Q * D_arr) @ T == Bmats[l]...Bmats[0]

    # post processing
    # elementwise product, Db*Ds = D
    Db_arr = np.zeros(D_arr.shape)
    Ds_arr = np.zeros(D_arr.shape)
    for i in range(N):
        if np.abs(D_arr[i]) > 1:
            Db_arr[i] = D_arr[i]
            Ds_arr[i] = 1
        else:
            Db_arr[i] = 1
            Ds_arr[i] = D_arr[i]

    # result
    g = la.solve(
        1 / Db_arr[:, None] * Q.T + Ds_arr[:, None] * T, 1 / Db_arr[:, None] * Q.T
    )

    # det(Db)/|det(Db)|
    Db_sign = np.prod(np.sign(Db_arr))
    # det(stuff)/|det(stuff)|
    extra = la.solve(Q, 1 / Db_arr[:, None] * Q.T + Ds_arr[:, None] * T)
    detextra = np.linalg.det(extra)
    sign = Db_sign * detextra / np.abs(detextra)
    # sign2,_ = np.linalg.slogdet(g)
    # if sign != sign2:
    # print(sign,sign2)

    """check == False for simulations, only for experiment"""
    if check:
        print("checking if batch size", batch_size, "is safe")
        # start with B_0
        Bmat_0 = invexpdtK * expVData[:, 0]
        Q, R, perm = la.qr(Bmat_0, pivoting=True)
        inv_perm = np.argsort(perm)
        D_arr = np.diagonal(R)
        T = 1 / D_arr[:, None] * R[:, inv_perm]  # T = inv(D) @ R @ P^T

        # Matrix products via QR trick
        for l in range(1, L):
            # l-th B matrix
            Bmat_l = invexpdtK * expVData[:, l]
            C = Bmat_l @ Q * D_arr
            Q, R, perm = la.qr(C, pivoting=True)
            inv_perm = np.argsort(perm)
            D_arr = np.diagonal(R)
            T = ((1 / D_arr[:, None]) * R[:, inv_perm]) @ T

        # post processing
        Db_arr = np.zeros(D_arr.shape)
        Ds_arr = np.zeros(D_arr.shape)
        for i in range(N):
            if np.abs(D_arr[i]) > 1:
                Db_arr[i] = D_arr[i]
                Ds_arr[i] = 1
            else:
                Db_arr[i] = 1
                Ds_arr[i] = D_arr[i]

        # check result
        g_check = la.solve(
            1 / Db_arr[:, None] * Q.T + Ds_arr[:, None] * T, 1 / Db_arr[:, None] * Q.T
        )

        print(
            "deviation from when every B matrix is factored =",
            la.norm(g - g_check, ord=np.inf),
        )
        if la.norm(g - g_check, ord=np.inf) > 1e-8:
            raise RuntimeError("QR batch size for GF calculation is unsafe")

    return g, sign


def half_wrap(G, expdt_halfK, invexpdt_halfK):
    """Produce symmetric Trotter decomp version of GF from asymmetric version

    Args:
        G, unwrapped GF matrix used in simulation?? TODO: check we can use unwrapped GF
        for everything else and just do wrapping before measurement. YES
        TODO: how much overhead do these functions add? Make them inline

    Returns:
        (N,N) np array, wrapped GF matrix"""
    return expdt_halfK @ G @ invexpdt_halfK


def ij_pairs_list(Nx: int, Ny: int, dx, dy):
    """Produce Gblock indices (ilist,jlist) for fixed hopping distance vector (site_j-site_i)

    Args:
        Nx, Ny: dimension of lattice
        dx, dy: desired hopping direction vector
    Out:
        A tuple of lists (ilist, jlist) representing row and column indices
        of N*N Gblock matrix we want to extract
    """
    ilist, jlist = [], []

    # starting location (x,y)
    for y in range(Ny):
        for x in range(Nx):
            n_init = x + Nx * y
            # termination location (x+dx,y+dy) periodic BC
            pt_end = ((x + dx) % Nx, (y + dy) % Ny)
            n_end = pt_end[0] + Nx * pt_end[1]
            ilist.append(n_init)
            jlist.append(n_end)
    return ilist, jlist


def Make_Vec_To_IJ_Dict(Nx, Ny):
    """Make lookup table/dictionary mapping distance (dx,dy) to lists of
    total i,j indices in Gblock matrix separated by distance (dx,dy)
    Args:
        Nx, Ny: dimension of lattice
    Out:
        A nested dictionary, [dx][dy]-th entry is a tuple of lists
            (ilist, jlist) representing row and column indices of N*N
            Gblock matrix that are separated by same distance (dx,dy).
            Each list is length N"""
    table = {}
    for dx in range(Nx):
        table[dx] = {}
        for dy in range(Ny):
            indices = ij_pairs_list(Nx, Ny, dx, dy)
            table[dx][dy] = indices

    return table


def Make_IJ_To_Vec_Dict(Nx, Ny):
    """Make lookup table/dictionary, mapping total index of Gblock[i][j] to
    site_j-site_i distance tuple (dx,dy). This is a many to 1 mapping,
    N (i,j) pairs will map to same (dx,dy) tuple
    Args:
        Nx, Ny: dimension of lattice
    Out:
        A nested dictionary, [i][j]-th entry is a tuple (dx,dy) representing
        spatial distance vector (site_j - site_i) considering periodic wrapping"""
    N = Nx * Ny
    table = {}
    for i in range(N):
        table[i] = {}
        for j in range(N):
            ix = i % Nx
            iy = i // Nx
            jx = j % Nx
            jy = j // Nx
            table[i][j] = ((jx - ix) % Nx, (jy - iy) % Ny)

    return table


def Trans_Symm_Compress(vec2ij_dict, Gt0s):
    """Apply translational symmetry: G(pt_i,pt_j) = G(pt_j-pt_i). Average over elements with
    same spatial separation in raw Gt0 measurements to get compressed measurement values
    Input:
        (L,N,N) np-array representing first block column of O^{-1}. Usually outout of Get_Gt0s
    Returns:
        (L,Nx,Ny) np-array: compressd measurements, each page is F ordered to represent G[]
    """
    assert Gt0s.shape[1] == Gt0s.shape[2] and Gt0s.shape[1] == Nx * Ny
    L, N, _ = Gt0s.shape
    # initialize compressed measurement
    compressed = np.zeros((L, Nx, Ny))
    for dx in range(Nx):
        for dy in range(Ny):
            # avoid recomputing neighbor indices....
            indices = vec2ij_dict[dx][dy]
            for l in range(L):
                # n = dx+Nx*dy
                compressed[l, dx, dy] = np.mean(Gt0s[l, indices[0], indices[1]])

    # if compressing equal time correlator, then remove extra dim to return (Nx,Ny) 2D array
    return np.squeeze(compressed)


def permute_col(plist, G):
    """convenience function for bond-bond measurement
    TODO: how much overhead do these functions add? Make them inline"""
    return G[:, plist]


def permute_row(plist, G):
    """convenience function for bond-bond measurement Make them inline"""
    return G[plist, :]
