import marimo

__generated_with = "0.12.4"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    from scipy import linalg as la
    import matplotlib.pyplot as plt
    from itertools import product
    import dqmc_util as du

    import sys

    sys.path.insert(0, "../../dqmc-all-backup/util")
    sys.path.insert(0, "../../Code-DQMC-Data-Analysis")
    print(sys.path)
    import util  # Edwin's util file
    import data_analysis  # data analysis files

    # from bsofi_real import Get_Full_O_Inv #BSOFI get full Inverse of GF block matrix
    return data_analysis, du, la, np, plt, product, sys, util


@app.cell
def _(du, la, np):
    # Input Parameters
    Nx = 4
    Ny = 4
    U = -4
    t = 1
    tp = 0
    mu = -4
    if tp != 0:
        bps = 4
    else:
        bps = 2

    beta = 2  # inverse temperature
    L = np.maximum(
        5, np.ceil(beta / 0.1), casting="unsafe", dtype=int
    )  # number of time slices

    n_sweep_warm = 200
    n_sweep_meas = 200
    # number of time slice wraps between equal time measurements
    period_eqlt = 5
    # number of full sweeps between unequal time measurements
    period_uneqlt = 0
    # number of l-wraps between calculating GF from scratch
    wrap_every = 5
    # number of B matrices to directly multiply together
    batch_size = 2 * wrap_every

    # Derived constants
    N = Nx * Ny
    # total number of grid points
    dt = beta / L
    # imaginary time interval
    print(f"Running Markov chain for {Nx}*{Ny} Lattice")
    print(f"Interactions:U = {U}, t = {t}, t' = {tp}, mu = {mu}, bps={bps}")
    print("Temerature: beta = ", beta, ", dt = ", dt, ", L = ", L)
    print(
        "Recompute GF every", wrap_every, "l-wraps;", batch_size, "matmuls between QRs"
    )
    print(
        "Measure eqlt every",
        period_eqlt,
        "l-wraps; measure uneqlt every",
        period_uneqlt,
        "full sweeps",
    )


    lmbd = np.arccosh(np.exp(np.abs(U) * dt * 0.5))  # lambda in H-S transform
    print("lambda = ", lmbd)

    kmatrix = du.makeKMatrix(Nx, Ny, t=t, tp=tp, mu=mu)
    invexpdtK = la.expm(-dt * kmatrix)
    expdtK = la.expm(dt * kmatrix)
    # this matrix used only in G-fun wrapping
    invexpdt_halfK = la.expm(-0.5 * dt * kmatrix)
    # symmetric Trotter half wrap only before measurmeent
    expdt_halfK = la.expm(0.5 * dt * kmatrix)
    # symmetric Trotter half wrap only before measurement

    # Initialize Auxillary H-S field as N-by-L 2D array.
    # Each column is a different time slice
    S = np.random.choice([-1, 1], (N, L))

    params = {
        "Nx": Nx,
        "Ny": Ny,
        "N": N,
        "U": U,
        "t": t,
        "tp": tp,
        "mu": mu,
        "beta": beta,
        "L": L,
        "dt": dt,
        "lmbd": lmbd,
        "bps": bps,
        "n_sweep_warm": n_sweep_warm,
        "n_sweep_meas": n_sweep_meas,
        "period_eqlt": period_eqlt,
        "period_uneqlt": period_uneqlt,
        "wrap_every": wrap_every,
        "batch_size": batch_size,
    }

    # measurement accumulators
    eqlt_data = {
        "n_sample": np.zeros(1, dtype=int),
        "sign": np.zeros(1),
        "density": np.zeros(N),  # allow site dependence
        "double_occ": np.zeros(N),  # allow site dependence
        "zz": np.zeros((N, N)),  # before compression, z spin correlator
        "xx": np.zeros(
            (N, N)
        ),  # before compression, x or y spin correlator: they are exactly equivalent
        "nn": np.zeros((N, N)),  # before compression, charge correlator
        "pair_sw": np.zeros((N, N)),  # before compression, spin wave correlator
        "g00": np.zeros((N, N)),  # before compression, equal time GF
    }

    # lookup table to avoid recomputing neighbor indices
    vec2ij_dict = du.Make_Vec_To_IJ_Dict(Nx, Ny)
    # btmap = Bond_Type_Dir(bps)
    # dmdict = Delta_Mask_Dict(btmap)
    return (
        L,
        N,
        Nx,
        Ny,
        S,
        U,
        batch_size,
        beta,
        bps,
        dt,
        eqlt_data,
        expdtK,
        expdt_halfK,
        invexpdtK,
        invexpdt_halfK,
        kmatrix,
        lmbd,
        mu,
        n_sweep_meas,
        n_sweep_warm,
        params,
        period_eqlt,
        period_uneqlt,
        t,
        tp,
        vec2ij_dict,
        wrap_every,
    )


@app.cell
def _(U, run_attractive, run_repulsive):
    if U >= 0:
        run_repulsive()
    else:
        run_attractive()
    return


@app.cell
def _(eqlt_data, np):
    print("n_sample", eqlt_data["n_sample"])
    print("sign", eqlt_data["sign"] / eqlt_data["n_sample"])
    print("density", np.mean(eqlt_data["density"]) / eqlt_data["sign"])
    print("double_occ", np.mean(eqlt_data["double_occ"]) / eqlt_data["sign"])
    return


@app.cell
def _(
    L,
    N,
    S,
    U,
    batch_size,
    du,
    eqlt_data,
    expdtK,
    expdt_halfK,
    invexpdtK,
    invexpdt_halfK,
    lmbd,
    n_sweep_meas,
    n_sweep_warm,
    np,
    period_eqlt,
    wrap_every,
):
    def run_repulsive():
        assert U >= 0
        expVuData = du.makeExpLamSData(S, lmbd, spin=1)
        expVdData = du.makeExpLamSData(S, lmbd, spin=-1)
        Gu, _ = du.makeGMatrix_QR(expVuData, invexpdtK, 0, batch_size)
        Gd, _ = du.makeGMatrix_QR(expVdData, invexpdtK, 0, batch_size)
        for n in range(n_sweep_meas + n_sweep_warm):
            # Monitor progress
            if n % 200 == 0:
                print("Sweep # ", n)
            for l in range(L):
                """take equal time measurements more frequently than full sweeps"""
                if n >= n_sweep_warm and l % period_eqlt == 0:
                    # Recompute G-Funs before taking measurements?
                    # use l = 0 or current l?
                    Gu_00, sgn_up = du.makeGMatrix_QR(
                        expVuData, invexpdtK, l_shift=0, batch_size=batch_size
                    )  # 10% time
                    Gd_00, sgn_dn = du.makeGMatrix_QR(
                        expVdData, invexpdtK, l_shift=0, batch_size=batch_size
                    )  # 10% time
                    Gu_00 = du.half_wrap(Gu_00, expdt_halfK, invexpdt_halfK)
                    Gd_00 = du.half_wrap(Gd_00, expdt_halfK, invexpdt_halfK)
                    # TODO: check that this is OK way to get sign? O(N^3)
                    sgn = sgn_up * sgn_dn
                    # if sgn < 0:
                    #    print('sign < 0', sgn_up,sgn_dn)

                    # bookkeeping
                    eqlt_data["n_sample"] += 1
                    eqlt_data["sign"] += sgn
                    eqlt_data["g00"] += (
                        sgn * 0.5 * (Gu_00 + Gd_00)
                    )  # avg over spin up and dn domains

                    # diagonal elements of GFs
                    Gu_00_diag = np.diagonal(Gu_00)
                    Gd_00_diag = np.diagonal(Gd_00)
                    diag_sum = Gu_00_diag + Gd_00_diag
                    diag_diff = Gu_00_diag - Gd_00_diag

                    # single site measurements, vectorized
                    # TODO assumed round off error while accumulating measurements is neglegible
                    eqlt_data["density"] += sgn * (2 - diag_sum)
                    eqlt_data["double_occ"] += sgn * (
                        (1 - Gu_00_diag) * (1 - Gd_00_diag)
                    )

                    # two site measurements. vectorized, acting on N*N matrix
                    t1_zz = np.outer(diag_diff, diag_diff)
                    t1_charge = np.outer(2 - diag_sum, 2 - diag_sum)
                    t3 = Gu_00.T * Gu_00 + Gd_00.T * Gd_00  # a mixed product
                    t3_xx = Gu_00.T * Gd_00 + Gd_00.T * Gu_00  # another mixed product
                    eqlt_data["zz"] += sgn * 0.25 * (t1_zz + (np.diag(diag_sum) - t3))
                    eqlt_data["xx"] += sgn * 0.25 * (np.diag(diag_sum) - t3_xx)
                    eqlt_data["nn"] += sgn * (t1_charge + (np.diag(diag_sum) - t3))
                    eqlt_data["pair_sw"] += sgn * (Gu_00 * Gd_00)

                if l % wrap_every == 0:
                    # recompute GFs from scratch, reduce roundoff err from wrapping
                    # Don't need determinant here
                    Gu, _ = du.makeGMatrix_QR(
                        expVuData,
                        invexpdtK,
                        l_shift=l,
                        batch_size=batch_size,
                        check=False,
                    )  # 10% time
                    Gd, _ = du.makeGMatrix_QR(
                        expVdData,
                        invexpdtK,
                        l_shift=l,
                        batch_size=batch_size,
                        check=False,
                    )  # 10% time

                """ Update: after maybe taking measurement, go through every 
                [i,l] pair and propose flipping s[i,l]. 
                Result: A S-field drawn with probability propto w'_s = |w_s|"""
                for i in range(N):
                    # Implicitly propose flipping s[i,l], calculate accept ratio
                    detu = 1 + (1 - Gu[i, i]) * (np.exp(-2 * lmbd * S[i, l]) - 1)
                    detd = 1 + (1 - Gd[i, i]) * (np.exp(2 * lmbd * S[i, l]) - 1)
                    alpha = detu * detd
                    # if alpha<0:
                    # print("alpha<0",alpha)
                    # phase = alpha/np.absolute(alpha)
                    r = np.random.random()
                    # Probability Min(1,abs(alpha)) of accepting flip
                    # If accept, update Gfuns, S[i,l], expVu[i,l],expVd[i,l]
                    # If reject, do nothing, look at next entry of S
                    if r <= np.absolute(alpha):
                        # Update G-funs using old s-field entries
                        # Note : have to copy arrays to avoid modifying G-Funs prematurely
                        col = Gu[:, i].copy()
                        col[i] = col[i] - 1
                        # i-th column of (Gup - delta)
                        row = Gu[i, :].copy()
                        # ith row of Gup
                        # TODO: Delayed update increase performance? Eq. (23) has redundant zz
                        # 10% time
                        mat = np.outer(col, row)
                        Gu = Gu + (np.exp(-2 * lmbd * S[i, l]) - 1) * mat / detu

                        col = Gd[:, i].copy()
                        col[i] = col[i] - 1
                        # i-th column of (Gdn - delta)
                        row = Gd[i, :].copy()
                        # ith row of Gdn
                        # TODO: Delayed update increase performance?
                        # 10% time
                        mat = np.outer(col, row)
                        Gd = Gd + (np.exp(2 * lmbd * S[i, l]) - 1) * mat / detd

                        # Update S matrix
                        S[i, l] *= -1

                        # Update [i,l] entry of ExpLamSData_up, ExpLamSData_dn
                        expVuData[i, l] = np.exp(lmbd * S[i, l])
                        expVdData[i, l] = np.exp(-lmbd * S[i, l])

                # After each l slice, wrap the B matrices.
                # We do this in order to continue using nice formulas that allow us to quickly
                #  calculate alpha and update G-funs.
                # After l runs through 0...L-1, G-fun returns to textbook form,
                #  with some roundoff errors from wrapping.
                diag_u = expVuData[:, l]
                diag_d = expVdData[:, l]
                # == inv(diag_up)

                Blu = invexpdtK * diag_u
                invBlu = diag_d[:, None] * expdtK
                Gu = Blu @ Gu @ invBlu

                Bld = invexpdtK * diag_d
                invBld = diag_u[:, None] * expdtK
                Gd = Bld @ Gd @ invBld
    return (run_repulsive,)


@app.cell
def _(
    L,
    N,
    S,
    U,
    batch_size,
    du,
    eqlt_data,
    expdt_halfK,
    invexpdtK,
    invexpdt_halfK,
    lmbd,
    n_sweep_meas,
    n_sweep_warm,
    np,
    period_eqlt,
):
    # RUN MARKOV CHAIN
    def run_attractive():
        assert U <= 0
        # print(np.sum(S))
        expVuData = du.makeExpLamSData(S, lmbd, spin=1)
        expVdData = du.makeExpLamSData(S, lmbd, spin=1)
        Gu, _ = du.makeGMatrix_QR(expVuData, invexpdtK, 0, batch_size)
        Gd, _ = du.makeGMatrix_QR(expVdData, invexpdtK, 0, batch_size)
        assert np.allclose(Gu, Gd)
        assert np.allclose(expVuData, expVdData)
        for n in range(n_sweep_meas + n_sweep_warm):
            # print(np.sum(S))
            # Monitor progress
            if n % 100 == 0:
                print("Sweep # ", n)
            for l in range(L):
                """take equal time measurements more frequently than full sweeps"""
                if n >= n_sweep_warm and l % period_eqlt == 0:
                    # Recompute G-Funs before taking measurements?
                    # use l = 0 or current l?
                    Gu_00, sgn_up = du.makeGMatrix_QR(
                        expVuData, invexpdtK, l_shift=0, batch_size=batch_size
                    )  # 10% time
                    Gd_00, sgn_dn = du.makeGMatrix_QR(
                        expVdData, invexpdtK, l_shift=0, batch_size=batch_size
                    )  # 10% time
                    Gu_00 = du.half_wrap(Gu_00, expdt_halfK, invexpdt_halfK)
                    Gd_00 = du.half_wrap(Gd_00, expdt_halfK, invexpdt_halfK)
                    # TODO: check that this is OK way to get sign? O(N^3)
                    sgn = sgn_up * sgn_dn
                    # if sgn < 0:
                    #    print('sign < 0', sgn_up,sgn_dn)

                    # bookkeeping
                    eqlt_data["n_sample"] += 1
                    eqlt_data["sign"] += sgn
                    eqlt_data["g00"] += (
                        sgn * 0.5 * (Gu_00 + Gd_00)
                    )  # avg over spin up and dn domains

                    # diagonal elements of GFs
                    Gu_00_diag = np.diagonal(Gu_00)
                    Gd_00_diag = np.diagonal(Gd_00)
                    diag_sum = Gu_00_diag + Gd_00_diag
                    diag_diff = Gu_00_diag - Gd_00_diag

                    # single site measurements, vectorized
                    # TODO assumed round off error while accumulating measurements is neglegible
                    eqlt_data["density"] += sgn * (2 - diag_sum)
                    eqlt_data["double_occ"] += sgn * (
                        (1 - Gu_00_diag) * (1 - Gd_00_diag)
                    )

                    # two site measurements. vectorized, acting on N*N matrix
                    t1_zz = np.outer(diag_diff, diag_diff)
                    t1_charge = np.outer(2 - diag_sum, 2 - diag_sum)
                    t3 = Gu_00.T * Gu_00 + Gd_00.T * Gd_00  # a mixed product
                    t3_xx = Gu_00.T * Gd_00 + Gd_00.T * Gu_00  # another mixed product
                    eqlt_data["zz"] += sgn * 0.25 * (t1_zz + (np.diag(diag_sum) - t3))
                    eqlt_data["xx"] += sgn * 0.25 * (np.diag(diag_sum) - t3_xx)
                    eqlt_data["nn"] += sgn * (t1_charge + (np.diag(diag_sum) - t3))
                    eqlt_data["pair_sw"] += sgn * (Gu_00 * Gd_00)

                # recompute GFs from scratch, reduce roundoff err from wrapping
                Gu, _ = du.makeGMatrix_QR(
                    expVuData,
                    invexpdtK,
                    l_shift=l,
                    batch_size=batch_size,
                    check=False,
                )
                Gd, _ = du.makeGMatrix_QR(
                    expVdData,
                    invexpdtK,
                    l_shift=l,
                    batch_size=batch_size,
                    check=False,
                )
                assert np.allclose(Gu, Gd)
                assert np.allclose(expVuData, expVdData)

                """ Update: after maybe taking measurement, go through every 
                [i,l] pair and propose flipping s[i,l]. 
                Result: A S-field drawn with probability propto w'_s = |w_s|"""
                for i in range(N):
                    # Implicitly propose flipping s[i,l], calculate accept ratio
                    detu = 1 + (1 - Gu[i, i]) * (np.exp(-2 * lmbd * S[i, l]) - 1)
                    detd = 1 + (1 - Gd[i, i]) * (np.exp(-2 * lmbd * S[i, l]) - 1)
                    alpha = detu * detd * (np.exp(2 * lmbd * S[i, l]))
                    print(detu,detd,alpha)
                    r = np.random.random()
                    # Probability Min(1,abs(alpha)) of accepting flip
                    # If accept, update Gfuns, S[i,l], expVu[i,l],expVd[i,l]
                    # If reject, do nothing, look at next entry of S
                    if r <= np.absolute(alpha):
                        # Update S matrix
                        S[i, l] *= -1
                        # Update [i,l] entry of ExpLamSData_up, ExpLamSData_dn
                        expVuData[i, l] = np.exp(lmbd * S[i, l])
                        expVdData[i, l] = np.exp(lmbd * S[i, l])
    return (run_attractive,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
