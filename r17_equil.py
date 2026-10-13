# -*- coding: utf-8 -*-
"""Admission conditions (1) and (3):
 (3) REAL equilibrated flux: solve the RT0 mixed saddle-point problem, use the constructed
     sigma~ in the guaranteed estimate eta_guar (Neumann problem, cos*cos manufactured solution).
 (1) 2D H(curl) edge-element (lowest order Whitney) discretisation of the nonlinear
     curl-curl problem with tree-cotree gauge + estimator with tangential jumps.
"""
import json, math, sys
import numpy as np

sys.path.insert(0, r"C:\Users\artem\SeekDeep\Workspaces\1\work")
import importlib.util
spec = importlib.util.spec_from_file_location("r17", r"C:\Users\artem\SeekDeep\Workspaces\1\work\r17_numerics.py")
r17 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r17)

OUT = r"C:\Users\artem\SeekDeep\Workspaces\1\work\r17_equil.json"

# ------------------------------------------------------------------ Neumann exact solution
def uN(x, y):
    return math.cos(math.pi * x) * math.cos(math.pi * y)


def gN(x, y):
    return (-math.pi * math.sin(math.pi * x) * math.cos(math.pi * y),
            -math.pi * math.cos(math.pi * x) * math.sin(math.pi * y))


def fluxN(x, y, alpha, p):
    gx, gy = gN(x, y)
    s = 1.0 + gx * gx + gy * gy
    c = alpha * s ** ((p - 2.0) / 2.0)
    return c * gx, c * gy


def fN(x, y, alpha_fn, p, h=1e-5):
    fxp = fluxN(x + h, y, alpha_fn(x + h, y), p); fxm = fluxN(x - h, y, alpha_fn(x - h, y), p)
    fyp = fluxN(x, y + h, alpha_fn(x, y + h), p); fym = fluxN(x, y - h, alpha_fn(x, y - h), p)
    return -((fxp[0] - fxm[0]) + (fyp[1] - fym[1])) / (2 * h)


# ------------------------------------------------------------------ scalar Neumann FEM (P1)
def solve_neumann(N, kappa, p, alpha_fn, tol=1e-10):
    coords, tris = r17.build_mesh(N)
    grads, areas, cent = r17.tri_geometry(coords, tris)
    n = len(coords)
    alpha_e = np.array([alpha_fn(cx, cy) for cx, cy in cent])
    fq = np.array([fN(cx, cy, alpha_fn, p) for cx, cy in cent])
    u = np.zeros(n)

    def proj(v):
        return v - v.mean()

    for it in range(80):
        R, G, flux = r17.residual(u, tris, grads, areas, cent, alpha_e, p, fq)
        Rn = float(np.linalg.norm(proj(R)))
        if Rn < tol:
            break
        b = proj(-R)
        mv = lambda v: proj(r17.tangent_matvec(proj(v), tris, grads, areas, alpha_e, p, G))
        du = r17.cg(mv, b, tol=1e-10, maxit=6000)
        t = 1.0
        for _ in range(25):
            trial = u + t * du
            Rt, _, _ = r17.residual(trial, tris, grads, areas, cent, alpha_e, p, fq)
            if float(np.linalg.norm(proj(Rt))) < Rn:
                break
            t *= 0.5
        u = u + t * du
        u = u - u.mean()
    return coords, tris, grads, areas, cent, alpha_e, u, fq


# ------------------------------------------------------------------ RT0 equilibrated flux
def equilibrated_flux(coords, tris, grads, areas, cent, alpha_e, u, fq, p):
    """RT0 field sigma~ with div sigma~ = -pi_0 f and sigma~.n = 0 on the boundary."""
    # edges
    edges = {}
    for k, (a, b, c) in enumerate(tris):
        for (i, j) in ((a, b), (b, c), (c, a)):
            key = (min(i, j), max(i, j))
            e = edges.setdefault(key, {"tris": [], "sign": {}})
            e["tris"].append(k)
    keys = sorted(edges)
    edof = {key: t for t, key in enumerate(keys)}          # global dof (orientation: min->max)
    nfree = len(keys)
    M = np.zeros((nfree, nfree)); B = np.zeros((len(tris), nfree))
    basis = {}                                            # (tri, key) -> (c_e, point P_opposite)
    for key, dat in edges.items():
        i, j = key
        P, Q = coords[i], coords[j]
        L = float(np.linalg.norm(Q - P))
        tvec = (Q - P) / L
        nvec = np.array([-tvec[1], tvec[0]])               # global normal
        for k in dat["tris"]:
            a, b, c = tris[k]
            opp = [v for v in (a, b, c) if v not in key]
            assert len(opp) == 1
            R = coords[opp[0]]
            # integrate (x - R).n over the edge
            s = 0.5 * L * float(np.dot(0.5 * (P + Q) - R, nvec))
            c_e = 1.0 / s
            basis[(k, key)] = (c_e, R, nvec)
            B[k, edof[key]] = 2.0 * c_e * areas[k]              # int_T div phi_e dx = 2 c_e |T|
    # mass matrix by 3-point quadrature
    qp = [(1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3)]
    for k, (a, b, c) in enumerate(tris):
        A0, B0, C0 = coords[a], coords[b], coords[c]
        loc = [edof[(min(x, y), max(x, y))] for (x, y) in ((a, b), (b, c), (c, a))]
        for (b1, b2) in qp:
            b3 = 1 - b1 - b2
            X = b1 * A0 + b2 * B0 + b3 * C0
            vals = []
            for (x, y) in ((a, b), (b, c), (c, a)):
                key = (min(x, y), max(x, y))
                c_e, R, nv = basis[(k, key)]
                vals.append(c_e * (X - R))
            for ii in range(3):
                for jj in range(3):
                    M[loc[ii], loc[jj]] += (areas[k] / 3.0) * float(np.dot(vals[ii], vals[jj]))
    # boundary edges: sigma~.n = 0 -> those dofs are fixed to zero
    free = [edof[key] for key, dat in edges.items() if len(dat["tris"]) == 2]
    free_set = set(free)
    allidx = list(range(nfree))
    fixed = [t for t in allidx if t not in free_set]
    Mf = M[np.ix_(free, free)]
    Bf = B[:, free]
    rhs = -(fq * areas)                                    # B sigma = -pi_0 f |T|
    S = Bf @ np.linalg.solve(Mf, Bf.T)                     # SPD (up to constants)
    # pin the constant: add a rank-one term (mean zero constraint on p)
    S = S + np.ones((len(tris), len(tris))) / len(tris)
    piv = np.linalg.solve(S, -rhs)                         # M sigma + B^T p = 0, B sigma = -pi_0 f |T|
    sig = -np.linalg.solve(Mf, Bf.T @ piv)
    return keys, free, edof, sig, B, M, basis


def psi_sigma(cent, alpha_e, areas, keys, free, edof, sig, p, basis, tris, coords):
    """Psi(sigma~) = int alpha^{1-p'} |sigma~|^{p'} / p'  (3-point quadrature)."""
    pp = p / (p - 1.0)
    qp = [(1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3)]
    val = 0.0
    sigmap = {d: s for d, s in zip(free, sig)}
    for k, (a, b, c) in enumerate(tris):
        A0, B0, C0 = coords[a], coords[b], coords[c]
        for (b1, b2) in qp:
            b3 = 1 - b1 - b2
            X = b1 * A0 + b2 * B0 + b3 * C0
            v = np.zeros(2)
            for (x, y) in ((a, b), (b, c), (c, a)):
                key = (min(x, y), max(x, y))
                if edof[key] in sigmap:
                    c_e, R, nv = basis[(k, key)]
                    v = v + sigmap[edof[key]] * c_e * (X - R)
            val += (areas[k] / 3.0) * (alpha_e[k] ** (1 - pp)) * (float(v @ v) ** (pp / 2.0)) / pp
    return val


def quadrature_scalars(coords, tris, areas, cent, alpha_e, p, alpha_fn, u=None):
    """Phi(u*), (f,u*), rho, (f-pi_0 f, u*), Phi(u_h), (f,u_h) with ONE AND THE SAME 7-point rule
    (u_h enters through its P1 interpolant, so all quantities are quadrature-consistent)."""
    qp = np.array([[1/3, 1/3], [0.0597158717, 0.4701420641], [0.4701420641, 0.0597158717],
                   [0.4701420641, 0.4701420641], [0.7974269853, 0.1012865073],
                   [0.1012865073, 0.7974269853], [0.1012865073, 0.1012865073]])
    qw = np.array([0.225, 0.1323941527, 0.1323941527, 0.1323941527,
                   0.1259391805, 0.1259391805, 0.1259391805])
    Phi_s = fu_s = rho = defc = fu_h = 0.0
    Phi_h_q = 0.0
    grads, _, _ = r17.tri_geometry(coords, tris)
    G = r17.element_grad(u, tris, grads) if u is not None else None
    for k in range(len(tris)):
        A = coords[tris[k, 0]]; B = coords[tris[k, 1]]; C = coords[tris[k, 2]]
        a = alpha_e[k]
        for (b1, b2), w in zip(qp, qw):
            b3 = 1 - b1 - b2
            x, y = b1 * A[0] + b2 * B[0] + b3 * C[0], b1 * A[1] + b2 * B[1] + b3 * C[1]
            gx, gy = gN(x, y)
            s2 = 1.0 + gx * gx + gy * gy
            Phi_s += w * areas[k] * a * s2 ** (p / 2.0) / p
            fv = fN(x, y, alpha_fn, p)
            fu_s += w * areas[k] * fv * uN(x, y)
            defc += w * areas[k] * (fv - fq[k]) * uN(x, y)
            if u is not None:
                uh = b1 * u[tris[k, 0]] + b2 * u[tris[k, 1]] + b3 * u[tris[k, 2]]
                fu_h += w * areas[k] * fv * uh
                s2h = 1.0 + float(G[k] @ G[k])
                Phi_h_q += w * areas[k] * a * s2h ** (p / 2.0) / p
                ex, ey = gx - G[k, 0], gy - G[k, 1]
                rho += w * areas[k] * a * (math.hypot(ex, ey) ** p)
    return Phi_s, fu_s, rho, defc, fu_h, Phi_h_q


if __name__ == "__main__":
    res = []
    for kappa in (1.0, 1e2, 1e4):
        alpha_fn = (lambda k: (lambda x, y: 1.0 if x < 0.5 else k))(kappa)
        for p in (2.0, 4.0):
            N = 16
            coords, tris, grads, areas, cent, alpha_e, u, fq = solve_neumann(N, kappa, p, alpha_fn)
            Rchk, _, _ = r17.residual(u, tris, grads, areas, cent, alpha_e, p, fq)
            keys, free, edof, sig, B, M, basis = equilibrated_flux(coords, tris, grads, areas, cent, alpha_e, u, fq, p)
            # verification: div sigma~ == -pi_0 f ?
            sigmap = {d: s for d, s in zip(free, sig)}
            divmax = 0.0
            for k in range(len(tris)):
                d = 0.0
                for (x, y) in ((tris[k, 0], tris[k, 1]), (tris[k, 1], tris[k, 2]), (tris[k, 2], tris[k, 0])):
                    key = (min(x, y), max(x, y))
                    if edof[key] in sigmap:
                        c_e, R, nv = basis[(k, key)]
                        d += sigmap[edof[key]] * 2.0 * c_e
                divmax = max(divmax, abs(d + fq[k]))
            G = r17.element_grad(u, tris, grads)
            s2 = 1.0 + np.sum(G * G, axis=1)
            Phi_h = float(np.sum(areas * alpha_e * s2 ** (p / 2.0) / p))
            Psi = psi_sigma(cent, alpha_e, areas, keys, free, edof, sig, p, basis, tris, coords)
            Phi_s, fu_s, rho, defc, fu_h, Phi_h_q = quadrature_scalars(coords, tris, areas, cent, alpha_e, p, alpha_fn, u)
            E_h, E_s = Phi_h_q - fu_h, Phi_s - fu_s
            eta_comput = Phi_h_q - fu_h + Psi
            eta_with_def = eta_comput + defc
            # error measure err = (int alpha (|grad u*|+|grad u_h|)^{p-2}|grad e|^2)^{1/2}
            errE = 0.0
            for k in range(len(tris)):
                x, y = cent[k]
                gx, gy = gN(x, y)
                ex, ey = gx - G[k, 0], gy - G[k, 1]
                wgt = math.hypot(gx, gy) + math.hypot(float(G[k, 0]), float(G[k, 1]))
                errE += alpha_e[k] * areas[k] * (wgt ** (p - 2.0)) * (ex * ex + ey * ey)
            errE = math.sqrt(errE)
            rec = {"kappa": kappa, "p": p, "N": N, "newton_residual": float(np.linalg.norm(Rchk - Rchk.mean())), "div_residual_max": divmax,
                   "gap": E_h - E_s, "eta_guar_computed": eta_comput,
                   "deficit_term": defc, "eta_guar_with_deficit": eta_with_def,
                   "ratio_guar_over_gap": eta_with_def / (E_h - E_s) if E_h != E_s else float("nan"),
                   "rho": rho, "err_energy": errE,
                   "guaranteed_holds": bool(eta_with_def >= E_h - E_s)}
            res.append(rec)
            print("kap=%8.0e p=%.1f  |div|max=%.2e  newton=%.1e  gap=%.4e  eta_guar=%.4e (+def %.2e) "
                  "ratio=%.3f  holds=%s" % (kappa, p, divmax, rec["newton_residual"], E_h - E_s, eta_comput, defc,
                                            rec["ratio_guar_over_gap"], rec["guaranteed_holds"]))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("saved:", OUT)
