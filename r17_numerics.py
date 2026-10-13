# -*- coding: utf-8 -*-
"""Independent 2D P1-FEM experiment for the weighted a posteriori estimator.

  -div( alpha(x) (1+|grad u|^2)^((p-2)/2) grad u ) = f   in (0,1)^2,  u=0 on bd
  alpha = 1 for x<1/2, alpha = kappa for x>1/2
  exact solution u* = sin(pi x) sin(pi y); f computed from the exact flux by
  central differences of the analytic flux (machine-accurate to ~1e-10).

Estimators (faces are the edges of the triangulation):
  eta_res  = ( sum_T (h_T |R_T|)^p' alpha_T^{1-p'} |T| )^{1/p'}
  eta_jmp  = ( sum_F (h_F |j_F|)^p' * W_F )^{1/p'},   W_F = split weight (default),
                                                       or harmonic / arithmetic mean
  eta_unw  = same without alpha weights.
Error:  err = ( int alpha (|grad u*|+|grad u_h|)^{p-2} |grad e|^2 )^{1/2}.
"""
import json, math, sys
import numpy as np

OUT = r"C:\Users\artem\SeekDeep\Workspaces\1\work\r17_numerics.json"


# ---------------------------------------------------------------- exact solution / rhs
def u_exact(x, y):
    return np.sin(math.pi * x) * np.sin(math.pi * y)


def grad_u_exact(x, y):
    return (math.pi * math.cos(math.pi * x) * math.sin(math.pi * y),
            math.pi * math.sin(math.pi * x) * math.cos(math.pi * y))


def flux_exact(x, y, alpha, p):
    gx, gy = grad_u_exact(x, y)
    s = 1.0 + gx * gx + gy * gy
    c = alpha * s ** ((p - 2.0) / 2.0)
    return c * gx, c * gy


def f_rhs(x, y, alpha_fn, p, h=1e-5):
    """-div(flux) by central differences of the analytic flux."""
    fxp = flux_exact(x + h, y, alpha_fn(x + h, y), p)
    fxm = flux_exact(x - h, y, alpha_fn(x - h, y), p)
    fyp = flux_exact(x, y + h, alpha_fn(x, y + h), p)
    fym = flux_exact(x, y - h, alpha_fn(x, y - h), p)
    dfx = (fxp[0] - fxm[0]) / (2 * h)
    dfy = (fyp[1] - fym[1]) / (2 * h)
    return -(dfx + dfy)


# ---------------------------------------------------------------- mesh
def build_mesh(N):
    xs = np.linspace(0.0, 1.0, N + 1)
    nodes = [(i, j) for j in range(N + 1) for i in range(N + 1)]
    index = {n: k for k, n in enumerate(nodes)}
    coords = np.array([[i / N, j / N] for (i, j) in nodes])
    tris = []
    for j in range(N):
        for i in range(N):
            n00 = index[(i, j)]; n10 = index[(i + 1, j)]
            n01 = index[(i, j + 1)]; n11 = index[(i + 1, j + 1)]
            tris.append((n00, n10, n11))
            tris.append((n00, n11, n01))
    return coords, np.array(tris, dtype=int)


def tri_geometry(coords, tris):
    """gradients of the three P1 basis functions, areas, centroids."""
    grads = np.zeros((len(tris), 3, 2))
    areas = np.zeros(len(tris))
    cent = np.zeros((len(tris), 2))
    for k, (a, b, c) in enumerate(tris):
        xa, ya = coords[a]; xb, yb = coords[b]; xc, yc = coords[c]
        J = np.array([[xb - xa, xc - xa], [yb - ya, yc - ya]])
        det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        areas[k] = abs(det) / 2.0
        Jinv = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / det
        grads[k, 1] = Jinv[0, :]      # grad of the reference coordinate xi  = first row of J^{-1}
        grads[k, 2] = Jinv[1, :]      # grad of the reference coordinate eta = second row of J^{-1}
        grads[k, 0] = -grads[k, 1] - grads[k, 2]
        cent[k] = coords[[a, b, c]].mean(axis=0)
    return grads, areas, cent


# ---------------------------------------------------------------- assembly / solver
def element_grad(u, tris, grads):
    return np.einsum('kij,ki->kj', grads, u[tris])


def residual(u, tris, grads, areas, cent, alpha_e, p, fq):
    G = element_grad(u, tris, grads)                      # (ntri,2)
    s = 1.0 + np.sum(G * G, axis=1)
    c = alpha_e * s ** ((p - 2.0) / 2.0)
    flux = c[:, None] * G                                 # (ntri,2)
    Rloc = (np.einsum('kij,kj->ki', grads, flux) - fq[:, None] / 3.0) * areas[:, None]
    R = np.zeros_like(u)
    for i in range(3):
        np.add.at(R, tris[:, i], Rloc[:, i])
    return R, G, flux


def tangent_matvec(v, tris, grads, areas, alpha_e, p, G):
    """Hessian-vector product of the discrete energy."""
    Vg = element_grad(v, tris, grads)
    s = 1.0 + np.sum(G * G, axis=1)
    w = alpha_e * s ** ((p - 2.0) / 2.0)
    if abs(p - 2.0) > 1e-12:
        w = w + alpha_e * (p - 2.0) * s ** ((p - 4.0) / 2.0) * np.sum(G * Vg, axis=1)
    out = np.einsum('kij,kj->ki', grads, w[:, None] * Vg)
    R = np.zeros_like(v)
    for i in range(3):
        np.add.at(R, tris[:, i], out[:, i] * areas)
    return R


def cg(matvec, b, tol=1e-12, maxit=3000):
    x = np.zeros_like(b)
    r = b.copy()
    z = r.copy()
    p = z.copy()
    rz = float(r @ z)
    bn = math.sqrt(float(b @ b)) + 1e-300
    for _ in range(maxit):
        Ap = matvec(p)
        pAp = float(p @ Ap)
        if pAp <= 0:
            break
        a = rz / pAp
        x += a * p
        r -= a * Ap
        if math.sqrt(float(r @ r)) / bn < tol:
            break
        z = r.copy()
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x


def solve_fem(N, kappa, p, alpha_fn, newton_tol=1e-10):
    coords, tris = build_mesh(N)
    grads, areas, cent = tri_geometry(coords, tris)
    n = len(coords)
    alpha_e = np.array([alpha_fn(cx, cy) for cx, cy in cent])
    # rhs: -div of the exact flux, per element (integrated against P1 basis: fq*|T|/3)
    fq = np.array([f_rhs(cx, cy, alpha_fn, p) for cx, cy in cent])
    u = np.zeros(n)
    inner = np.zeros(n, dtype=bool)
    inner[(coords[:, 0] > 0) & (coords[:, 0] < 1) & (coords[:, 1] > 0) & (coords[:, 1] < 1)] = True
    free = np.where(inner)[0]
    def pad(v):
        w = np.zeros(n)
        w[free] = v
        return w
    for it in range(60):
        R, G, flux = residual(u, tris, grads, areas, cent, alpha_e, p, fq)
        Rn = np.linalg.norm(R[free])
        if Rn < newton_tol:
            break
        b = -R[free]
        mv = lambda v: tangent_matvec(pad(v), tris, grads, areas, alpha_e, p, G)[free]
        du = cg(mv, b, tol=1e-10, maxit=4000)
        # damped update (backtracking on the residual norm)
        t = 1.0
        for _ in range(25):
            trial = u.copy(); trial[free] += t * du
            Rt, _, _ = residual(trial, tris, grads, areas, cent, alpha_e, p, fq)
            if np.linalg.norm(Rt[free]) < Rn * (1 - 1e-4 * t):
                break
            t *= 0.5
        u[free] += t * du
    R, G, flux = residual(u, tris, grads, areas, cent, alpha_e, p, fq)
    return coords, tris, grads, areas, cent, alpha_e, u, G, flux, float(np.linalg.norm(R[free]))


# ---------------------------------------------------------------- estimators
def edge_table(tris):
    """interior edges: (node_a, node_b) sorted, with the two adjacent triangles."""
    edges = {}
    for k, (a, b, c) in enumerate(tris):
        for (i, j) in ((a, b), (b, c), (c, a)):
            key = (min(i, j), max(i, j))
            edges.setdefault(key, []).append(k)
    return edges


def estimators(coords, tris, grads, areas, cent, alpha_e, u, G, flux, fvals, p, kappa):
    pp = p / (p - 1.0)
    hT = np.sqrt(2.0) / (len(coords) ** 0.5 - 1)            # element diameter
    # element residual: R_T = f  (P1 flux is constant per element)
    RT = fvals * 3.0
    # volume terms
    def vol(weighted):
        if weighted:
            return float(np.sum((hT * np.abs(RT)) ** pp * alpha_e ** (1 - pp) * areas))
        return float(np.sum((hT * np.abs(RT)) ** pp * areas))
    # edges
    edges = edge_table(tris)
    eta_split = eta_harm = eta_arith = eta_unw = 0.0
    for (i, j), ks in edges.items():
        if len(ks) != 2:
            continue
        k1, k2 = ks
        x1, y1 = coords[i]; x2, y2 = coords[j]
        hF = math.hypot(x2 - x1, y2 - y1)
        # unit normal consistent for both triangles
        tx, ty = (x2 - x1) / hF, (y2 - y1) / hF
        nx, ny = -ty, tx
        jf = (flux[k1, 0] - flux[k2, 0]) * nx + (flux[k1, 1] - flux[k2, 1]) * ny
        a1, a2 = alpha_e[k1], alpha_e[k2]
        base = (hF * abs(jf)) ** pp
        eta_split += base * 0.5 * (a1 ** (1 - pp) + a2 ** (1 - pp))
        eta_arith += base * ((a1 + a2) / 2.0) ** (1 - pp)
        eta_harm += base * (2.0 * a1 * a2 / (a1 + a2)) ** (1 - pp)
        eta_unw += base
    res = (vol(True)) ** (1 / pp)
    return {
        "eta_res": res,
        "eta_jmp_split": eta_split ** (1 / pp),
        "eta_jmp_arith": eta_arith ** (1 / pp),
        "eta_jmp_harm": eta_harm ** (1 / pp),
        "eta_jmp_unw": eta_unw ** (1 / pp),
        "eta_w_split": res + eta_split ** (1 / pp),
        "eta_w_arith": res + eta_arith ** (1 / pp),
        "eta_w_harm": res + eta_harm ** (1 / pp),
        "eta_unw": res + eta_unw ** (1 / pp),
    }


def errors(coords, tris, grads, areas, cent, alpha_e, u, p):
    """weighted energy error and gradient error against the exact solution."""
    G = element_grad(u, tris, grads)
    errE = 0.0
    errp = 0.0
    for k in range(len(tris)):
        x, y = cent[k]
        gx, gy = grad_u_exact(x, y)
        ex, ey = gx - G[k, 0], gy - G[k, 1]
        w = math.hypot(gx, gy) + math.hypot(G[k, 0], G[k, 1])
        a = alpha_e[k] * areas[k]
        errE += a * (w ** (p - 2.0)) * (ex * ex + ey * ey) if w > 0 else a * (ex * ex + ey * ey)
        errp += a * (math.hypot(ex, ey) ** p)
    return math.sqrt(errE), errp ** (1.0 / p)


# ---------------------------------------------------------------- driver
def run(N, kappa, p, alpha_fn):
    coords, tris, grads, areas, cent, alpha_e, u, G, flux, rn = solve_fem(N, kappa, p, alpha_fn)
    fvals = np.array([f_rhs(cx, cy, alpha_fn, p) for cx, cy in cent]) / 3.0
    est = estimators(coords, tris, grads, areas, cent, alpha_e, u, G, flux, fvals, p, kappa)
    errE, errp = errors(coords, tris, grads, areas, cent, alpha_e, u, p)
    out = {"N": N, "kappa": kappa, "p": p, "residual_norm": rn,
           "err_energy": errE, "err_grad": errp}
    out.update(est)
    for k in ("eta_w_split", "eta_w_arith", "eta_w_harm", "eta_unw"):
        out["eff_" + k] = out[k] / errE if errE > 0 else float("nan")
    return out


if __name__ == "__main__":
    res = []
    # validation: kappa = 1, p = 2, convergence and effectivity
    for N in (8, 16, 32):
        r = run(N, 1.0, 2.0, lambda x, y: 1.0)
        res.append(r)
        print("VALID N=%2d p=2 kap=1  errE=%.4e errgrad=%.4e eta_w=%.4e eff_w=%.3f eff_unw=%.3f"
              % (N, r["err_energy"], r["err_grad"], r["eta_w_split"], r["eff_eta_w_split"], r["eff_eta_unw"]))
    # contrast study
    for kappa in (1.0, 1e2, 1e4, 1e6):
        alpha_fn = (lambda k: (lambda x, y: 1.0 if x < 0.5 else k))(kappa)
        for p in (2.0, 4.0):
            for N in (16, 32):
                r = run(N, kappa, p, alpha_fn)
                res.append(r)
                print("kap=%8.0e p=%3.1f N=%2d errE=%.3e eta_w=%.3e eta_unw=%.3e | "
                      "eff_w=%.3f eff_arith=%.3f eff_harm=%.3f eff_unw=%.3f | ratio_unw/w=%.3f kap^(1/p)=%.3f"
                      % (kappa, p, N, r["err_energy"], r["eta_w_split"], r["eta_unw"],
                         r["eff_eta_w_split"], r["eff_eta_w_arith"], r["eff_eta_w_harm"], r["eff_eta_unw"],
                         r["eta_unw"] / r["eta_w_split"], kappa ** (1.0 / p)))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("saved:", OUT)

