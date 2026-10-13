# -*- coding: utf-8 -*-
"""3D P1-FEM experiment (item 1): reliability/efficiency of the weighted estimator in 3D,
   contrast study, and convergence check.  Freudenthal (Kuhn) tetrahedral mesh of (0,1)^3.
   -div( alpha (1+|grad u|^2)^((p-2)/2) grad u ) = f,  u = 0 on the boundary,
   alpha = 1 (x<1/2), kappa (x>1/2),  manufactured u* = sin(pi x) sin(pi y) sin(pi z)
   (the normal derivative vanishes on the interface x=1/2, so the transmission
    condition holds exactly for every p and kappa).
"""
import json, math, sys
import numpy as np

OUT = r"C:\Users\artem\SeekDeep\Workspaces\1\work\r17_3d.json"


def u_exact(x, y, z):
    return math.sin(math.pi * x) * math.sin(math.pi * y) * math.sin(math.pi * z)


def grad_u_exact(x, y, z):
    return (math.pi * math.cos(math.pi * x) * math.sin(math.pi * y) * math.sin(math.pi * z),
            math.pi * math.sin(math.pi * x) * math.cos(math.pi * y) * math.sin(math.pi * z),
            math.pi * math.sin(math.pi * x) * math.sin(math.pi * y) * math.cos(math.pi * z))


def flux_exact(x, y, z, alpha, p):
    gx, gy, gz = grad_u_exact(x, y, z)
    s = 1.0 + gx * gx + gy * gy + gz * gz
    c = alpha * s ** ((p - 2.0) / 2.0)
    return c * gx, c * gy, c * gz


def f_rhs(x, y, z, alpha_fn, p, h=1e-5):
    fxp = flux_exact(x + h, y, z, alpha_fn(x + h, y, z), p)
    fxm = flux_exact(x - h, y, z, alpha_fn(x - h, y, z), p)
    fyp = flux_exact(x, y + h, z, alpha_fn(x, y + h, z), p)
    fym = flux_exact(x, y - h, z, alpha_fn(x, y - h, z), p)
    fzp = flux_exact(x, y, z + h, alpha_fn(x, y, z + h), p)
    fzm = flux_exact(x, y, z - h, alpha_fn(x, y, z - h), p)
    return -((fxp[0] - fxm[0]) + (fyp[1] - fym[1]) + (fzp[2] - fzm[2])) / (2 * h)


def build_mesh3d(N):
    """Freudenthal triangulation: 6 tetrahedra per cube (conforming)."""
    idx = lambda i, j, k: (k * (N + 1) + j) * (N + 1) + i
    coords = []
    for k in range(N + 1):
        for j in range(N + 1):
            for i in range(N + 1):
                coords.append((i / N, j / N, k / N))
    coords = np.array(coords)
    tets = []
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    for k in range(N):
        for j in range(N):
            for i in range(N):
                v000 = idx(i, j, k)
                corners = {(a, b, c): idx(i + a, j + b, k + c) for a in (0, 1) for b in (0, 1) for c in (0, 1)}
                v111 = corners[(1, 1, 1)]
                for (o1, o2, o3) in perms:
                    e = [0, 0, 0]
                    e[o1] = 1
                    p1 = tuple(e)
                    e2 = list(p1); e2[o2] = 1
                    p2 = tuple(e2)
                    tets.append((v000, corners[p1], corners[p2], v111))
    return coords, np.array(tets, dtype=int)


def tet_geometry(coords, tets):
    grads = np.zeros((len(tets), 4, 3))
    vols = np.zeros(len(tets))
    cent = np.zeros((len(tets), 3))
    for k, (a, b, c, d) in enumerate(tets):
        A, B, C, D = coords[a], coords[b], coords[c], coords[d]
        J = np.column_stack((B - A, C - A, D - A))
        det = float(np.linalg.det(J))
        vols[k] = abs(det) / 6.0
        Jinv = np.linalg.inv(J)
        grads[k, 1] = Jinv[0, :]
        grads[k, 2] = Jinv[1, :]
        grads[k, 3] = Jinv[2, :]
        grads[k, 0] = -(grads[k, 1] + grads[k, 2] + grads[k, 3])
        cent[k] = (A + B + C + D) / 4.0
    # element diameters
    diam = np.zeros(len(tets))
    for k, (a, b, c, d) in enumerate(tets):
        pts = coords[[a, b, c, d]]
        diam[k] = max(np.linalg.norm(pts[i] - pts[j]) for i in range(4) for j in range(i + 1, 4))
    return grads, vols, cent, diam


def element_grad(u, tets, grads):
    return np.einsum('kij,ki->kj', grads, u[tets])


def residual(u, tets, grads, vols, alpha_e, p, fq):
    G = element_grad(u, tets, grads)
    s = 1.0 + np.sum(G * G, axis=1)
    flux = (alpha_e * s ** ((p - 2.0) / 2.0))[:, None] * G
    Rloc = (np.einsum('kij,kj->ki', grads, flux) - fq[:, None] / 4.0) * vols[:, None]
    R = np.zeros_like(u)
    for i in range(4):
        np.add.at(R, tets[:, i], Rloc[:, i])
    return R, G, flux


def tangent_matvec(v, tets, grads, vols, alpha_e, p, G):
    Vg = element_grad(v, tets, grads)
    s = 1.0 + np.sum(G * G, axis=1)
    w = alpha_e * s ** ((p - 2.0) / 2.0)
    if abs(p - 2.0) > 1e-12:
        w = w + alpha_e * (p - 2.0) * s ** ((p - 4.0) / 2.0) * np.sum(G * Vg, axis=1)
    out = np.einsum('kij,kj->ki', grads, w[:, None] * Vg)
    R = np.zeros_like(v)
    for i in range(4):
        np.add.at(R, tets[:, i], out[:, i] * vols)
    return R


def cg(matvec, b, tol=1e-10, maxit=4000):
    x = np.zeros_like(b); r = b.copy(); p = r.copy(); rz = float(r @ r)
    bn = math.sqrt(rz) + 1e-300
    for _ in range(maxit):
        Ap = matvec(p); pAp = float(p @ Ap)
        if pAp <= 0:
            break
        a = rz / pAp
        x += a * p; r -= a * Ap
        if math.sqrt(float(r @ r)) / bn < tol:
            break
        rz_new = float(r @ r)
        p = r + (rz_new / rz) * p
        rz = rz_new
    return x


def solve_fem(N, kappa, p, alpha_fn, p_steps=5):
    coords, tets = build_mesh3d(N)
    grads, vols, cent, diam = tet_geometry(coords, tets)
    n = len(coords)
    alpha_e = np.array([alpha_fn(cx, cy, cz) for cx, cy, cz in cent])
    fq = np.array([f_rhs(cx, cy, cz, alpha_fn, p) for cx, cy, cz in cent])
    onb = ((coords[:, 0] <= 0) | (coords[:, 0] >= 1) | (coords[:, 1] <= 0) |
           (coords[:, 1] >= 1) | (coords[:, 2] <= 0) | (coords[:, 2] >= 1))
    free = np.where(~onb)[0]
    u = np.zeros(n)
    ps = np.linspace(2.0, p, p_steps) if abs(p - 2.0) > 1e-12 else np.array([2.0])

    def pad(v):
        w = np.zeros(n); w[free] = v; return w
    for pcur in ps:
        for it in range(50):
            R, G, flux = residual(u, tets, grads, vols, alpha_e, float(pcur), fq)
            Rn = float(np.linalg.norm(R[free]))
            if Rn < 1e-9:
                break
            b = -R[free]
            mv = lambda v: tangent_matvec(pad(v), tets, grads, vols, alpha_e, float(pcur), G)[free]
            du = cg(mv, b, tol=1e-9, maxit=6000)
            t = 1.0
            for _ in range(25):
                trial = u.copy(); trial[free] += t * du
                Rt, _, _ = residual(trial, tets, grads, vols, alpha_e, float(pcur), fq)
                if float(np.linalg.norm(Rt[free])) < Rn:
                    break
                t *= 0.5
            u[free] += t * du
    R, G, flux = residual(u, tets, grads, vols, alpha_e, p, fq)
    return coords, tets, grads, vols, cent, diam, alpha_e, u, fq, float(np.linalg.norm(R[free]))


def faces3d(tets):
    faces = {}
    for k, (a, b, c, d) in enumerate(tets):
        for f in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            key = tuple(sorted(f))
            faces.setdefault(key, []).append(k)
    return faces


def estimators3d(coords, tets, grads, vols, cent, diam, alpha_e, u, fq, p, kappa):
    pp = p / (p - 1.0)
    RT = fq
    G = element_grad(u, tets, grads)
    s = 1.0 + np.sum(G * G, axis=1)
    flux = (alpha_e * s ** ((p - 2.0) / 2.0))[:, None] * G
    vol = float(np.sum((diam * np.abs(RT)) ** pp * alpha_e ** (1 - pp) * vols))
    faces = faces3d(tets)
    et = {"split": 0.0, "arith": 0.0, "unw": 0.0, "split_if": 0.0, "arith_if": 0.0}
    for key, ks in faces.items():
        if len(ks) != 2:
            continue
        k1, k2 = ks
        P, Q, R = coords[key[0]], coords[key[1]], coords[key[2]]
        nrm = np.cross(Q - P, R - P)
        area = 0.5 * float(np.linalg.norm(nrm))
        if area < 1e-14:
            continue
        nvec = nrm / (2 * area)
        hF = max(np.linalg.norm(P - Q), np.linalg.norm(Q - R), np.linalg.norm(R - P))
        # consistent orientation for both elements: use the normal as is, take |jump|
        jf = float(np.dot(flux[k1] - flux[k2], nvec))
        a1, a2 = alpha_e[k1], alpha_e[k2]
        base = (hF * abs(jf)) ** pp
        w_split = 0.5 * (a1 ** (1 - pp) + a2 ** (1 - pp))
        w_arith = ((a1 + a2) / 2.0) ** (1 - pp)
        et["split"] += base * w_split
        et["arith"] += base * w_arith
        et["unw"] += base
        if abs(a1 - a2) > 1e-12 * max(a1, a2):
            et["split_if"] += base * w_split
            et["arith_if"] += base * w_arith
    res = {"eta_vol": vol ** (1 / pp),
           "eta_split": (vol + et["split"]) ** (1 / pp),
           "eta_arith": (vol + et["arith"]) ** (1 / pp),
           "eta_unw": (vol + et["unw"]) ** (1 / pp),
           "eta_iface_split": et["split_if"] ** (1 / pp),
           "eta_iface_arith": et["arith_if"] ** (1 / pp)}
    return res


def errors3d(coords, tets, grads, vols, cent, alpha_e, u, p):
    G = element_grad(u, tets, grads)
    errE = 0.0; errp = 0.0
    for k in range(len(tets)):
        x, y, z = cent[k]
        gx, gy, gz = grad_u_exact(x, y, z)
        ex, ey, ez = gx - G[k, 0], gy - G[k, 1], gz - G[k, 2]
        w = math.sqrt(gx * gx + gy * gy + gz * gz) + math.sqrt(float(G[k] @ G[k]))
        a = alpha_e[k] * vols[k]
        errE += a * (w ** (p - 2.0)) * (ex * ex + ey * ey + ez * ez) if w > 0 else a * (ex * ex + ey * ey + ez * ez)
        errp += a * (math.sqrt(ex * ex + ey * ey + ez * ez) ** p)
    return math.sqrt(errE), errp ** (1.0 / p)


if __name__ == "__main__":
    res = []
    for kappa, p, N in [(1.0, 2.0, 6), (1.0, 2.0, 8), (1.0, 2.0, 10),
                        (1e4, 2.0, 8), (1e6, 2.0, 8),
                        (1.0, 4.0, 8), (1e4, 4.0, 8), (1e6, 4.0, 8)]:
        alpha_fn = (lambda k: (lambda x, y, z: 1.0 if x < 0.5 else k))(kappa)
        out = solve_fem(N, kappa, p, alpha_fn)
        coords, tets, grads, vols, cent, diam, alpha_e, u, fq, rn = out
        est = estimators3d(coords, tets, grads, vols, cent, diam, alpha_e, u, fq, p, kappa)
        errE, errp = errors3d(coords, tets, grads, vols, cent, alpha_e, u, p)
        rec = {"N": N, "kappa": kappa, "p": p, "err_energy": errE, "err_grad": errp, "residual": rn,
               "nodes": len(coords), "tets": len(tets)}
        rec.update(est)
        rec["eff_split"] = est["eta_split"] / errE
        rec["eff_arith"] = est["eta_arith"] / errE
        rec["eff_unw"] = est["eta_unw"] / errE
        res.append(rec)
        print("3D N=%2d kap=%8.0e p=%.1f err=%.3e | eta: split=%.3e arith=%.3e unw=%.3e | "
              "eff: split=%.2f arith=%.2f unw=%.2f | iface share(arith)=%.2f  res=%.1e"
              % (N, kappa, p, errE, est["eta_split"], est["eta_arith"], est["eta_unw"],
                 rec["eff_split"], rec["eff_arith"], rec["eff_unw"],
                 est["eta_iface_arith"] / est["eta_arith"], rn))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("saved:", OUT)
