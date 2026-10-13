# -*- coding: utf-8 -*-
"""Joint extreme regime kappa = 1e6..1e8, p = 1.5 / 1.2 (2D scalar problem).

Why the earlier attempt failed: the Newton loop used unpreconditioned-ish CG whose iteration
count grows like sqrt(kappa)/h; with maxit = 4000-6000 the linear solves were inaccurate, so the
nonlinear residual stalled at O(1e3).  Here the tangent system is assembled densely and solved
DIRECTLY (numpy.linalg.solve), which is affordable up to N=32 (961 free dofs), and the solution is
continued first in kappa at p = 2 (linear problem) and then in p down to 1.2.
"""
import importlib.util, json, math, sys, time
import numpy as np

sys.path.insert(0, r"C:\Users\artem\SeekDeep\Workspaces\1\work")
sp = importlib.util.spec_from_file_location("r17", r"C:\Users\artem\SeekDeep\Workspaces\1\work\r17_numerics.py")
r17 = importlib.util.module_from_spec(sp); sp.loader.exec_module(r17)
sp2 = importlib.util.spec_from_file_location("p2", r"C:\Users\artem\SeekDeep\Workspaces\1\work\r17_phase2.py")
p2 = importlib.util.module_from_spec(sp2); sp2.loader.exec_module(p2)

OUT = r"C:\Users\artem\SeekDeep\Workspaces\1\work\r17_extreme2.json"


def assemble(N, p, alpha_fn):
    coords, tris = r17.build_mesh(N)
    grads, areas, cent = r17.tri_geometry(coords, tris)
    n = len(coords)
    alpha_e = np.array([alpha_fn(cx, cy) for cx, cy in cent])
    fq = np.array([r17.f_rhs(cx, cy, alpha_fn, p) for cx, cy in cent])
    onb = ((coords[:, 0] <= 0) | (coords[:, 0] >= 1) | (coords[:, 1] <= 0) | (coords[:, 1] >= 1))
    free = np.where(~onb)[0]
    pos = -np.ones(n, dtype=int); pos[free] = np.arange(len(free))
    return dict(coords=coords, tris=tris, grads=grads, areas=areas, cent=cent, alpha_e=alpha_e,
                fq=fq, free=free, pos=pos, n=n)


def tangent_dense(S, p, G):
    """dense Hessian of the discrete energy restricted to the free dofs."""
    tris, grads, areas, alpha_e, pos, free = S["tris"], S["grads"], S["areas"], S["alpha_e"], S["pos"], S["free"]
    nf = len(free)
    M = np.zeros((nf, nf))
    s = 1.0 + np.sum(G * G, axis=1)
    w0 = alpha_e * s ** ((p - 2.0) / 2.0)
    w1 = alpha_e * (p - 2.0) * s ** ((p - 4.0) / 2.0) if abs(p - 2.0) > 1e-12 else np.zeros_like(w0)
    for k in range(len(tris)):
        ids = pos[tris[k]]
        if np.any(ids < 0):
            keep = ids >= 0
        else:
            keep = np.ones(3, dtype=bool)
        gl = grads[k]                                   # (3,2)
        A = areas[k]
        base = np.einsum('id,jd->ij', gl, gl)           # (3,3)
        outer = np.einsum('id,jd->ij', gl, gl * 0.0)    # placeholder
        gg = np.outer(gl @ G[k], gl @ G[k])
        loc = A * (w0[k] * base + w1[k] * gg)
        ii, jj = np.meshgrid(np.arange(3), np.arange(3), indexing='ij')
        m = keep[ii] & keep[jj]
        np.add.at(M, (ids[ii[m]], ids[jj[m]]), loc[m])
    return M


def newton_direct(S, p, u0=None, tol=1e-11, maxit=40, verbose=False):
    n, free, fq, areas, tris, grads, cent, alpha_e = (S["n"], S["free"], S["fq"], S["areas"],
                                                      S["tris"], S["grads"], S["cent"], S["alpha_e"])
    u = np.zeros(n) if u0 is None else u0.copy()
    hist = []
    t0 = time.perf_counter()
    for it in range(maxit):
        R, G, flux = r17.residual(u, tris, grads, areas, cent, alpha_e, p, fq)
        Rn = float(np.linalg.norm(R[free]))
        hist.append(Rn)
        if Rn < tol:
            break
        M = tangent_dense(S, p, G)
        try:
            du = np.linalg.solve(M, -R[free])
        except np.linalg.LinAlgError:
            du = np.linalg.lstsq(M, -R[free], rcond=None)[0]
        t = 1.0
        for _ in range(30):
            trial = u.copy(); trial[free] += t * du
            Rt, _, _ = r17.residual(trial, tris, grads, areas, cent, alpha_e, p, fq)
            if float(np.linalg.norm(Rt[free])) < Rn * (1 - 1e-4 * t):
                break
            t *= 0.5
        u[free] += t * du
    R, G, flux = r17.residual(u, tris, grads, areas, cent, alpha_e, p, fq)
    return u, float(np.linalg.norm(R[free])), len(hist), time.perf_counter() - t0, hist


def run(N, kappas, ps):
    """continuation: kappa at p=2 (linear, exact), then p downward."""
    res = []
    u = None
    for kappa in kappas:
        af = (lambda k: (lambda x, y: 1.0 if x < 0.5 else k))(kappa)
        S = assemble(N, 2.0, af)
        u, rn, its, dt, hist = newton_direct(S, 2.0, u0=u)
        for p in ps:
            Sp = assemble(N, p, af)
            u, rn, its, dt, hist = newton_direct(Sp, p, u0=u)
            bd = p2.breakdown(Sp["coords"], Sp["tris"], Sp["grads"], Sp["areas"], Sp["cent"],
                              Sp["alpha_e"], u, Sp["fq"], p)
            errE, errp = r17.errors(Sp["coords"], Sp["tris"], Sp["grads"], Sp["areas"], Sp["cent"],
                                    Sp["alpha_e"], u, p)
            rec = {"N": N, "kappa": kappa, "p": p, "residual": rn, "newton_steps": its, "time_s": dt,
                   "err_energy": errE, "err_grad": errp,
                   "eff_split": bd["split_total"] / errE, "eff_arith": bd["arith_total"] / errE,
                   "eff_harm": bd["harm_total"] / errE,
                   "iface_share_split": bd["split_iface"] / bd["split_total"],
                   "iface_share_arith": bd["arith_iface"] / bd["arith_total"]}
            res.append(rec)
            print("N=%2d kap=%.0e p=%.1f  res=%.2e steps=%2d %5.2fs | err=%.4e | eff split=%9.2f arith=%8.2f harm=%9.2f | iface %.2f / %.2f"
                  % (N, kappa, p, rn, its, dt, errE, rec["eff_split"], rec["eff_arith"], rec["eff_harm"],
                     rec["iface_share_split"], rec["iface_share_arith"]), flush=True)
    return res


if __name__ == "__main__":
    out = []
    for N in (16, 32):
        out += run(N, [1e4, 1e6, 1e8], [2.0, 1.8, 1.6, 1.5, 1.4, 1.2])
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved:", OUT)
