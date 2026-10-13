# -*- coding: utf-8 -*-
"""Phase 2:
 (2) decomposition of the estimator (volume / interface faces / interior faces) for three
     face-weight variants -> exact dependence of the constants on the local contrast;
 (3) the case 1 < p < 2 (p-continuation of Newton);
 (1) numerical validation of the duality-gap (guaranteed) bound with the exact equilibrated flux.
"""
import importlib.util, json, math, sys
import numpy as np

sys.path.insert(0, r"C:\Users\artem\SeekDeep\Workspaces\1\work")
spec = importlib.util.spec_from_file_location("r17", r"C:\Users\artem\SeekDeep\Workspaces\1\work\r17_numerics.py")
r17 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r17)

OUT = r"C:\Users\artem\SeekDeep\Workspaces\1\work\r17_phase2.json"


def solve_p(N, kappa, p_target, alpha_fn, steps=6):
    """p-continuation: start from p=2 (linear) and continue to p_target."""
    ps = np.linspace(2.0, p_target, steps) if p_target != 2.0 else np.array([2.0])
    u_init = None
    res = None
    for p in ps:
        coords, tris = r17.build_mesh(N)
        grads, areas, cent = r17.tri_geometry(coords, tris)
        n = len(coords)
        alpha_e = np.array([alpha_fn(cx, cy) for cx, cy in cent])
        fq = np.array([r17.f_rhs(cx, cy, alpha_fn, float(p)) for cx, cy in cent])
        inner = np.zeros(n, dtype=bool)
        inner[(coords[:, 0] > 0) & (coords[:, 0] < 1) & (coords[:, 1] > 0) & (coords[:, 1] < 1)] = True
        free = np.where(inner)[0]
        u = np.zeros(n) if u_init is None else u_init.copy()

        def pad(v):
            w = np.zeros(n); w[free] = v; return w
        for it in range(80):
            R, G, flux = r17.residual(u, tris, grads, areas, cent, alpha_e, float(p), fq)
            Rn = float(np.linalg.norm(R[free]))
            if Rn < 1e-10:
                break
            b = -R[free]
            mv = lambda v: r17.tangent_matvec(pad(v), tris, grads, areas, alpha_e, float(p), G)[free]
            du = r17.cg(mv, b, tol=1e-10, maxit=6000)
            t = 1.0
            for _ in range(30):
                trial = u.copy(); trial[free] += t * du
                Rt, _, _ = r17.residual(trial, tris, grads, areas, cent, alpha_e, float(p), fq)
                if float(np.linalg.norm(Rt[free])) < Rn * (1 - 1e-4 * t):
                    break
                t *= 0.5
            u[free] += t * du
        u_init = u
        res = (coords, tris, grads, areas, cent, alpha_e, u, fq, free)
    return res


def breakdown(coords, tris, grads, areas, cent, alpha_e, u, fq, p):
    """split eta into volume / interface-face / interior-face parts for 3 weight variants."""
    pp = p / (p - 1.0)
    n = len(coords)
    hT = math.sqrt(2.0) / (round(math.sqrt(n)) - 1)
    RT = fq
    vol = float(np.sum((hT * np.abs(RT)) ** pp * alpha_e ** (1 - pp) * areas))
    G = r17.element_grad(u, tris, grads)
    s = 1.0 + np.sum(G * G, axis=1)
    flux = (alpha_e * s ** ((p - 2.0) / 2.0))[:, None] * G
    edges = r17.edge_table(tris)
    acc = {"split": [0.0, 0.0], "arith": [0.0, 0.0], "harm": [0.0, 0.0]}   # [interface, interior]
    for (i, j), ks in edges.items():
        if len(ks) != 2:
            continue
        k1, k2 = ks
        x1, y1 = coords[i]; x2, y2 = coords[j]
        hF = math.hypot(x2 - x1, y2 - y1)
        nx, ny = -(y2 - y1) / hF, (x2 - x1) / hF
        jf = (flux[k1, 0] - flux[k2, 0]) * nx + (flux[k1, 1] - flux[k2, 1]) * ny
        a1, a2 = alpha_e[k1], alpha_e[k2]
        base = (hF * abs(jf)) ** pp
        is_iface = abs(a1 - a2) > 1e-12 * max(a1, a2)
        idx = 0 if is_iface else 1
        acc["split"][idx] += base * 0.5 * (a1 ** (1 - pp) + a2 ** (1 - pp))
        acc["arith"][idx] += base * ((a1 + a2) / 2.0) ** (1 - pp)
        acc["harm"][idx] += base * (2.0 * a1 * a2 / (a1 + a2)) ** (1 - pp)
    out = {"volume": vol ** (1 / pp)}
    for k in acc:
        out[k + "_iface"] = acc[k][0] ** (1 / pp)
        out[k + "_interior"] = acc[k][1] ** (1 / pp)
        out[k + "_total"] = out["volume"] + out[k + "_iface"] + out[k + "_interior"]
    return out


def exact_solution_quantities(coords, tris, grads, areas, cent, alpha_e, p, fq, alpha_fn):
    """Phi(u*), (f,u*), Psi(sigma*) with high-accuracy element quadrature."""
    # 7-point Dunavant rule on the reference triangle
    qp = np.array([[1/3, 1/3], [0.0597158717, 0.4701420641], [0.4701420641, 0.0597158717],
                   [0.4701420641, 0.4701420641], [0.7974269853, 0.1012865073],
                   [0.1012865073, 0.7974269853], [0.1012865073, 0.1012865073]])
    qw = np.array([0.225, 0.1323941527, 0.1323941527, 0.1323941527,
                   0.1259391805, 0.1259391805, 0.1259391805])
    Phi = 0.0; fu = 0.0; Psi = 0.0; fu_h = 0.0
    for k in range(len(tris)):
        xa, ya = coords[tris[k, 0]]; xb, yb = coords[tris[k, 1]]; xc, yc = coords[tris[k, 2]]
        a = alpha_e[k]
        for (b1, b2), w in zip(qp, qw):
            b3 = 1.0 - b1 - b2
            x = b1 * xa + b2 * xb + b3 * xc
            y = b1 * ya + b2 * yb + b3 * yc
            gx, gy = r17.grad_u_exact(x, y)
            s2 = 1.0 + gx * gx + gy * gy
            Phi += w * areas[k] * a * s2 ** (p / 2.0) / p
            c = a * s2 ** ((p - 2.0) / 2.0)
            sx, sy = c * gx, c * gy
            Psi += w * areas[k] * a ** (1 - p / (p - 1.0)) * (sx * sx + sy * sy) ** (p / (2 * (p - 1.0))) / (p / (p - 1.0))
            fv = r17.f_rhs(x, y, alpha_fn, p)
            fu += w * areas[k] * fv * r17.u_exact(x, y)
            uh = (b1 * coords[tris[k, 0]][1] * 0 + 0)  # placeholder (filled below)
    # (f, u_h) with the same rule using the P1 interpolant
    for k in range(len(tris)):
        xa, ya = coords[tris[k, 0]]; xb, yb = coords[tris[k, 1]]; xc, yc = coords[tris[k, 2]]
        ua, ub, uc = None, None, None
    return Phi, fu, Psi


def duality_gap(coords, tris, grads, areas, cent, alpha_e, u, fq, p, alpha_fn):
    """E(u_h) - E(u*) vs Phi(u_h) - (f,u_h) + Psi(sigma*)  (exact flux)."""
    # Phi(u_h) exact for P1
    G = r17.element_grad(u, tris, grads)
    s2 = 1.0 + np.sum(G * G, axis=1)
    Phi_h = float(np.sum(areas * alpha_e * s2 ** (p / 2.0) / p))
    fu_h = float(np.sum(areas / 3.0 * fq * (u[tris[:, 0]] + u[tris[:, 1]] + u[tris[:, 2]])))
    # exact-solution quantities by quadrature (7-point rule)
    qp = np.array([[1/3, 1/3], [0.0597158717, 0.4701420641], [0.4701420641, 0.0597158717],
                   [0.4701420641, 0.4701420641], [0.7974269853, 0.1012865073],
                   [0.1012865073, 0.7974269853], [0.1012865073, 0.1012865073]])
    qw = np.array([0.225, 0.1323941527, 0.1323941527, 0.1323941527,
                   0.1259391805, 0.1259391805, 0.1259391805])
    pp = p / (p - 1.0)
    Phi_s = 0.0; fu_s = 0.0; Psi_s = 0.0; rho = 0.0
    for k in range(len(tris)):
        A = coords[tris[k, 0]]; B = coords[tris[k, 1]]; C = coords[tris[k, 2]]
        a = alpha_e[k]
        for (b1, b2), w in zip(qp, qw):
            b3 = 1.0 - b1 - b2
            x, y = b1 * A[0] + b2 * B[0] + b3 * C[0], b1 * A[1] + b2 * B[1] + b3 * C[1]
            gx, gy = r17.grad_u_exact(x, y)
            s2 = 1.0 + gx * gx + gy * gy
            Phi_s += w * areas[k] * a * s2 ** (p / 2.0) / p
            c = a * s2 ** ((p - 2.0) / 2.0)
            sx, sy = c * gx, c * gy
            Psi_s += w * areas[k] * (a ** (1 - pp)) * (sx * sx + sy * sy) ** (pp / 2.0) / pp
            fu_s += w * areas[k] * r17.f_rhs(x, y, alpha_fn, p) * r17.u_exact(x, y)
            # error modular rho = int alpha |grad e|^p  (element gradient constant)
        gxk, gyk = G[k]
        for (b1, b2), w in zip(qp, qw):
            b3 = 1.0 - b1 - b2
            x, y = b1 * A[0] + b2 * B[0] + b3 * C[0], b1 * A[1] + b2 * B[1] + b3 * C[1]
            gx, gy = r17.grad_u_exact(x, y)
            ex, ey = gx - gxk, gy - gyk
            rho += w * areas[k] * a * (math.hypot(ex, ey) ** p)
    E_h = Phi_h - fu_h
    E_s = Phi_s - fu_s
    eta_g = Phi_h - fu_h + Psi_s
    return {"Phi_h": Phi_h, "fu_h": fu_h, "E_h": E_h, "E_s": E_s, "Psi_sigma_exact": Psi_s,
            "gap": E_h - E_s, "eta_guar": eta_g, "ratio_eta_gap": eta_g / (E_h - E_s) if E_h != E_s else float("nan"),
            "rho": rho, "eta_over_rho": eta_g / rho if rho > 0 else float("nan")}


if __name__ == "__main__":
    out = []
    # ---- (3) p < 2 study and (2) breakdown
    for kappa in (1.0, 1e2, 1e4):
        alpha_fn = (lambda k: (lambda x, y: 1.0 if x < 0.5 else k))(kappa)
        for p in (2.0, 1.5):
            for N in (16, 32):
                coords, tris, grads, areas, cent, alpha_e, u, fq, free = solve_p(N, kappa, p, alpha_fn)
                bd = breakdown(coords, tris, grads, areas, cent, alpha_e, u, fq, p)
                errE, errp = r17.errors(coords, tris, grads, areas, cent, alpha_e, u, p)
                rec = {"N": N, "kappa": kappa, "p": p, "err_energy": errE, "err_grad": errp}
                rec.update(bd)
                rec["eff_split"] = bd["split_total"] / errE
                rec["eff_arith"] = bd["arith_total"] / errE
                rec["eff_harm"] = bd["harm_total"] / errE
                out.append(rec)
                print("kap=%8.0e p=%.1f N=%2d err=%.3e | vol=%.3e iface: split=%.3e arith=%.3e | "
                      "eff: split=%.2f arith=%.2f harm=%.2f | iface share: split=%.2f arith=%.2f"
                      % (kappa, p, N, errE, bd["volume"], bd["split_iface"], bd["arith_iface"],
                         rec["eff_split"], rec["eff_arith"], rec["eff_harm"],
                         bd["split_iface"] / bd["split_total"], bd["arith_iface"] / bd["arith_total"]))
    # ---- (1) duality gap with the exact flux
    print("\n--- duality gap (exact flux sigma*) ---")
    for kappa in (1.0, 1e4, 1e6):
        alpha_fn = (lambda k: (lambda x, y: 1.0 if x < 0.5 else k))(kappa)
        for p in (2.0, 4.0):
            N = 32
            coords, tris, grads, areas, cent, alpha_e, u, fq, free = solve_p(N, kappa, p, alpha_fn)
            dg = duality_gap(coords, tris, grads, areas, cent, alpha_e, u, fq, p, alpha_fn)
            dg.update({"N": N, "kappa": kappa, "p": p})
            out.append(dg)
            print("kap=%8.0e p=%.1f  gap=%.4e eta_guar=%.4e ratio=%.3f  rho=%.3e eta/rho=%.3e"
                  % (kappa, p, dg["gap"], dg["eta_guar"], dg["ratio_eta_gap"], dg["rho"], dg["eta_over_rho"]))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved:", OUT)
