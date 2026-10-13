# -*- coding: utf-8 -*-
"""Joint extreme regime: compare the estimator with BOTH error measures.
   err_E = (int alpha W^{p-2}|grad e|^2)^{1/2}   (energy type)
   rho   = int alpha |grad e|^p                 (modular, appears in the reliability inequality)
   eff_E = eta/err_E ;  eff_rho = eta / rho^{1/p'}  (the quantity the reliability bound controls)."""
import importlib.util, json, math, sys
import numpy as np
sys.path.insert(0, r"C:\Users\artem\SeekDeep\Workspaces\1\work")
sp = importlib.util.spec_from_file_location("ex", r"C:\Users\artem\SeekDeep\Workspaces\1\work\r17_extreme2.py")
ex = importlib.util.module_from_spec(sp); sp.loader.exec_module(ex)
sp2 = importlib.util.spec_from_file_location("r17", r"C:\Users\artem\SeekDeep\Workspaces\1\work\r17_numerics.py")
r17 = importlib.util.module_from_spec(sp2); sp2.loader.exec_module(r17)
sp3 = importlib.util.spec_from_file_location("p2", r"C:\Users\artem\SeekDeep\Workspaces\1\work\r17_phase2.py")
p2 = importlib.util.module_from_spec(sp3); sp3.loader.exec_module(p2)

OUT = r"C:\Users\artem\SeekDeep\Workspaces\1\work\r17_extreme3.json"
out = []
N = 32
u = None
for kappa in (1e4, 1e6, 1e8):
    af = (lambda k: (lambda x, y: 1.0 if x < 0.5 else k))(kappa)
    S = ex.assemble(N, 2.0, af)
    u, rn, its, dt, hist = ex.newton_direct(S, 2.0, u0=u, tol=1e-12, maxit=60)
    for p in (2.0, 1.6, 1.5, 1.2):
        Sp = ex.assemble(N, p, af)
        u, rn, its, dt, hist = ex.newton_direct(Sp, p, u0=u, tol=1e-12, maxit=60)
        bd = p2.breakdown(Sp["coords"], Sp["tris"], Sp["grads"], Sp["areas"], Sp["cent"],
                          Sp["alpha_e"], u, Sp["fq"], p)
        errE, errp = r17.errors(Sp["coords"], Sp["tris"], Sp["grads"], Sp["areas"], Sp["cent"],
                                Sp["alpha_e"], u, p)
        rho = errp ** p
        pp = p / (p - 1.0)
        rec = {"N": N, "kappa": kappa, "p": p, "residual": rn, "newton_steps": its,
               "err_energy": errE, "rho": rho, "eta_split": bd["split_total"],
               "eta_arith": bd["arith_total"], "eta_harm": bd["harm_total"],
               "eta_unw": bd["unw_total"] if "unw_total" in bd else None,
               "effE_split": bd["split_total"] / errE, "effE_arith": bd["arith_total"] / errE,
               "effR_split": bd["split_total"] / rho ** (1.0 / pp),
               "effR_arith": bd["arith_total"] / rho ** (1.0 / pp),
               "iface_share_arith": bd["arith_iface"] / bd["arith_total"],
               "iface_share_split": bd["split_iface"] / bd["split_total"]}
        out.append(rec)
        print("kap=%.0e p=%.1f res=%.1e steps=%2d | errE=%.3e rho^1/p'=%.3e | effE: split=%.2f arith=%.2f | "
              "effR: split=%.2f arith=%.2f | iface %.2f/%.2f"
              % (kappa, p, rn, its, errE, rho ** (1.0 / pp), rec["effE_split"], rec["effE_arith"],
                 rec["effR_split"], rec["effR_arith"], rec["iface_share_split"], rec["iface_share_arith"]),
              flush=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("saved:", OUT)
