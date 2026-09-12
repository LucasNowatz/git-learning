#!/usr/bin/env python3
"""Reference solution for afterquery/no2-emission-inversion.

The chemical sink saturates with column amount, so the state is not linear in
the emission scales and there is no basis decomposition to exploit.  The
fifteen unknowns are fitted jointly by bounded nonlinear least squares with a
finite-difference Jacobian, forward runs distributed over the episodes.

The retrieval carries an across-track artefact that the quality flag does not
capture.  It is found here the way an analyst finds it: fit once, bin the
normalised residuals by across-track index and episode, flag the positions that
carry a significant systematic offset, drop them, and fit again.  Nothing in
this file reads the generator, its seed or the sealed evaluation values.
"""
import csv, json, os, sys, time
import numpy as np
from netCDF4 import Dataset
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

DATA = "/app/data"
OUT = "/app"
DX = 4000.0
DT = 60.0
NREG = 6
UG_PER_MOL_NO2 = 46.0055e-3 * 1.0e9
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:8.1f}s]", *a, flush=True)


# ================================================================== inputs ==
def load_inputs():
    d = {}
    with Dataset(f"{DATA}/domain.nc") as ds:
        d["x"] = np.asarray(ds["x"][:]); d["y"] = np.asarray(ds["y"][:])
        d["area"] = np.asarray(ds["cell_area"][:])
        d["gamma"] = np.deg2rad(np.asarray(ds["grid_convergence"][:]))
        d["region_id"] = np.asarray(ds["region_id"][:])
    with Dataset(f"{DATA}/emissions.nc") as ds:
        d["road"] = (np.asarray(ds["road_prior"][:])
                     / (ds.molar_mass_no2_kg_per_mol * 1e6 * 3600.0))
        d["fixed"] = (np.asarray(ds["fixed_sources"][:])
                      / (ds.molar_mass_n_kg_per_mol * 1e6 * 3600.0))
        d["road_factor"] = np.asarray(ds["road_time_factor"][:])
        d["fixed_factor"] = np.asarray(ds["fixed_time_factor"][:])
        d["diurnal"] = np.asarray(ds["road_diurnal_factor"][:])
    with Dataset(f"{DATA}/meteorology.nc") as ds:
        d["K"] = float(ds.horizontal_diffusivity_m2_s)
        d["xm"] = np.asarray(ds["xm"][:]); d["ym"] = np.asarray(ds["ym"][:])
        d["mtime"] = np.asarray(ds["time"][:])
        zb = np.asarray(ds["level_bounds"][:])
        u = np.asarray(ds["u_east"][:], float); v = np.asarray(ds["v_north"][:], float)
        h = np.asarray(ds["blh"][:], float)
        d["blh"] = h
        d["f_no2"] = np.asarray(ds["f_no2"][:], float)
        d["photo"] = np.asarray(ds["photolysis_factor"][:], float)
        nu = np.zeros_like(h); nv = np.zeros_like(h); den = np.zeros_like(h)
        for lv in range(zb.shape[0]):
            th = np.clip(np.minimum(h, zb[lv, 1]) - zb[lv, 0], 0.0, None)
            if lv == 0:
                th = np.maximum(th, 1.0)
            den += th; nu += th * u[:, :, lv]; nv += th * v[:, :, lv]
        d["ue"] = nu / den; d["vn"] = nv / den
        d["t0"] = np.asarray(ds["episode_start"][:])
        d["t1"] = np.asarray(ds["episode_end"][:])
        d["tsat"] = np.asarray(ds["overpass_time"][:])
        d["regime"] = [str(s) for s in ds["wind_regime"][:]]
    with open(f"{DATA}/data_manifest.json") as fh:
        d["man"] = json.load(fh)
    return d


def sigma_total(value, floor_meas, floor_repr, rel_repr):
    return np.sqrt(floor_meas ** 2 + floor_repr ** 2 + (rel_repr * np.abs(value)) ** 2)


# ================================================================ geometry ==
def clip_rect(poly, x0, x1, y0, y1):
    def half(pts, sel, cut):
        out = []
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            sa, sb = sel(a), sel(b)
            if sa:
                out.append(a)
                if not sb:
                    out.append(cut(a, b))
            elif sb:
                out.append(cut(a, b))
        return out
    P = [tuple(p) for p in poly]
    fx = lambda a, b, v: (v, a[1] + (v - a[0]) * (b[1] - a[1]) / (b[0] - a[0]))
    fy = lambda a, b, v: (a[0] + (v - a[1]) * (b[0] - a[0]) / (b[1] - a[1]), v)
    for sel, cut in ((lambda p: p[0] >= x0, lambda a, b: fx(a, b, x0)),
                     (lambda p: p[0] <= x1, lambda a, b: fx(a, b, x1)),
                     (lambda p: p[1] >= y0, lambda a, b: fy(a, b, y0)),
                     (lambda p: p[1] <= y1, lambda a, b: fy(a, b, y1))):
        P = half(P, sel, cut)
        if not P:
            return P
    return P


def poly_area(P):
    if len(P) < 3:
        return 0.0
    s = 0.0
    for i in range(len(P)):
        x1, y1 = P[i]; x2, y2 = P[(i + 1) % len(P)]
        s += x1 * y2 - x2 * y1
    return 0.5 * abs(s)


def footprint_matrix(cx, cy, nx, ny):
    r, c, v = [], [], []
    for k in range(cx.shape[0]):
        poly = np.stack([cx[k], cy[k]], 1)
        i0 = max(int(poly[:, 0].min() // DX), 0)
        i1 = min(int(np.ceil(poly[:, 0].max() / DX)), nx)
        j0 = max(int(poly[:, 1].min() // DX), 0)
        j1 = min(int(np.ceil(poly[:, 1].max() / DX)), ny)
        cc, vv = [], []
        for j in range(j0, j1):
            for i in range(i0, i1):
                a = poly_area(clip_rect(poly, i * DX, (i + 1) * DX, j * DX, (j + 1) * DX))
                if a > 0:
                    cc.append(j * nx + i); vv.append(a)
        tot = float(sum(vv))
        for cj, vj in zip(cc, vv):
            r.append(k); c.append(cj); v.append(vj / tot)
    return coo_matrix((v, (r, c)), shape=(cx.shape[0], nx * ny)).tocsr()


# ================================================================== solver ==
class Bilin:
    def __init__(self, sx, sy, tx, ty):
        self.i = np.clip(np.searchsorted(sx, tx) - 1, 0, len(sx) - 2)
        self.j = np.clip(np.searchsorted(sy, ty) - 1, 0, len(sy) - 2)
        self.a = np.clip((tx - sx[self.i]) / np.diff(sx)[self.i], 0, 1)[None, :]
        self.b = np.clip((ty - sy[self.j]) / np.diff(sy)[self.j], 0, 1)[:, None]

    def __call__(self, F):
        i, j, a, b = self.i, self.j, self.a, self.b
        return ((1 - a) * (1 - b) * F[np.ix_(j, i)] + a * (1 - b) * F[np.ix_(j, i + 1)]
                + (1 - a) * b * F[np.ix_(j + 1, i)] + a * b * F[np.ix_(j + 1, i + 1)])


def vanleer(a, b):
    p = a * b
    o = np.zeros_like(a)
    m = p > 0
    o[m] = 2 * p[m] / (a[m] + b[m])
    return o


def phi(zeta, z0):
    z = np.asarray(zeta, float)
    return np.where((z >= 0) & (z <= 1),
                    np.exp(-z / z0) / (z0 * (1 - np.exp(-1 / z0))), 0.0)


def diurnal(t, tab):
    h = (np.asarray(t, float) % 86400.0) / 3600.0
    i = np.floor(h).astype(int) % 24
    w = h - np.floor(h)
    return tab[i] * (1 - w) + tab[(i + 1) % 24] * w


class Model:
    def __init__(self, d):
        self.d = d
        self.nx, self.ny = len(d["x"]), len(d["y"])
        self.ip = Bilin(d["xm"], d["ym"], d["x"], d["y"])
        XT, YT = np.meshgrid(d["x"] / (self.nx * DX), d["y"] / (self.ny * DX))
        self.XT, self.YT = XT, YT

    def _met(self, e, t):
        tt = self.d["mtime"][e]
        k = int(np.clip(np.searchsorted(tt, t) - 1, 0, len(tt) - 2))
        w = (t - tt[k]) / (tt[k + 1] - tt[k])
        L = lambda A: (1 - w) * A[k] + w * A[k + 1]
        return (self.ip(L(self.d["ue"][e])), self.ip(L(self.d["vn"][e])),
                self.ip(L(self.d["blh"][e])), self.ip(L(self.d["f_no2"][e])),
                self.ip(L(self.d["photo"][e])))

    def run(self, e, P, W, sta):
        d = self.d
        nx, ny = self.nx, self.ny
        bg = P["b0"] + P["bx"] * self.XT + P["by"] * self.YT
        bgp = np.pad(bg, 2, mode="edge")
        c = bg.copy()
        emis_road = np.tensordot(P["s"], d["road"], axes=(0, 0))
        emis_fix = P["fixed_scale"] * float(d["fixed_factor"][e]) * d["fixed"]
        cd, sd = np.cos(np.deg2rad(P["delta"])), np.sin(np.deg2rad(P["delta"]))
        cg, sg = np.cos(d["gamma"]), np.sin(d["gamma"])
        t = float(d["t0"][e]); tend = float(d["t1"][e]); tsat = float(d["tsat"][e])
        n = int(round((tend - t) / DT))
        y_sat = None
        acc = np.zeros(len(sta["ta"])); cnt = np.zeros(len(sta["ta"]))
        si = np.minimum((sta["x"] // DX).astype(int), nx - 1)
        sj = np.minimum((sta["y"] // DX).astype(int), ny - 1)
        done = False
        for _ in range(n):
            tm = t + 0.5 * DT
            u, v, h, f, ph = self._met(e, tm)
            ue = P["a"] * (cd * u - sd * v); vn = P["a"] * (sd * u + cd * v)
            ug = cg * ue + sg * vn; vg = -sg * ue + cg * vn
            uf = np.empty((ny, nx + 1))
            uf[:, 1:-1] = 0.5 * (ug[:, :-1] + ug[:, 1:]); uf[:, 0] = ug[:, 0]; uf[:, -1] = ug[:, -1]
            vf = np.empty((ny + 1, nx))
            vf[1:-1] = 0.5 * (vg[:-1] + vg[1:]); vf[0] = vg[0]; vf[-1] = vg[-1]
            em = emis_fix + float(diurnal(tm, d["diurnal"])) * float(d["road_factor"][e]) * emis_road

            p = bgp.copy(); p[2:-2, 2:-2] = c
            a_ = p[2:-2, :]
            s_ = vanleer(a_[:, 1:-1] - a_[:, :-2], a_[:, 2:] - a_[:, 1:-1])
            cl = a_[:, 1:-2] + 0.5 * s_[:, :-1]; cr = a_[:, 2:-1] - 0.5 * s_[:, 1:]
            cx = np.where(uf >= 0, cl, cr)
            cx[:, 0] = np.where(uf[:, 0] >= 0, a_[:, 1], a_[:, 2])
            cx[:, -1] = np.where(uf[:, -1] >= 0, a_[:, -3], a_[:, -2])
            fx = uf * cx
            b_ = p[:, 2:-2]
            s_ = vanleer(b_[1:-1] - b_[:-2], b_[2:] - b_[1:-1])
            cl = b_[1:-2] + 0.5 * s_[:-1]; cr = b_[2:-1] - 0.5 * s_[1:]
            cy = np.where(vf >= 0, cl, cr)
            cy[0] = np.where(vf[0] >= 0, b_[1], b_[2])
            cy[-1] = np.where(vf[-1] >= 0, b_[-3], b_[-2])
            fy = vf * cy
            div = (fx[:, 1:] - fx[:, :-1]) / DX + (fy[1:] - fy[:-1]) / DX
            gx = np.zeros((ny, nx + 1)); gx[:, 1:-1] = (c[:, 1:] - c[:, :-1]) / DX
            gy = np.zeros((ny + 1, nx)); gy[1:-1] = (c[1:] - c[:-1]) / DX
            lap = d["K"] * ((gx[:, 1:] - gx[:, :-1]) / DX + (gy[1:] - gy[:-1]) / DX)
            cprev = c
            cn = c + DT * (-div + lap + em)
            k = ph / (P["tau0"] * (1.0 + np.maximum(cn, 0.0) / P["c_ref"]))
            c = cn * np.exp(-DT * k)

            if (not done) and (t + DT >= tsat):
                fr = (tsat - t) / DT
                cs = (1 - fr) * cprev + fr * c
                _, _, _, fs, _ = self._met(e, tsat)
                if W is not None and W.shape[0]:
                    y_sat = W @ (cs * fs).reshape(-1)
                done = True
            sel = (sta["ta"] <= t + DT) & (t + DT <= sta["tb"])
            if sel.any():
                z = sta["h"][sel] / h[sj[sel], si[sel]]
                acc[sel] += (f[sj[sel], si[sel]] * c[sj[sel], si[sel]]
                             * phi(z, P["zeta0"]) / h[sj[sel], si[sel]]) * UG_PER_MOL_NO2
                cnt[sel] += 1.0
            t += DT
        return y_sat, acc / np.maximum(cnt, 1.0)


# ================================================================= fitting ==
# z = [s1..s6, log tau0, log c_ref, a, delta/10, b0*1e5, bx*1e5, by*1e5,
#      fixed_scale, zeta0]
LO = np.array([0.30] * 6 + [np.log(1800.0), np.log(1.0e-5), 0.70, -2.0,
                            0.0, -8.0, -8.0, 0.40, 0.12])
HI = np.array([2.50] * 6 + [np.log(28800.0), np.log(1.0e-3), 1.30, 2.0,
                            20.0, 8.0, 8.0, 2.20, 0.80])


def unpack(z):
    return dict(s=np.asarray(z[:6], float), tau0=float(np.exp(z[6])),
                c_ref=float(np.exp(z[7])), a=float(z[8]), delta=float(z[9] * 10.0),
                b0=float(z[10] * 1e-5), bx=float(z[11] * 1e-5), by=float(z[12] * 1e-5),
                fixed_scale=float(z[13]), zeta0=float(z[14]))


_MODEL = None
_BLOCKS = None
_POOL = None


def _work(arg):
    e, z = arg
    b = _BLOCKS[e]
    return _MODEL.run(e, unpack(z), b["W"], b["sta"])


def predict(eps, z):
    args = [(e, z) for e in eps]
    if _POOL is not None:
        return list(_POOL.map(_work, args, chunksize=1))
    return [_work(a) for a in args]


def residuals(z, eps, mask):
    out = []
    for e, (ys, yt) in zip(eps, predict(eps, z)):
        b = _BLOCKS[e]
        m = mask[e]
        out.append(((ys[m] - b["sat"]["value"][m]) / b["sat"]["sig"][m]))
        out.append((yt - b["sta"]["value"]) / b["sta"]["sig"])
    return np.concatenate(out)


def load_training(d, model):
    rep = d["man"]["representation_error"]
    fs, rs = rep["satellite_floor_mol_m2"], rep["satellite_relative"]
    ft, rt = rep["station_floor_ug_m3"], rep["station_relative"]
    blocks = {}
    sdir = f"{DATA}/swaths"
    for fn in sorted(os.listdir(sdir)):
        if not fn.endswith(".nc"):
            continue
        with Dataset(f"{sdir}/{fn}") as ds:
            e = int(ds.episode_id)
            col = ds["no2_column"][:]
            qa = np.asarray(ds["qa_value"][:])
            good = np.where((~np.ma.getmaskarray(col)) & (qa >= float(ds.qa_acceptance_threshold)))[0]
            val = np.asarray(col)[good]
            sig_m = np.asarray(ds["no2_column_uncertainty"][:])[good]
            blocks[e] = dict(sat=dict(
                obs_id=np.array([str(s) for s in ds["obs_id"][:]])[good],
                value=val, sig=sigma_total(val, sig_m, fs, rs),
                gp=np.asarray(ds["ground_pixel"][:])[good],
                sens=np.asarray(ds["vertical_sensitivity"][:])[good]))
            blocks[e]["W"] = footprint_matrix(np.asarray(ds["corner_x"][:])[good],
                                              np.asarray(ds["corner_y"][:])[good],
                                              model.nx, model.ny)
            blocks[e]["W"] = _scale_rows(blocks[e]["W"], blocks[e]["sat"]["sens"])
    rows = {}
    with open(f"{DATA}/stations.csv") as fh:
        for r in csv.DictReader(fh):
            rows.setdefault(int(r["episode_id"]), []).append(r)
    for e, rr in rows.items():
        val = np.array([float(r["no2_ug_m3"]) for r in rr])
        sig_m = np.array([float(r["sigma_ug_m3"]) for r in rr])
        blocks[e]["sta"] = dict(
            ta=np.array([float(r["interval_start_utc"]) for r in rr]),
            tb=np.array([float(r["interval_end_utc"]) for r in rr]),
            x=np.array([float(r["x_m"]) for r in rr]),
            y=np.array([float(r["y_m"]) for r in rr]),
            h=np.array([float(r["inlet_height_m"]) for r in rr]),
            value=val, sig=sigma_total(val, sig_m, ft, rt))
    return blocks, (fs, rs, ft, rt)


def _scale_rows(W, f):
    W = W.tocsr().copy()
    for i in range(W.shape[0]):
        W.data[W.indptr[i]:W.indptr[i + 1]] *= f[i]
    return W


def detect_row_anomaly(z, eps, mask):
    """Flag across-track positions whose residuals carry a systematic offset.

    Robust to a poor current fit: each position is compared with the median
    over positions within the same episode, and the scale is a median absolute
    deviation, so a global misfit shifts every position together and flags
    nothing.  A detector artefact occupies a contiguous run of positions, so
    only maximal runs of at least four flagged positions are acted on.  The
    retrieval reads high where it is affected, which makes the model minus
    observation residual read low, so the test is one sided.
    """
    flags = {}
    for e, (ys, _) in zip(eps, predict(eps, z)):
        b = _BLOCKS[e]
        keep = mask[e]
        r = (ys - b["sat"]["value"]) / b["sat"]["sig"]
        gp = b["sat"]["gp"]
        pos, mu = [], []
        for p in np.unique(gp):
            m = (gp == p) & keep
            if m.sum() >= 5:
                pos.append(int(p)); mu.append(float(np.median(r[m])))
        if len(pos) < 8:
            flags[e] = []
            continue
        pos = np.array(pos); mu = np.array(mu)
        # The reference level is the upper quartile, not the median: an
        # artefact can occupy a third of the detector, which drags the median
        # and inflates a median absolute deviation computed about it.  The
        # scale comes from the unaffected side only.
        ref = float(np.percentile(mu, 75))
        up = mu[mu >= ref]
        scale = 1.4826 * float(np.median(np.abs(up - ref))) if up.size else 0.0
        cut = max(0.80, 4.0 * scale)
        hit = (mu - ref) < -cut
        runs, cur = [], []
        for i, p in enumerate(pos):
            if hit[i] and (not cur or p == cur[-1] + 1):
                cur.append(int(p))
            elif hit[i]:
                runs.append(cur); cur = [int(p)]
            elif cur:
                runs.append(cur); cur = []
        if cur:
            runs.append(cur)
        band = max((c for c in runs if len(c) >= 4), key=len, default=[])
        flags[e] = band
    return flags


def main():
    global _MODEL, _BLOCKS, _POOL
    log("loading inputs")
    d = load_inputs()
    model = Model(d)
    _MODEL = model
    blocks, errmodel = load_training(d, model)
    _BLOCKS = blocks
    eps = sorted(blocks)
    ncpu = max(1, min(4, os.cpu_count() or 1))
    if ncpu > 1:
        import multiprocessing as mp
        _POOL = mp.get_context("fork").Pool(ncpu)
        log(f"{ncpu} worker processes")
    mask = {e: np.ones(len(blocks[e]["sat"]["value"]), bool) for e in eps}
    n_obs = sum(mask[e].sum() + len(blocks[e]["sta"]["value"]) for e in eps)
    log(f"episodes {eps}, {n_obs} accepted observations")

    z0 = np.array([1.0] * 6 + [np.log(9000.0), np.log(1.5e-4), 1.0, 0.0,
                              2.0, 0.0, 0.0, 1.0, 0.35])

    def fit(z_start, mask_, label, maxfev=900):
        r = least_squares(residuals, z_start, args=(eps, mask_), bounds=(LO, HI),
                          method="trf", x_scale="jac", diff_step=6e-3,
                          xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=maxfev,
                          verbose=0)
        n = len(r.fun)
        log(f"{label}: chi2/n = {2 * r.cost / n:.4f} over {n} residuals, "
            f"nfev={r.nfev}")
        return r

    log("first pass over all accepted retrievals")
    r = fit(z0, mask, "pass 1")

    for round_ in (1, 2, 3):
        flags = detect_row_anomaly(r.x, eps, mask)
        tot = 0
        for e in eps:
            if flags[e]:
                drop = np.isin(blocks[e]["sat"]["gp"], flags[e]) & mask[e]
                mask[e] = mask[e] & ~drop
                tot += int(drop.sum())
                if drop.sum():
                    log(f"  round {round_} episode {e:2d}: across-track "
                        f"{flags[e][0]}-{flags[e][-1]}, {int(drop.sum())} retrievals")
        log(f"screening round {round_}: dropped {tot} retrievals")
        if tot == 0:
            break
        r = fit(r.x, mask, f"pass {round_ + 1}")
    kept = sum(int(mask[e].sum()) for e in eps)
    log(f"retained {kept} of {sum(len(mask[e]) for e in eps)} retrievals")
    best = r
    log("independent restart to check for a local minimum")
    z_alt = np.array([1.8, 0.6, 1.4, 1.4, 0.8, 1.6, np.log(20000.0), np.log(4.0e-5),
                      1.25, -1.6, 3.0, -2.0, 2.0, 1.7, 0.6])
    r3 = fit(z_alt, mask, "restart")
    if r3.cost < best.cost:
        best = r3
    P = unpack(best.x)
    log(f"fit: tau0={P['tau0']:.1f}s c_ref={P['c_ref']:.4e} a={P['a']:.4f} "
        f"delta={P['delta']:+.3f} fixed={P['fixed_scale']:.4f} zeta0={P['zeta0']:.4f}")
    log(f"     s={np.round(P['s'], 4).tolist()}")
    log(f"     bg=({P['b0']:.4e},{P['bx']:.4e},{P['by']:.4e})")
    export(d, model, P)
    return 0


def export(d, model, P):
    log("predicting held-out queries")
    with Dataset(f"{DATA}/prediction_queries.nc") as ds:
        qs = dict(obs_id=np.array([str(s) for s in ds["sat_obs_id"][:]]),
                  ep=np.asarray(ds["sat_episode_id"][:]),
                  cx=np.asarray(ds["sat_corner_x"][:]),
                  cy=np.asarray(ds["sat_corner_y"][:]),
                  sens=np.asarray(ds["sat_vertical_sensitivity"][:]))
        qt = dict(obs_id=np.array([str(s) for s in ds["sta_obs_id"][:]]),
                  ep=np.asarray(ds["sta_episode_id"][:]),
                  x=np.asarray(ds["sta_x"][:]), y=np.asarray(ds["sta_y"][:]),
                  h=np.asarray(ds["sta_inlet_height_m"][:]),
                  ta=np.asarray(ds["sta_interval_start_utc"][:]),
                  tb=np.asarray(ds["sta_interval_end_utc"][:]))
    rows = []
    for e in sorted(set(qs["ep"].tolist()) | set(qt["ep"].tolist())):
        ms = qs["ep"] == e; mt = qt["ep"] == e
        W = _scale_rows(footprint_matrix(qs["cx"][ms], qs["cy"][ms], model.nx, model.ny),
                        qs["sens"][ms])
        spec = dict(ta=qt["ta"][mt], tb=qt["tb"][mt], x=qt["x"][mt],
                    y=qt["y"][mt], h=qt["h"][mt])
        ys, yt = model.run(int(e), P, W, spec)
        rows += [(o, "satellite", float(v), "mol m-2") for o, v in zip(qs["obs_id"][ms], ys)]
        rows += [(o, "station", float(v), "ug m-3") for o, v in zip(qt["obs_id"][mt], yt)]
        log(f"  episode {e}: {int(ms.sum())} satellite + {int(mt.sum())} station")

    area = float(d["area"][0, 0])
    base = d["road"].sum(axis=(1, 2)) * area
    totals = {str(e): float((P["s"] * base).sum() * float(d["road_factor"][e])
                            * float(diurnal(d["tsat"][e], d["diurnal"])))
              for e in range(len(d["t0"]))}
    with open(f"{OUT}/result.json", "w") as fh:
        json.dump(dict(schema_version="1.0",
                       region_ids=[int(v) for v in d["region_id"]],
                       emission_scale=[float(v) for v in P["s"]],
                       reference_loss_time_s=P["tau0"],
                       loss_saturation_column=P["c_ref"],
                       loss_saturation_units="mol m-2",
                       wind_speed_scale=P["a"], wind_rotation_deg=P["delta"],
                       background_coefficients=[P["b0"], P["bx"], P["by"]],
                       background_units="mol m-2",
                       fixed_source_scale=P["fixed_scale"],
                       vertical_shape_zeta0=P["zeta0"],
                       integrated_road_mol_s=totals,
                       integrated_road_units="mol s-1"), fh, indent=2)
    with Dataset(f"{OUT}/posterior.nc", "w", format="NETCDF4") as ds:
        ds.schema_version = "1.0"
        ds.createDimension("x", model.nx); ds.createDimension("y", model.ny)
        ds.createDimension("region", NREG); ds.createDimension("nv", 2)
        v = ds.createVariable("x", "f8", ("x",)); v[:] = d["x"]; v.units = "m"
        v = ds.createVariable("y", "f8", ("y",)); v[:] = d["y"]; v.units = "m"
        v = ds.createVariable("x_bnds", "f8", ("x", "nv"))
        v[:] = np.stack([d["x"] - DX / 2, d["x"] + DX / 2], 1)
        v = ds.createVariable("y_bnds", "f8", ("y", "nv"))
        v[:] = np.stack([d["y"] - DX / 2, d["y"] + DX / 2], 1)
        v = ds.createVariable("region_id", "i4", ("region",)); v[:] = d["region_id"]
        v = ds.createVariable("corrected_road_flux", "f8", ("region", "y", "x"))
        v[:] = P["s"][:, None, None] * d["road"]; v.units = "mol m-2 s-1"
        v = ds.createVariable("total_source_flux", "f8", ("y", "x"))
        v[:] = (P["s"][:, None, None] * d["road"]).sum(0) + P["fixed_scale"] * d["fixed"]
        v.units = "mol m-2 s-1"
        c = ds.createVariable("crs", "i4"); c.epsg_code = "EPSG:3035"
    with open(f"{OUT}/predicted_observations.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["obs_id", "instrument_type", "predicted_value", "unit"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.10e}", r[3]])
    log(f"wrote {len(rows)} predictions")


if __name__ == "__main__":
    sys.exit(main())
