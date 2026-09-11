#!/usr/bin/env python3
"""Reference solution for afterquery/no2-emission-inversion.

Strategy: for fixed (lifetime, wind speed scale, wind rotation) the transport
model is linear in the six road scale factors and the three background
coefficients, so the inner problem is a bounded weighted linear least squares
over ten pre-integrated basis states.  Only the three nonlinear parameters are
searched, first on a coarse grid and then with Nelder-Mead.

This file has no access to the generator or to tests/.
"""
import csv, json, os, sys, time
import numpy as np
from netCDF4 import Dataset
from scipy.optimize import lsq_linear, minimize

DATA = "/app/data"
OUT = "/app"
DX = 4000.0
DT = 60.0
N_REGIONS = 6
T_START = time.time()


def log(*a):
    print(f"[{time.time()-T_START:8.1f}s]", *a, flush=True)


# --------------------------------------------------------------- data input --
def load_inputs():
    d = {}
    with Dataset(f"{DATA}/domain.nc") as ds:
        d["x"] = ds["x"][:].data
        d["y"] = ds["y"][:].data
        d["area"] = ds["cell_area"][:].data
        d["gamma"] = ds["grid_convergence"][:].data
        d["region_id"] = ds["region_id"][:].data
    with Dataset(f"{DATA}/emissions.nc") as ds:
        m_no2 = ds.molar_mass_no2_kg_per_mol
        m_n = ds.molar_mass_n_kg_per_mol
        road_kg = ds["road_prior"][:].data           # kg NO2-equiv km-2 h-1
        fixed_kg = ds["fixed_sources"][:].data       # kg N km-2 h-1
        d["road"] = road_kg / (m_no2 * 1.0e6 * 3600.0)
        d["fixed"] = fixed_kg / (m_n * 1.0e6 * 3600.0)
        d["road_factor"] = ds["road_time_factor"][:].data
        d["fixed_factor"] = ds["fixed_time_factor"][:].data
        d["diurnal"] = ds["road_diurnal_factor"][:].data
    with Dataset(f"{DATA}/meteorology.nc") as ds:
        d["K"] = float(ds.horizontal_diffusivity_m2_s)
        d["zeta0"] = float(ds.vertical_shape_zeta0)
        d["xm"] = ds["xm"][:].data
        d["ym"] = ds["ym"][:].data
        d["mtime"] = ds["time"][:].data
        d["zb"] = ds["level_bounds"][:].data
        d["u_east"] = ds["u_east"][:].data.astype(np.float64)
        d["v_north"] = ds["v_north"][:].data.astype(np.float64)
        d["blh"] = ds["blh"][:].data.astype(np.float64)
        d["f_no2"] = ds["f_no2"][:].data.astype(np.float64)
        d["t0"] = ds["episode_start"][:].data
        d["t1"] = ds["episode_end"][:].data
        d["t_sat"] = ds["overpass_time"][:].data
        d["regime"] = [str(s) for s in ds["wind_regime"][:]]
    return d


def load_training_obs(d):
    sat, sta = [], []
    sdir = f"{DATA}/swaths"
    for fn in sorted(os.listdir(sdir)):
        if not fn.endswith(".nc"):
            continue
        with Dataset(f"{sdir}/{fn}") as ds:
            e = int(ds.episode_id)
            thr = float(ds.qa_acceptance_threshold)
            col = ds["no2_column"][:]                       # masked, unpacked
            qa = ds["qa_value"][:].data
            good = (~np.ma.getmaskarray(col)) & (qa >= thr)
            g = np.where(good)[0]
            sat.append(dict(
                episode=e, idx=g,
                obs_id=np.array([str(s) for s in ds["obs_id"][:]])[g],
                value=np.asarray(col)[g],
                sigma=ds["no2_column_uncertainty"][:].data[g],
                cx=ds["corner_x"][:].data[g], cy=ds["corner_y"][:].data[g],
                sens=ds["vertical_sensitivity"][:].data[g]))
    with open(f"{DATA}/stations.csv") as fh:
        for row in csv.DictReader(fh):
            sta.append(row)
    return sat, sta


def load_queries():
    with Dataset(f"{DATA}/prediction_queries.nc") as ds:
        sat = dict(obs_id=np.array([str(s) for s in ds["sat_obs_id"][:]]),
                   episode=ds["sat_episode_id"][:].data,
                   cx=ds["sat_corner_x"][:].data, cy=ds["sat_corner_y"][:].data,
                   sens=ds["sat_vertical_sensitivity"][:].data,
                   sigma=ds["sat_uncertainty"][:].data)
        sta = dict(obs_id=np.array([str(s) for s in ds["sta_obs_id"][:]]),
                   station=np.array([str(s) for s in ds["sta_station_id"][:]]),
                   episode=ds["sta_episode_id"][:].data,
                   x=ds["sta_x"][:].data, y=ds["sta_y"][:].data,
                   h=ds["sta_inlet_height_m"][:].data,
                   ta=ds["sta_interval_start_utc"][:].data,
                   tb=ds["sta_interval_end_utc"][:].data,
                   sigma=ds["sta_uncertainty"][:].data)
    return sat, sta


# ----------------------------------------------------------------- geometry --
def clip_box(poly, x0, x1, y0, y1):
    def half(pts, keep, cut):
        out = []
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            ka, kb = keep(a), keep(b)
            if ka:
                out.append(a)
                if not kb:
                    out.append(cut(a, b))
            elif kb:
                out.append(cut(a, b))
        return out
    P = [tuple(p) for p in poly]
    cx = lambda a, b, v: (v, a[1] + (v - a[0]) / (b[0] - a[0]) * (b[1] - a[1]))
    cy = lambda a, b, v: (a[0] + (v - a[1]) / (b[1] - a[1]) * (b[0] - a[0]), v)
    P = half(P, lambda p: p[0] >= x0, lambda a, b: cx(a, b, x0))
    if not P: return P
    P = half(P, lambda p: p[0] <= x1, lambda a, b: cx(a, b, x1))
    if not P: return P
    P = half(P, lambda p: p[1] >= y0, lambda a, b: cy(a, b, y0))
    if not P: return P
    return half(P, lambda p: p[1] <= y1, lambda a, b: cy(a, b, y1))


def shoelace(P):
    if len(P) < 3: return 0.0
    s = 0.0
    for i in range(len(P)):
        x1, y1 = P[i]; x2, y2 = P[(i + 1) % len(P)]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def footprint_matrix(cx, cy, nx, ny):
    """Sparse area-weight rows mapping the flattened grid to footprint means."""
    rows, cols, vals = [], [], []
    for k in range(len(cx)):
        poly = np.stack([cx[k], cy[k]], axis=1)
        i0 = max(int(np.floor(poly[:, 0].min() / DX)), 0)
        i1 = min(int(np.ceil(poly[:, 0].max() / DX)), nx)
        j0 = max(int(np.floor(poly[:, 1].min() / DX)), 0)
        j1 = min(int(np.ceil(poly[:, 1].max() / DX)), ny)
        c_, v_ = [], []
        for j in range(j0, j1):
            for i in range(i0, i1):
                a = shoelace(clip_box(poly, i * DX, (i + 1) * DX, j * DX, (j + 1) * DX))
                if a > 0.0:
                    c_.append(j * nx + i); v_.append(a)
        tot = sum(v_)
        for c, v in zip(c_, v_):
            rows.append(k); cols.append(c); vals.append(v / tot)
    from scipy.sparse import coo_matrix
    return coo_matrix((vals, (rows, cols)), shape=(len(cx), nx * ny)).tocsr()


# ------------------------------------------------------------------ physics --
def phi(zeta, z0):
    z = np.asarray(zeta, float)
    return np.where((z >= 0) & (z <= 1),
                    np.exp(-z / z0) / (z0 * (1 - np.exp(-1 / z0))), 0.0)


def bl_wind(d):
    zb = d["zb"]
    h = d["blh"]
    num_u = np.zeros_like(h); num_v = np.zeros_like(h); den = np.zeros_like(h)
    for lv in range(zb.shape[0]):
        th = np.clip(np.minimum(h, zb[lv, 1]) - zb[lv, 0], 0.0, None)
        if lv == 0:
            th = np.maximum(th, 1.0)
        den += th
        num_u += th * d["u_east"][:, :, lv]
        num_v += th * d["v_north"][:, :, lv]
    return num_u / den, num_v / den


class Interp:
    def __init__(self, sx, sy, dx_, dy_):
        self.ix = np.clip(np.searchsorted(sx, dx_) - 1, 0, len(sx) - 2)
        self.iy = np.clip(np.searchsorted(sy, dy_) - 1, 0, len(sy) - 2)
        self.tx = np.clip((dx_ - sx[self.ix]) / (sx[self.ix + 1] - sx[self.ix]), 0, 1)
        self.ty = np.clip((dy_ - sy[self.iy]) / (sy[self.iy + 1] - sy[self.iy]), 0, 1)

    def __call__(self, f):
        ix, iy, tx, ty = self.ix, self.iy, self.tx, self.ty
        a = f[np.ix_(iy, ix)]; b = f[np.ix_(iy, ix + 1)]
        c = f[np.ix_(iy + 1, ix)]; e = f[np.ix_(iy + 1, ix + 1)]
        TX = tx[None, :]; TY = ty[:, None]
        return (1 - TX) * (1 - TY) * a + TX * (1 - TY) * b + \
               (1 - TX) * TY * c + TX * TY * e


def vl(a, b):
    p = a * b
    out = np.zeros_like(a)
    m = p > 0
    out[m] = 2 * p[m] / (a[m] + b[m])
    return out


def step(c, bgp, ufx, vfy, emis, dt, tau, K, nx, ny):
    nb = c.shape[0]
    p = bgp.copy()
    p[:, 2:-2, 2:-2] = c

    a = p[:, 2:-2, :]
    s = vl(a[..., 1:-1] - a[..., :-2], a[..., 2:] - a[..., 1:-1])
    cl = a[..., 1:-2] + 0.5 * s[..., :-1]
    cr = a[..., 2:-1] - 0.5 * s[..., 1:]
    cx = np.where(ufx >= 0, cl, cr)
    cx[..., 0] = np.where(ufx[..., 0] >= 0, a[..., 1], a[..., 2])
    cx[..., -1] = np.where(ufx[..., -1] >= 0, a[..., -3], a[..., -2])
    fx = ufx * cx

    b = p[:, :, 2:-2]
    s = vl(b[:, 1:-1] - b[:, :-2], b[:, 2:] - b[:, 1:-1])
    cl = b[:, 1:-2] + 0.5 * s[:, :-1]
    cr = b[:, 2:-1] - 0.5 * s[:, 1:]
    cy = np.where(vfy >= 0, cl, cr)
    cy[:, 0] = np.where(vfy[:, 0] >= 0, b[:, 1], b[:, 2])
    cy[:, -1] = np.where(vfy[:, -1] >= 0, b[:, -3], b[:, -2])
    fy = vfy * cy

    div = (fx[..., 1:] - fx[..., :-1]) / DX + (fy[:, 1:] - fy[:, :-1]) / DX
    gx = np.zeros((nb, ny, nx + 1)); gx[..., 1:-1] = (c[..., 1:] - c[..., :-1]) / DX
    gy = np.zeros((nb, ny + 1, nx)); gy[:, 1:-1] = (c[:, 1:] - c[:, :-1]) / DX
    lap = K * ((gx[..., 1:] - gx[..., :-1]) / DX + (gy[:, 1:] - gy[:, :-1]) / DX)
    return (c + dt * (-div + lap + emis)) * np.exp(-dt / tau)


UG_PER_MOL = 46.0055e-3 * 1.0e9          # mol m-3 -> ug m-3 for NO2


def diurnal(t, table):
    h = (np.asarray(t, float) % 86400.0) / 3600.0
    i0 = np.floor(h).astype(int) % 24
    w = h - np.floor(h)
    return table[i0] * (1 - w) + table[(i0 + 1) % 24] * w


class Model:
    def __init__(self, d):
        self.d = d
        self.nx, self.ny = len(d["x"]), len(d["y"])
        self.ip = Interp(d["xm"], d["ym"], d["x"], d["y"])
        self.ue, self.vn = bl_wind(d)
        self.gam = np.deg2rad(d["gamma"])
        xt = d["x"] / (self.nx * DX)
        yt = d["y"] / (self.ny * DX)
        XT, YT = np.meshgrid(xt, yt)
        self.bg_basis = np.stack([1 - XT - YT, XT, YT])       # corner parametrisation
        self.nb = 1 + N_REGIONS + 3

    def _met(self, e, t):
        tt = self.d["mtime"][e]
        k = int(np.clip(np.searchsorted(tt, t) - 1, 0, len(tt) - 2))
        w = (t - tt[k]) / (tt[k + 1] - tt[k])
        L = lambda A: (1 - w) * A[k] + w * A[k + 1]
        return (self.ip(L(self.ue[e])), self.ip(L(self.vn[e])),
                self.ip(L(self.d["blh"][e])), self.ip(L(self.d["f_no2"][e])))

    def run(self, e, tau, a_sc, delta, sat_W=None, sta=None):
        """Integrate the basis states for episode e.

        Returns (A_sat, A_sta) with a trailing basis axis.
        """
        d = self.d
        nx, ny, nb = self.nx, self.ny, self.nb
        bg = np.zeros((nb, ny, nx))
        bg[1 + N_REGIONS:] = self.bg_basis
        bgp = np.pad(bg, ((0, 0), (2, 2), (2, 2)), mode="edge")
        c = bg.copy()
        emis_static = np.zeros((nb, ny, nx))
        emis_static[0] = d["fixed_factor"][e] * d["fixed"]
        road = d["road"]                                     # (6, ny, nx)
        dcos, dsin = np.cos(np.deg2rad(delta)), np.sin(np.deg2rad(delta))
        gc, gs = np.cos(self.gam), np.sin(self.gam)

        t = float(d["t0"][e]); t_end = float(d["t1"][e]); t_sat = float(d["t_sat"][e])
        n = int(round((t_end - t) / DT))
        A_sat = None
        if sta is not None:
            acc = np.zeros((len(sta["ta"]), nb)); cnt = np.zeros(len(sta["ta"]))
            si = np.minimum((sta["x"] // DX).astype(int), nx - 1)
            sj = np.minimum((sta["y"] // DX).astype(int), ny - 1)
        sat_done = False
        for it in range(n):
            tm = t + 0.5 * DT
            u, v, h, f = self._met(e, tm)
            ue = a_sc * (dcos * u - dsin * v)
            vn = a_sc * (dsin * u + dcos * v)
            ug = gc * ue + gs * vn
            vg = -gs * ue + gc * vn
            ufx = np.empty((ny, nx + 1))
            ufx[:, 1:-1] = 0.5 * (ug[:, :-1] + ug[:, 1:]); ufx[:, 0] = ug[:, 0]; ufx[:, -1] = ug[:, -1]
            vfy = np.empty((ny + 1, nx))
            vfy[1:-1] = 0.5 * (vg[:-1] + vg[1:]); vfy[0] = vg[0]; vfy[-1] = vg[-1]
            em = emis_static.copy()
            df = float(diurnal(tm, d["diurnal"])) * d["road_factor"][e]
            em[1:1 + N_REGIONS] = df * road
            cp = c
            c = step(c, bgp, ufx[None], vfy[None], em, DT, tau, d["K"], nx, ny)
            if (not sat_done) and (t + DT >= t_sat) and sat_W is not None:
                w = (t_sat - t) / DT
                cs = (1 - w) * cp + w * c
                _, _, _, fs = self._met(e, t_sat)
                fc = (cs * fs[None]).reshape(nb, -1)
                A_sat = (sat_W @ fc.T)
            if (not sat_done) and (t + DT >= t_sat):
                sat_done = True
            if sta is not None:
                sel = (sta["ta"] <= t + DT) & (t + DT <= sta["tb"])
                if sel.any():
                    z = sta["h"][sel] / h[sj[sel], si[sel]]
                    w_ = (f[sj[sel], si[sel]] * phi(z, d["zeta0"])
                          / h[sj[sel], si[sel]]) * UG_PER_MOL
                    acc[sel] += c[:, sj[sel], si[sel]].T * w_[:, None]
                    cnt[sel] += 1.0
            t += DT
        A_sta = acc / np.maximum(cnt, 1.0)[:, None] if sta is not None else None
        return A_sat, A_sta


def build_training_blocks(model, sat, sta_rows, d):
    """Footprint operators and station specs per training episode."""
    blocks = {}
    for s in sat:
        e = s["episode"]
        blocks.setdefault(e, {})["sat"] = s
        blocks[e]["W"] = footprint_matrix(s["cx"], s["cy"], model.nx, model.ny)
    by_ep = {}
    for r in sta_rows:
        by_ep.setdefault(int(r["episode_id"]), []).append(r)
    for e, rows in by_ep.items():
        blocks.setdefault(e, {})["sta"] = dict(
            ta=np.array([float(r["interval_start_utc"]) for r in rows]),
            tb=np.array([float(r["interval_end_utc"]) for r in rows]),
            x=np.array([float(r["x_m"]) for r in rows]),
            y=np.array([float(r["y_m"]) for r in rows]),
            h=np.array([float(r["inlet_height_m"]) for r in rows]),
            value=np.array([float(r["no2_ug_m3"]) for r in rows]),
            sigma=np.array([float(r["sigma_ug_m3"]) for r in rows]))
    return blocks


S_LO, S_HI = 0.30, 2.50
TAU_LO, TAU_HI = 3600.0, 28800.0
A_LO, A_HI = 0.70, 1.30
D_LO, D_HI = -20.0, 20.0


def inner_fit(A, obs, sig):
    M = A[:, 1:]
    off = A[:, 0]
    W = 1.0 / sig
    lo = np.array([S_LO] * N_REGIONS + [0.0, 0.0, 0.0])
    hi = np.array([S_HI] * N_REGIONS + [np.inf] * 3)
    r = lsq_linear(M * W[:, None], (obs - off) * W, bounds=(lo, hi),
                   method="trf", tol=1e-12, max_iter=300)
    res = M @ r.x + off - obs
    return r.x, float(np.sum((res / sig) ** 2))


_MODEL = None
_BLOCKS = None
_POOL = None


def _worker(arg):
    e, tau, a_sc, delta = arg
    b = _BLOCKS[e]
    return _MODEL.run(e, tau, a_sc, delta, sat_W=b["W"], sta=b["sta"])


def assemble(model, blocks, eps, tau, a_sc, delta):
    args = [(e, tau, a_sc, delta) for e in eps]
    if _POOL is not None:
        parts = list(_POOL.map(_worker, args, chunksize=1))
    else:
        parts = [_worker(a) for a in args]
    A, obs, sig = [], [], []
    for e, (As, At) in zip(eps, parts):
        b = blocks[e]
        A.append(As * b["sat"]["sens"][:, None])
        obs.append(b["sat"]["value"]); sig.append(b["sat"]["sig_tot"])
        A.append(At); obs.append(b["sta"]["value"]); sig.append(b["sta"]["sig_tot"])
    return np.vstack(A), np.concatenate(obs), np.concatenate(sig)


def main():
    log("loading inputs")
    d = load_inputs()
    with open(f"{DATA}/data_manifest.json") as fh:
        man = json.load(fh)
    sr_sat = float(man["representation_error"]["satellite_mol_m2"])
    sr_sta = float(man["representation_error"]["station_ug_m3"])

    sat, sta_rows = load_training_obs(d)
    model = Model(d)
    blocks = build_training_blocks(model, sat, sta_rows, d)
    for e, b in blocks.items():
        b["sat"]["sig_tot"] = np.hypot(b["sat"]["sigma"], sr_sat)
        b["sta"]["sig_tot"] = np.hypot(b["sta"]["sigma"], sr_sta)
    eps = sorted(blocks)
    n_obs = sum(len(blocks[e]["sat"]["value"]) + len(blocks[e]["sta"]["value"]) for e in eps)
    log(f"training episodes {eps}, {n_obs} accepted observations")

    global _MODEL, _BLOCKS, _POOL
    _MODEL, _BLOCKS = model, blocks
    ncpu = max(1, min(4, os.cpu_count() or 1))
    if ncpu > 1:
        import multiprocessing as mp
        _POOL = mp.get_context("fork").Pool(ncpu)
        log(f"using {ncpu} worker processes")

    # one episode per wind regime carries the coarse search cheaply
    seen, scout = set(), []
    for e in eps:
        r = d["regime"][e]
        if r not in seen:
            seen.add(r); scout.append(e)
    scout = sorted(scout)
    log(f"scout episodes for the coarse stage: {scout}")

    def objective_on(z, which):
        tau = float(np.exp(z[0])); a_sc = float(z[1]); delta = float(z[2])
        if not (TAU_LO <= tau <= TAU_HI and A_LO <= a_sc <= A_HI and D_LO <= delta <= D_HI):
            return 1e18
        A, obs, sig = assemble(model, blocks, which, tau, a_sc, delta)
        p, chi2 = inner_fit(A, obs, sig)
        objective_on.last = (p, chi2, tau, a_sc, delta)
        return chi2

    def objective(z):
        j = objective_on(z, eps)
        objective.last = objective_on.last
        return j

    log("coarse search over lifetime, wind speed scale and rotation")
    best = None
    for tau_h in (1.4, 2.4, 3.6, 5.2, 7.2):
        for a_sc in (0.80, 1.00, 1.20):
            for delta in (-15.0, -5.0, 5.0, 15.0):
                z = np.array([np.log(tau_h * 3600.0), a_sc, delta])
                j = objective_on(z, scout)
                if best is None or j < best[0]:
                    best = (j, z.copy())
                    log(f"  tau={tau_h:.2f}h a={a_sc:.2f} d={delta:+.1f} chi2={j:.1f} *")
    log(f"coarse best chi2 {best[0]:.1f} on the scout subset")

    log("intermediate refinement on the scout subset")
    r0 = minimize(lambda z: objective_on(z, scout), best[1], method="Nelder-Mead",
                  options=dict(xatol=2e-3, fatol=1.0, maxfev=150, adaptive=True))
    log(f"scout optimum chi2 {r0.fun:.1f} at tau={np.exp(r0.x[0])/3600:.3f}h "
        f"a={r0.x[1]:.4f} delta={r0.x[2]:+.3f}")

    log("final refinement on all training episodes")
    r = minimize(objective, r0.x, method="Nelder-Mead",
                 options=dict(xatol=1e-4, fatol=1e-2, maxfev=180, adaptive=True))
    objective(r.x)
    p, chi2, tau, a_sc, delta = objective.last
    s = p[:N_REGIONS]
    c00, c10, c01 = p[N_REGIONS:]
    b0, bx, by = c00, c10 - c00, c01 - c00
    log(f"fit: chi2={chi2:.1f} n={n_obs} tau={tau/3600:.3f}h a={a_sc:.4f} "
        f"delta={delta:+.3f} deg")
    log(f"     scales={np.round(s,4).tolist()}")
    log(f"     background=({b0:.4e},{bx:.4e},{by:.4e})")

    theta = np.concatenate([[1.0], s, [c00, c10, c01]])

    log("predicting held-out queries")
    qsat, qsta = load_queries()
    rows = []
    for e in sorted(set(qsat["episode"].tolist()) | set(qsta["episode"].tolist())):
        ms = qsat["episode"] == e
        mt = qsta["episode"] == e
        W = footprint_matrix(qsat["cx"][ms], qsat["cy"][ms], model.nx, model.ny)
        spec = dict(ta=qsta["ta"][mt], tb=qsta["tb"][mt], x=qsta["x"][mt],
                    y=qsta["y"][mt], h=qsta["h"][mt])
        As, At = model.run(e, tau, a_sc, delta, sat_W=W, sta=spec)
        ys = (As @ theta) * qsat["sens"][ms]
        yt = At @ theta
        for oid, v in zip(qsat["obs_id"][ms], ys):
            rows.append((oid, "satellite", float(v), "mol m-2"))
        for oid, v in zip(qsta["obs_id"][mt], yt):
            rows.append((oid, "station", float(v), "ug m-3"))
        log(f"  episode {e}: {ms.sum()} satellite + {mt.sum()} station queries")

    log("writing outputs")
    area = float(d["area"][0, 0])
    base = d["road"].sum(axis=(1, 2)) * area                 # mol s-1 per region
    totals = {}
    for e in range(len(d["t0"])):
        f = float(d["road_factor"][e]) * float(diurnal(d["t_sat"][e], d["diurnal"]))
        totals[str(int(e))] = float((s * base).sum() * f)

    with open(f"{OUT}/result.json", "w") as fh:
        json.dump(dict(
            schema_version="1.0",
            region_ids=[int(v) for v in d["region_id"]],
            emission_scale=[float(v) for v in s],
            effective_lifetime_s=float(tau),
            wind_speed_scale=float(a_sc),
            wind_rotation_deg=float(delta),
            background_coefficients=[float(b0), float(bx), float(by)],
            background_units="mol m-2",
            integrated_road_mol_s=totals,
            integrated_road_units="mol s-1",
        ), fh, indent=2)

    with Dataset(f"{OUT}/posterior.nc", "w", format="NETCDF4") as ds:
        ds.schema_version = "1.0"
        ds.createDimension("x", model.nx); ds.createDimension("y", model.ny)
        ds.createDimension("region", N_REGIONS); ds.createDimension("nv", 2)
        v = ds.createVariable("x", "f8", ("x",)); v[:] = d["x"]; v.units = "m"
        v = ds.createVariable("y", "f8", ("y",)); v[:] = d["y"]; v.units = "m"
        v = ds.createVariable("x_bnds", "f8", ("x", "nv"))
        v[:] = np.stack([d["x"] - DX / 2, d["x"] + DX / 2], 1)
        v = ds.createVariable("y_bnds", "f8", ("y", "nv"))
        v[:] = np.stack([d["y"] - DX / 2, d["y"] + DX / 2], 1)
        v = ds.createVariable("region_id", "i4", ("region",)); v[:] = d["region_id"]
        v = ds.createVariable("corrected_road_flux", "f8", ("region", "y", "x"))
        v[:] = s[:, None, None] * d["road"]; v.units = "mol m-2 s-1"
        v = ds.createVariable("total_source_flux", "f8", ("y", "x"))
        v[:] = (s[:, None, None] * d["road"]).sum(0) + d["fixed"]
        v.units = "mol m-2 s-1"
        c = ds.createVariable("crs", "i4"); c.epsg_code = "EPSG:3035"

    with open(f"{OUT}/predicted_observations.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["obs_id", "instrument_type", "predicted_value", "unit"])
        for r_ in rows:
            w.writerow([r_[0], r_[1], f"{r_[2]:.10e}", r_[3]])
    log(f"wrote {len(rows)} predictions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
