"""Independent forward recomputation used by the sealed verifier.

Deliberately a different discretisation from the reference solution: Strang
directional splitting with a van Leer limited flux, rather than an unsplit
update.  It exists to check that a submitted parameter vector actually produces
the submitted predictions, so it must not share code with solution/.
"""
import csv, json, os
import numpy as np
from netCDF4 import Dataset

DX = 4000.0
DT = 60.0
NREG = 6
UG_PER_MOL_NO2 = 46.0055e-3 * 1.0e9


def read_inputs(data_dir):
    I = {}
    with Dataset(os.path.join(data_dir, "domain.nc")) as ds:
        I["x"] = np.asarray(ds["x"][:]); I["y"] = np.asarray(ds["y"][:])
        I["gamma"] = np.asarray(ds["grid_convergence"][:])
        I["area"] = np.asarray(ds["cell_area"][:])
        I["region_id"] = np.asarray(ds["region_id"][:])
    with Dataset(os.path.join(data_dir, "emissions.nc")) as ds:
        I["road"] = (np.asarray(ds["road_prior"][:])
                     / (ds.molar_mass_no2_kg_per_mol * 1.0e6 * 3600.0))
        I["fixed"] = (np.asarray(ds["fixed_sources"][:])
                      / (ds.molar_mass_n_kg_per_mol * 1.0e6 * 3600.0))
        I["road_factor"] = np.asarray(ds["road_time_factor"][:])
        I["fixed_factor"] = np.asarray(ds["fixed_time_factor"][:])
        I["diurnal"] = np.asarray(ds["road_diurnal_factor"][:])
    with Dataset(os.path.join(data_dir, "meteorology.nc")) as ds:
        I["K"] = float(ds.horizontal_diffusivity_m2_s)
        I["xm"] = np.asarray(ds["xm"][:]); I["ym"] = np.asarray(ds["ym"][:])
        I["mtime"] = np.asarray(ds["time"][:])
        zb = np.asarray(ds["level_bounds"][:])
        u = np.asarray(ds["u_east"][:], dtype=float)
        v = np.asarray(ds["v_north"][:], dtype=float)
        h = np.asarray(ds["blh"][:], dtype=float)
        I["f_no2"] = np.asarray(ds["f_no2"][:], dtype=float)
        I["photo"] = np.asarray(ds["photolysis_factor"][:], dtype=float)
        I["blh"] = h
        num_u = np.zeros_like(h); num_v = np.zeros_like(h); den = np.zeros_like(h)
        for lv in range(zb.shape[0]):
            th = np.clip(np.minimum(h, zb[lv, 1]) - zb[lv, 0], 0.0, None)
            if lv == 0:
                th = np.maximum(th, 1.0)
            den += th; num_u += th * u[:, :, lv]; num_v += th * v[:, :, lv]
        I["ue"] = num_u / den; I["vn"] = num_v / den
        I["t0"] = np.asarray(ds["episode_start"][:])
        I["t1"] = np.asarray(ds["episode_end"][:])
        I["tsat"] = np.asarray(ds["overpass_time"][:])
        I["regime"] = [str(s) for s in ds["wind_regime"][:]]
    return I


def read_queries(data_dir):
    with Dataset(os.path.join(data_dir, "prediction_queries.nc")) as ds:
        sat = dict(obs_id=np.array([str(s) for s in ds["sat_obs_id"][:]]),
                   episode=np.asarray(ds["sat_episode_id"][:]),
                   cx=np.asarray(ds["sat_corner_x"][:]),
                   cy=np.asarray(ds["sat_corner_y"][:]),
                   sens=np.asarray(ds["sat_vertical_sensitivity"][:]),
                   sigma=np.asarray(ds["sat_uncertainty"][:]))
        sta = dict(obs_id=np.array([str(s) for s in ds["sta_obs_id"][:]]),
                   episode=np.asarray(ds["sta_episode_id"][:]),
                   x=np.asarray(ds["sta_x"][:]), y=np.asarray(ds["sta_y"][:]),
                   h=np.asarray(ds["sta_inlet_height_m"][:]),
                   ta=np.asarray(ds["sta_interval_start_utc"][:]),
                   tb=np.asarray(ds["sta_interval_end_utc"][:]),
                   sigma=np.asarray(ds["sta_uncertainty"][:]))
    return sat, sta


# ---------------------------------------------------------------- geometry ---
def clip_rect(poly, x0, x1, y0, y1):
    def pass_(pts, sel, cut):
        res = []
        for i in range(len(pts)):
            a = pts[i]; b = pts[(i + 1) % len(pts)]
            sa, sb = sel(a), sel(b)
            if sa:
                res.append(a)
                if not sb:
                    res.append(cut(a, b))
            elif sb:
                res.append(cut(a, b))
        return res
    P = list(map(tuple, poly))
    fx = lambda a, b, v: (v, a[1] + (v - a[0]) * (b[1] - a[1]) / (b[0] - a[0]))
    fy = lambda a, b, v: (a[0] + (v - a[1]) * (b[0] - a[0]) / (b[1] - a[1]), v)
    for sel, cut in ((lambda p: p[0] >= x0, lambda a, b: fx(a, b, x0)),
                     (lambda p: p[0] <= x1, lambda a, b: fx(a, b, x1)),
                     (lambda p: p[1] >= y0, lambda a, b: fy(a, b, y0)),
                     (lambda p: p[1] <= y1, lambda a, b: fy(a, b, y1))):
        P = pass_(P, sel, cut)
        if not P:
            return P
    return P


def area_of(P):
    if len(P) < 3:
        return 0.0
    s = 0.0
    for i in range(len(P)):
        x1, y1 = P[i]; x2, y2 = P[(i + 1) % len(P)]
        s += x1 * y2 - x2 * y1
    return 0.5 * abs(s)


def footprint_rows(cx, cy, nx, ny):
    from scipy.sparse import coo_matrix
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
                a = area_of(clip_rect(poly, i * DX, (i + 1) * DX, j * DX, (j + 1) * DX))
                if a > 0:
                    cc.append(j * nx + i); vv.append(a)
        tot = float(sum(vv))
        for cj, vj in zip(cc, vv):
            r.append(k); c.append(cj); v.append(vj / tot)
    return coo_matrix((v, (r, c)), shape=(cx.shape[0], nx * ny)).tocsr()


# ------------------------------------------------------------------ solver ---
def _limited(cm, c0, cp):
    a = c0 - cm
    b = cp - c0
    p = a * b
    s = np.zeros_like(a)
    m = p > 0
    s[m] = 2.0 * p[m] / (a[m] + b[m])
    return s


def sweep_x(c, bg, uf, dt):
    ny, nx = c.shape
    e = np.concatenate([bg[:, :1], bg[:, :1], c, bg[:, -1:], bg[:, -1:]], axis=1)
    s = _limited(e[:, :-2], e[:, 1:-1], e[:, 2:])           # slopes for cols 1..nx+2
    cl = e[:, 1:-2] + 0.5 * s[:, :-1]
    cr = e[:, 2:-1] - 0.5 * s[:, 1:]
    f = uf * np.where(uf >= 0, cl, cr)
    f[:, 0] = uf[:, 0] * np.where(uf[:, 0] >= 0, e[:, 1], e[:, 2])
    f[:, -1] = uf[:, -1] * np.where(uf[:, -1] >= 0, e[:, -3], e[:, -2])
    return c - dt * (f[:, 1:] - f[:, :-1]) / DX


def sweep_y(c, bg, vf, dt):
    return sweep_x(c.T, bg.T, vf.T, dt).T


def diffuse(c, K, dt):
    gx = np.zeros((c.shape[0], c.shape[1] + 1))
    gx[:, 1:-1] = (c[:, 1:] - c[:, :-1]) / DX
    gy = np.zeros((c.shape[0] + 1, c.shape[1]))
    gy[1:-1] = (c[1:] - c[:-1]) / DX
    return c + dt * K * ((gx[:, 1:] - gx[:, :-1]) / DX + (gy[1:] - gy[:-1]) / DX)


def phi(zeta, z0):
    z = np.asarray(zeta, float)
    return np.where((z >= 0) & (z <= 1),
                    np.exp(-z / z0) / (z0 * (1.0 - np.exp(-1.0 / z0))), 0.0)


class Bilin:
    def __init__(self, sx, sy, tx_, ty_):
        self.i = np.clip(np.searchsorted(sx, tx_) - 1, 0, len(sx) - 2)
        self.j = np.clip(np.searchsorted(sy, ty_) - 1, 0, len(sy) - 2)
        self.a = np.clip((tx_ - sx[self.i]) / np.diff(sx)[self.i], 0, 1)
        self.b = np.clip((ty_ - sy[self.j]) / np.diff(sy)[self.j], 0, 1)

    def __call__(self, F):
        i, j, a, b = self.i, self.j, self.a[None, :], self.b[:, None]
        return ((1 - a) * (1 - b) * F[np.ix_(j, i)]
                + a * (1 - b) * F[np.ix_(j, i + 1)]
                + (1 - a) * b * F[np.ix_(j + 1, i)]
                + a * b * F[np.ix_(j + 1, i + 1)])


def diurnal_at(t, tab):
    h = (np.asarray(t, float) % 86400.0) / 3600.0
    i = np.floor(h).astype(int) % 24
    w = h - np.floor(h)
    return tab[i] * (1 - w) + tab[(i + 1) % 24] * w


def forward_episode(I, e, params, W_sat, sta, nx, ny):
    """Return (satellite predictions, station predictions) for one episode."""
    s = np.asarray(params["emission_scale"], float)
    tau0 = float(params["reference_loss_time_s"])
    c_ref = float(params["loss_saturation_column"])
    a_sc = float(params["wind_speed_scale"])
    delta = float(params["wind_rotation_deg"])
    b0, bx, by = (float(v) for v in params["background_coefficients"])
    fixed_scale = float(params["fixed_source_scale"])
    zeta0 = float(params["vertical_shape_zeta0"])

    ip = Bilin(I["xm"], I["ym"], I["x"], I["y"])
    XT, YT = np.meshgrid(I["x"] / (nx * DX), I["y"] / (ny * DX))
    bg = b0 + bx * XT + by * YT
    c = bg.copy()
    emis_road = np.tensordot(s, I["road"], axes=(0, 0))
    emis_fix = fixed_scale * float(I["fixed_factor"][e]) * I["fixed"]
    gam = np.deg2rad(I["gamma"])
    cd, sd = np.cos(np.deg2rad(delta)), np.sin(np.deg2rad(delta))
    cg, sg = np.cos(gam), np.sin(gam)
    tt = I["mtime"][e]

    t = float(I["t0"][e]); tend = float(I["t1"][e]); tsat = float(I["tsat"][e])
    n = int(round((tend - t) / DT))
    y_sat = None
    acc = np.zeros(len(sta["ta"])); cnt = np.zeros(len(sta["ta"]))
    si = np.minimum((sta["x"] // DX).astype(int), nx - 1)
    sj = np.minimum((sta["y"] // DX).astype(int), ny - 1)
    done = False
    for _ in range(n):
        tm = t + 0.5 * DT
        k = int(np.clip(np.searchsorted(tt, tm) - 1, 0, len(tt) - 2))
        w = (tm - tt[k]) / (tt[k + 1] - tt[k])
        L = lambda A: (1 - w) * A[k] + w * A[k + 1]
        u = ip(L(I["ue"][e])); v = ip(L(I["vn"][e]))
        h = ip(L(I["blh"][e])); f = ip(L(I["f_no2"][e]))
        ph = ip(L(I["photo"][e]))
        ue = a_sc * (cd * u - sd * v)
        vn = a_sc * (sd * u + cd * v)
        ug = cg * ue + sg * vn
        vg = -sg * ue + cg * vn
        uf = np.empty((ny, nx + 1))
        uf[:, 1:-1] = 0.5 * (ug[:, :-1] + ug[:, 1:]); uf[:, 0] = ug[:, 0]; uf[:, -1] = ug[:, -1]
        vf = np.empty((ny + 1, nx))
        vf[1:-1] = 0.5 * (vg[:-1] + vg[1:]); vf[0] = vg[0]; vf[-1] = vg[-1]
        em = emis_fix + float(diurnal_at(tm, I["diurnal"])) * float(I["road_factor"][e]) * emis_road

        cprev = c
        # Strang splitting: half x, full y, half x, then sources and loss
        c = sweep_x(c, bg, uf, 0.5 * DT)
        c = sweep_y(c, bg, vf, DT)
        c = sweep_x(c, bg, uf, 0.5 * DT)
        c = diffuse(c, I["K"], DT)
        c = c + DT * em
        kloss = ph / (tau0 * (1.0 + np.maximum(c, 0.0) / c_ref))
        c = c * np.exp(-DT * kloss)

        if (not done) and (t + DT >= tsat):
            frac = (tsat - t) / DT
            cs = (1 - frac) * cprev + frac * c
            k2 = int(np.clip(np.searchsorted(tt, tsat) - 1, 0, len(tt) - 2))
            w2 = (tsat - tt[k2]) / (tt[k2 + 1] - tt[k2])
            fs = ip((1 - w2) * I["f_no2"][e][k2] + w2 * I["f_no2"][e][k2 + 1])
            if W_sat is not None and W_sat.shape[0]:
                y_sat = W_sat @ (cs * fs).reshape(-1)
            done = True
        sel = (sta["ta"] <= t + DT) & (t + DT <= sta["tb"])
        if sel.any():
            z = sta["h"][sel] / h[sj[sel], si[sel]]
            acc[sel] += (f[sj[sel], si[sel]] * c[sj[sel], si[sel]]
                         * phi(z, zeta0) / h[sj[sel], si[sel]]) * UG_PER_MOL_NO2
            cnt[sel] += 1.0
        t += DT
    return y_sat, acc / np.maximum(cnt, 1.0)


def recompute_all(I, qsat, qsta, params, nx, ny):
    """Predictions for every held-out query, in query order."""
    ys = np.full(len(qsat["obs_id"]), np.nan)
    yt = np.full(len(qsta["obs_id"]), np.nan)
    for e in sorted(set(qsat["episode"].tolist()) | set(qsta["episode"].tolist())):
        ms = qsat["episode"] == e
        mt = qsta["episode"] == e
        W = footprint_rows(qsat["cx"][ms], qsat["cy"][ms], nx, ny)
        spec = dict(ta=qsta["ta"][mt], tb=qsta["tb"][mt], x=qsta["x"][mt],
                    y=qsta["y"][mt], h=qsta["h"][mt])
        a, b = forward_episode(I, int(e), params, W, spec, nx, ny)
        ys[ms] = a * qsat["sens"][ms]
        yt[mt] = b
    return ys, yt
