"""Construction of the static fields: emissions, meteorology, observation geometry."""
import numpy as np
from datetime import datetime, timezone
import config as C


def grid_centres(nx, ny, dx):
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    return x, y


def gaussian_blob(X, Y, x0, y0, sx, sy, rot_deg, amp):
    t = np.deg2rad(rot_deg)
    dx, dy = X - x0, Y - y0
    a = dx * np.cos(t) + dy * np.sin(t)
    b = -dx * np.sin(t) + dy * np.cos(t)
    return amp * np.exp(-0.5 * ((a / sx) ** 2 + (b / sy) ** 2))


def line_source(X, Y, pts, width, amp):
    """Smooth emission ribbon along a polyline (a road corridor)."""
    out = np.zeros_like(X)
    pts = np.asarray(pts, float)
    for k in range(len(pts) - 1):
        p, q = pts[k], pts[k + 1]
        d = q - p
        L2 = float(d @ d)
        t = ((X - p[0]) * d[0] + (Y - p[1]) * d[1]) / L2
        t = np.clip(t, 0.0, 1.0)
        px = p[0] + t * d[0]
        py = p[1] + t * d[1]
        r2 = (X - px) ** 2 + (Y - py) ** 2
        out = np.maximum(out, amp * np.exp(-0.5 * r2 / width ** 2))
    return out


def build_road_fields_fine():
    """Six road source groups on the 2 km truth grid, in mol NOx m-2 s-1."""
    x, y = grid_centres(C.NX_F, C.NY_F, C.DX_FINE)
    X, Y = np.meshgrid(x, y)
    k = 1000.0
    R = np.zeros((C.N_REGIONS, C.NY_F, C.NX_F))

    # R1: dense motorway ring around the central conurbation
    ring = []
    for ang in np.linspace(0, 2 * np.pi, 65):
        ring.append([118 * k + 17 * k * np.cos(ang), 96 * k + 13 * k * np.sin(ang)])
    R[0] = line_source(X, Y, ring, 2400.0, 1.55e-8)
    R[0] += gaussian_blob(X, Y, 118 * k, 96 * k, 9 * k, 7 * k, 0, 6.0e-9)

    # R2: north industrial corridor, two parallel trunk roads
    R[1] = line_source(X, Y, [[62 * k, 138 * k], [104 * k, 149 * k], [163 * k, 143 * k]],
                       2600.0, 7.8e-9)
    R[1] += line_source(X, Y, [[70 * k, 128 * k], [150 * k, 133 * k]], 2200.0, 4.6e-9)

    # R3: east urban arterials
    R[2] = line_source(X, Y, [[172 * k, 70 * k], [196 * k, 88 * k], [214 * k, 112 * k]],
                       2300.0, 9.2e-9)
    R[2] += gaussian_blob(X, Y, 196 * k, 90 * k, 7 * k, 8 * k, 20, 5.1e-9)

    # R4: south rural trunk road
    R[3] = line_source(X, Y, [[30 * k, 26 * k], [95 * k, 18 * k], [168 * k, 31 * k]],
                       2500.0, 5.3e-9)

    # R5: west transit motorway, crosses the inflow edge for the SW regime
    R[4] = line_source(X, Y, [[6 * k, 54 * k], [44 * k, 74 * k], [86 * k, 86 * k]],
                       2700.0, 1.02e-8)

    # R6: south-east secondary network
    R[5] = line_source(X, Y, [[150 * k, 46 * k], [186 * k, 38 * k], [222 * k, 52 * k]],
                       2400.0, 6.4e-9)
    R[5] += line_source(X, Y, [[160 * k, 60 * k], [200 * k, 66 * k]], 2000.0, 3.7e-9)
    return R * C.EMISSION_SCALE


def build_fixed_sources_fine():
    """Non-road point/area sources on the truth grid, mol NOx m-2 s-1."""
    x, y = grid_centres(C.NX_F, C.NY_F, C.DX_FINE)
    X, Y = np.meshgrid(x, y)
    k = 1000.0
    F = np.zeros((C.NY_F, C.NX_F))
    F += gaussian_blob(X, Y, 52 * k, 118 * k, 3.0 * k, 3.0 * k, 0, 1.40e-8)   # power plant
    F += gaussian_blob(X, Y, 205 * k, 148 * k, 2.6 * k, 2.6 * k, 0, 9.5e-9)   # cement works
    F += gaussian_blob(X, Y, 88 * k, 40 * k, 3.4 * k, 3.4 * k, 0, 6.8e-9)     # refinery
    F += gaussian_blob(X, Y, 132 * k, 96 * k, 6.0 * k, 5.0 * k, 0, 4.2e-9)    # domestic core
    return F * C.EMISSION_SCALE


def coarsen(field_fine, factor=2):
    """Area-weighted mean from the 2 km truth grid to the 4 km inference grid."""
    a = np.asarray(field_fine)
    lead = a.shape[:-2]
    ny, nx = a.shape[-2:]
    a = a.reshape(lead + (ny // factor, factor, nx // factor, factor))
    return a.mean(axis=(-3, -1))


def episode_times():
    starts = np.array([datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
                       .replace(tzinfo=timezone.utc).timestamp()
                       for s in C.EPISODE_START_ISO])
    return starts


def build_meteorology(rng):
    """Supplied meteorology on the 12 km grid: layer winds, BLH, NO2 fraction.

    Returns dict of arrays with dims (episode, time, level, y, x) for winds and
    (episode, time, y, x) for blh / f_no2.  Times are hourly, covering each
    episode window plus one hour of margin at each end.
    """
    xm, ym = grid_centres(C.NX_M, C.NY_M, C.DX_MET)
    Xm, Ym = np.meshgrid(xm, ym)
    xt = Xm / C.LX
    yt = Ym / C.LY

    nt = int(C.EPISODE_HOURS) + 3          # hourly, one hour margin each side
    starts = episode_times()
    met_times = np.zeros((C.N_EPISODES, nt))
    z_iface = np.array([0.0, 150.0, 450.0, 1200.0, 2500.0])
    nlev = len(z_iface) - 1

    uE = np.zeros((C.N_EPISODES, nt, nlev, C.NY_M, C.NX_M))
    vN = np.zeros_like(uE)
    blh = np.zeros((C.N_EPISODES, nt, C.NY_M, C.NX_M))
    fno2 = np.zeros_like(blh)

    for e in range(C.N_EPISODES):
        reg = C.EPISODE_REGIME[e]
        th0, sp0, spread = C.REGIME_WIND[reg]
        met_times[e] = starts[e] + 3600.0 * (np.arange(nt) - 1.0)
        # synoptic evolution within the episode
        dth = rng.normal(0.0, 9.0)
        dsp = rng.normal(0.0, spread)
        turn = rng.normal(0.0, 6.0)        # degrees turned over the window
        accel = rng.normal(0.0, 0.18)
        for it in range(nt):
            frac = it / (nt - 1.0)
            theta = np.deg2rad(th0 + dth + turn * (frac - 0.5) * 2.0)
            speed = max(0.8, sp0 + dsp + accel * sp0 * (frac - 0.5) * 2.0)
            # smooth horizontal shear and a weak diurnal low-level jet
            shear = 1.0 + 0.22 * (xt - 0.5) - 0.15 * (yt - 0.5)
            curl = np.deg2rad(7.0) * (yt - 0.5) - np.deg2rad(5.0) * (xt - 0.5)
            hod = np.sin(2 * np.pi * (frac - 0.15))
            for lv in range(nlev):
                zc = 0.5 * (z_iface[lv] + z_iface[lv + 1])
                prof = (zc / 300.0) ** 0.16
                lvturn = np.deg2rad(6.5) * np.log1p(zc / 120.0) / np.log1p(2500 / 120.0)
                ang = theta + curl + lvturn
                mag = speed * shear * prof * (1.0 + 0.07 * hod)
                uE[e, it, lv] = mag * np.cos(ang)
                vN[e, it, lv] = mag * np.sin(ang)
        # boundary layer height: diurnal growth, terrain-modulated
        for it in range(nt):
            hour = (met_times[e, it] % 86400.0) / 3600.0
            base = {0: 1180.0, 1: 780.0, 2: 620.0}[reg]
            night = {0: 210.0, 1: 150.0, 2: 130.0}[reg]
            grow = np.clip(np.sin(np.pi * (hour - 5.0) / 12.0), 0.0, 1.0) ** 1.3
            terr = 1.0 + 0.17 * np.sin(3.1 * xt + 0.7) * np.cos(2.4 * yt - 0.3)
            blh[e, it] = (night + (base - night) * grow) * terr
            # NO2 / NOx partitioning: lower at midday, higher at night
            fno2[e, it] = np.clip(0.815 - 0.20 * grow + 0.03 * np.cos(4.0 * yt), 0.55, 0.90)
    return dict(times=met_times, z_iface=z_iface, u_east=uE, v_north=vN,
                blh=blh, f_no2=fno2, x=xm, y=ym)


def grid_convergence_field(nx, ny, dx):
    """Angle from grid north to true north, degrees, positive counter-clockwise."""
    x, y = grid_centres(nx, ny, dx)
    X, Y = np.meshgrid(x, y)
    xt, yt = X / C.LX, Y / C.LY
    return -2.85 + 5.30 * xt + 0.85 * yt - 0.60 * xt * yt


def build_stations(rng):
    k = 1000.0
    sites = [
        (" 18.0", 14.0, 118.0, "urban_traffic", 3.5),
        (" 02.0", 121.5, 101.0, "urban_background", 4.0),
        (" 03.0", 104.0, 88.0, "urban_traffic", 3.0),
        (" 04.0", 66.0, 132.0, "suburban_industrial", 5.5),
        (" 05.0", 191.0, 92.0, "urban_traffic", 3.5),
        (" 06.0", 209.0, 118.0, "suburban_background", 8.0),
        (" 07.0", 72.0, 24.0, "rural_background", 4.5),
        (" 08.0", 144.0, 28.0, "rural_traffic", 3.0),
        (" 09.0", 30.0, 66.0, "suburban_traffic", 4.0),
        (" 10.0", 96.0, 60.0, "rural_background", 6.0),
        (" 11.0", 176.0, 152.0, "suburban_background", 5.0),
        (" 12.0", 46.0, 158.0, "rural_background", 9.5),
    ]
    out = []
    for i, (_, sx, sy, kind, inlet) in enumerate(sites):
        out.append(dict(station_id=f"ST{i+1:02d}", x=sx * k, y=sy * k,
                        inlet_height_m=inlet, site_type=kind,
                        sigma_ug_m3=float(np.round(rng.uniform(1.8, 3.6), 3))))
    return out


def build_swath(rng, episode, t_overpass, inside_only):
    """Irregular quadrilateral footprints for one synthetic overpass.

    The swath is a sheared, slightly curved across/along-track raster, so pixel
    corners are not aligned with the inference grid.
    """
    k = 1000.0
    track_ang = np.deg2rad(rng.uniform(96.0, 108.0))     # near-polar descending
    across = np.arange(-19, 20)                          # 39 across-track rows
    along = np.arange(0, 24)                             # 24 along-track scans
    w_across = rng.uniform(6.4, 7.4) * k
    w_along = rng.uniform(6.8, 7.8) * k
    cx = C.LX * 0.5 + rng.uniform(-9 * k, 9 * k)
    cy = C.LY * 0.5 + rng.uniform(-7 * k, 7 * k)
    ta, tn = np.cos(track_ang), np.sin(track_ang)        # along-track unit vector
    aa, an = -tn, ta                                     # across-track unit vector
    swell = rng.uniform(0.055, 0.095)                    # across-track widening
    bow = rng.uniform(-0.10, 0.10)                       # gentle curvature

    corners, centres, scan, pix = [], [], [], []
    for j in along:
        for i in across:
            f = i / 19.0
            wa = w_across * (1.0 + swell * f * f)
            s_al = (j - 11.5) * w_along + bow * f * f * w_along * 6.0
            s_ac = np.sign(i) * (abs(i) - 0.5 + 0.5 * swell * abs(f) ** 3 * 19) * w_across \
                if i != 0 else 0.0
            # integrate the widening so that footprints tile without gaps
            s_ac = w_across * (np.sign(i) * (abs(i) * (1.0 + swell * f * f / 3.0)))
            xc = cx + s_al * ta + s_ac * aa
            yc = cy + s_al * tn + s_ac * an
            du = 0.5 * wa
            dv = 0.5 * w_along
            quad = [
                (xc - du * aa - dv * ta, yc - du * an - dv * tn),
                (xc + du * aa - dv * ta, yc + du * an - dv * tn),
                (xc + du * aa + dv * ta, yc + du * an + dv * tn),
                (xc - du * aa + dv * ta, yc - du * an + dv * tn),
            ]
            qx = np.array([p[0] for p in quad])
            qy = np.array([p[1] for p in quad])
            if inside_only:
                if qx.min() < 1500.0 or qx.max() > C.LX - 1500.0:
                    continue
                if qy.min() < 1500.0 or qy.max() > C.LY - 1500.0:
                    continue
            else:
                if qx.max() < 0.0 or qx.min() > C.LX or qy.max() < 0.0 or qy.min() > C.LY:
                    continue
            corners.append(np.stack([qx, qy], axis=1))
            centres.append((xc, yc))
            scan.append(int(j))
            pix.append(int(i + 19))
    corners = np.array(corners)
    centres = np.array(centres)
    # acquisition time varies along track by about 1.1 s per scan line
    t_obs = t_overpass + (np.array(scan) - 11.5) * 1.1
    return dict(corners=corners, centres=centres, scanline=np.array(scan),
                ground_pixel=np.array(pix), time=t_obs)
