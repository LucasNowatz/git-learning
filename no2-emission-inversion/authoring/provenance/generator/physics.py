"""Truth-side forward model: transport, chemistry and observation operators.

Written independently of solution/solve.sh and tests/.  It runs on the 2 km
truth grid; the published inference grid is 4 km.
"""
import numpy as np
import config as C


# ------------------------------------------------------------ interpolation --
def bilinear_weights(src_x, src_y, dst_x, dst_y):
    """Clamped bilinear interpolation operator from a regular grid to points."""
    ix = np.clip(np.searchsorted(src_x, dst_x) - 1, 0, len(src_x) - 2)
    iy = np.clip(np.searchsorted(src_y, dst_y) - 1, 0, len(src_y) - 2)
    tx = np.clip((dst_x - src_x[ix]) / (src_x[ix + 1] - src_x[ix]), 0.0, 1.0)
    ty = np.clip((dst_y - src_y[iy]) / (src_y[iy + 1] - src_y[iy]), 0.0, 1.0)
    return ix, iy, tx, ty


def apply_bilinear(field, ix, iy, tx, ty):
    """field has shape (..., ny_src, nx_src); returns (..., ny_dst, nx_dst)."""
    f00 = field[..., iy[:, None], ix[None, :]]
    f10 = field[..., iy[:, None], ix[None, :] + 1]
    f01 = field[..., iy[:, None] + 1, ix[None, :]]
    f11 = field[..., iy[:, None] + 1, ix[None, :] + 1]
    TX = tx[None, :]
    TY = ty[:, None]
    return ((1 - TX) * (1 - TY) * f00 + TX * (1 - TY) * f10
            + (1 - TX) * TY * f01 + TX * TY * f11)


def bl_mean_wind(met):
    """Layer-thickness weighted mean wind inside the boundary layer.

    Computed on the meteorological grid at the meteorological times, as the
    published convention requires.  Returns (episode, time, ny_m, nx_m, 2).
    """
    z = met["z_iface"]
    u, v, h = met["u_east"], met["v_north"], met["blh"]
    nlev = u.shape[2]
    wsum = np.zeros_like(h)
    usum = np.zeros_like(h)
    vsum = np.zeros_like(h)
    for lv in range(nlev):
        thick = np.clip(np.minimum(h, z[lv + 1]) - z[lv], 0.0, None)
        if lv == 0:
            thick = np.maximum(thick, 1.0)      # h below the first interface
        wsum += thick
        usum += thick * u[:, :, lv]
        vsum += thick * v[:, :, lv]
    return np.stack([usum / wsum, vsum / wsum], axis=-1)


# ------------------------------------------------------------------ solver ---
def van_leer(dm, dp):
    prod = dm * dp
    out = np.zeros_like(dm)
    m = prod > 0.0
    out[m] = 2.0 * prod[m] / (dm[m] + dp[m])
    return out


class Advector:
    """Flux-form finite-volume advection-diffusion-loss on a regular grid."""

    def __init__(self, nx, ny, dx, background):
        self.nx, self.ny, self.dx = nx, ny, dx
        self.bg = background                     # (ny, nx) inflow / initial field
        self._bgp = None

    def _pad(self, c):
        """Two ghost layers holding the background field on every boundary."""
        if self._bgp is None:
            b = np.atleast_3d(self.bg.T).T if self.bg.ndim == 2 else self.bg
            b = self.bg.reshape((-1,) + self.bg.shape[-2:])
            self._bgp = np.pad(b, ((0, 0), (2, 2), (2, 2)), mode="edge")
        p = np.repeat(self._bgp, c.shape[0] // self._bgp.shape[0], axis=0).copy() \
            if c.ndim == 3 and self._bgp.shape[0] == 1 and c.shape[0] != 1 \
            else self._bgp.copy()
        if c.ndim == 2:
            p = p[0]
        p[..., 2:-2, 2:-2] = c
        return p

    def step(self, c, ufx, vfy, emis, dt, tau0, c_ref, photo):
        """One time step with concentration-dependent chemical loss.

        The sink is L(C) = C * P / (tau0 * (1 + C / C_ref)); the coefficient is
        evaluated at the current column and applied as an exponential update.
        """
        p = self._pad(c)
        dx = self.dx

        # ---- x fluxes on faces 0..nx (indices into padded array) ----
        a = p[..., 2:-2, :]                       # (..., ny, nx+4)
        dm = a[..., 1:-1] - a[..., :-2]
        dp = a[..., 2:] - a[..., 1:-1]
        sl = van_leer(dm, dp)                     # slopes for padded cells 1..nx+2
        cl = a[..., 1:-2] + 0.5 * sl[..., :-1]    # left state at faces 0..nx
        cr = a[..., 2:-1] - 0.5 * sl[..., 1:]     # right state at faces 0..nx
        up = ufx >= 0.0
        cx = np.where(up, cl, cr)
        # boundary faces fall back to first order donor cell
        cx[..., 0] = np.where(ufx[..., 0] >= 0.0, a[..., 1], a[..., 2])
        cx[..., -1] = np.where(ufx[..., -1] >= 0.0, a[..., -3], a[..., -2])
        fx = ufx * cx

        # ---- y fluxes on faces 0..ny ----
        b = p[..., :, 2:-2]                       # (..., ny+4, nx)
        dm = b[..., 1:-1, :] - b[..., :-2, :]
        dp = b[..., 2:, :] - b[..., 1:-1, :]
        sl = van_leer(dm, dp)
        cl = b[..., 1:-2, :] + 0.5 * sl[..., :-1, :]
        cr = b[..., 2:-1, :] - 0.5 * sl[..., 1:, :]
        up = vfy >= 0.0
        cy = np.where(up, cl, cr)
        cy[..., 0, :] = np.where(vfy[..., 0, :] >= 0.0, b[..., 1, :], b[..., 2, :])
        cy[..., -1, :] = np.where(vfy[..., -1, :] >= 0.0, b[..., -3, :], b[..., -2, :])
        fy = vfy * cy

        div = (fx[..., 1:] - fx[..., :-1]) / dx + (fy[..., 1:, :] - fy[..., :-1, :]) / dx

        # ---- diffusion, zero flux across the domain boundary ----
        gx = np.zeros(c.shape[:-1] + (self.nx + 1,), dtype=c.dtype)
        gx[..., 1:-1] = (c[..., 1:] - c[..., :-1]) / dx
        gy = np.zeros(c.shape[:-2] + (self.ny + 1, self.nx), dtype=c.dtype)
        gy[..., 1:-1, :] = (c[..., 1:, :] - c[..., :-1, :]) / dx
        lap = C.K_DIFF * ((gx[..., 1:] - gx[..., :-1]) / dx
                          + (gy[..., 1:, :] - gy[..., :-1, :]) / dx)

        cn = c + dt * (-div + lap + emis)
        k = photo / (tau0 * (1.0 + np.maximum(cn, 0.0) / c_ref))
        return cn * np.exp(-dt * k)


def face_velocities(ug, vg):
    """Arithmetic face velocities from cell-centred grid-axis components."""
    ny, nx = ug.shape
    ufx = np.empty((ny, nx + 1))
    ufx[:, 1:-1] = 0.5 * (ug[:, :-1] + ug[:, 1:])
    ufx[:, 0] = ug[:, 0]
    ufx[:, -1] = ug[:, -1]
    vfy = np.empty((ny + 1, nx))
    vfy[1:-1, :] = 0.5 * (vg[:-1, :] + vg[1:, :])
    vfy[0, :] = vg[0, :]
    vfy[-1, :] = vg[-1, :]
    return ufx, vfy


def corrected_grid_wind(u_east, v_north, a_scale, delta_deg, gamma_deg):
    """a * R(delta) applied in the east/north frame, then rotation into grid axes."""
    d = np.deg2rad(delta_deg)
    ue = a_scale * (np.cos(d) * u_east - np.sin(d) * v_north)
    vn = a_scale * (np.sin(d) * u_east + np.cos(d) * v_north)
    g = np.deg2rad(gamma_deg)
    ug = np.cos(g) * ue + np.sin(g) * vn
    vg = -np.sin(g) * ue + np.cos(g) * vn
    return ug, vg


def background_field(coef, x, y):
    xt = x / C.LX
    yt = y / C.LY
    XT, YT = np.meshgrid(xt, yt)
    return coef[0] + coef[1] * XT + coef[2] * YT


# ------------------------------------------------- vertical shape operators --
def phi_shape(zeta, zeta0):
    """Normalised in-layer vertical shape; integrates to one over zeta in [0,1]."""
    z = np.asarray(zeta, dtype=float)
    val = np.exp(-z / zeta0) / (zeta0 * (1.0 - np.exp(-1.0 / zeta0)))
    return np.where((z >= 0.0) & (z <= 1.0), val, 0.0)


# ---------------------------------------------------- polygon / cell overlap --
def clip_polygon_to_box(poly, x0, x1, y0, y1):
    """Sutherland-Hodgman clip of a convex polygon against an axis-aligned box."""
    def clip(pts, inside, inter):
        out = []
        n = len(pts)
        for i in range(n):
            cur, nxt = pts[i], pts[(i + 1) % n]
            ci, ni = inside(cur), inside(nxt)
            if ci:
                out.append(cur)
                if not ni:
                    out.append(inter(cur, nxt))
            elif ni:
                out.append(inter(cur, nxt))
        return out

    def ix_at(p, q, xv):
        t = (xv - p[0]) / (q[0] - p[0])
        return (xv, p[1] + t * (q[1] - p[1]))

    def iy_at(p, q, yv):
        t = (yv - p[1]) / (q[1] - p[1])
        return (p[0] + t * (q[0] - p[0]), yv)

    pts = [tuple(p) for p in poly]
    pts = clip(pts, lambda p: p[0] >= x0, lambda p, q: ix_at(p, q, x0))
    if not pts:
        return []
    pts = clip(pts, lambda p: p[0] <= x1, lambda p, q: ix_at(p, q, x1))
    if not pts:
        return []
    pts = clip(pts, lambda p: p[1] >= y0, lambda p, q: iy_at(p, q, y0))
    if not pts:
        return []
    pts = clip(pts, lambda p: p[1] <= y1, lambda p, q: iy_at(p, q, y1))
    return pts


def polygon_area(pts):
    if len(pts) < 3:
        return 0.0
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) * 0.5


def footprint_weights(corners, nx, ny, dx):
    """Area-weighted overlap of one footprint polygon with the grid.

    Returns (flat_cell_indices, weights) with weights summing to the covered
    fraction of the polygon that falls inside the domain.
    """
    qx = corners[:, 0]
    qy = corners[:, 1]
    i0 = max(int(np.floor(qx.min() / dx)), 0)
    i1 = min(int(np.ceil(qx.max() / dx)), nx)
    j0 = max(int(np.floor(qy.min() / dx)), 0)
    j1 = min(int(np.ceil(qy.max() / dx)), ny)
    idx, wts = [], []
    for j in range(j0, j1):
        for i in range(i0, i1):
            pts = clip_polygon_to_box(corners, i * dx, (i + 1) * dx, j * dx, (j + 1) * dx)
            a = polygon_area(pts)
            if a > 0.0:
                idx.append(j * nx + i)
                wts.append(a)
    if not idx:
        return np.zeros(0, dtype=int), np.zeros(0)
    idx = np.array(idx, dtype=int)
    wts = np.array(wts)
    return idx, wts / wts.sum()
