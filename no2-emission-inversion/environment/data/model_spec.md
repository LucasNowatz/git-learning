# Model specification and data conventions

This file defines the physical model, the unit system, the observation
operators and the file conventions of the supplied dataset. It is part of the
task input. It does not prescribe a numerical method.

Data in `/app/data` are **synthetic**. They were produced by an independently
implemented seeded forward model on a grid finer than the inference grid
defined in `domain.nc`. The spatial patterns of `road_prior` and
`fixed_sources` are exact: the only emission error is the set of per-region
scale factors defined below.

## 1. State variable and governing equation

The state is the vertically integrated NOx amount `C(x, y, t)` in `mol m-2`, on
the projected inference grid of `domain.nc`.

    dC/dt + div(u C) = div(K grad C) + E - C / tau

* `K` is the prescribed horizontal eddy diffusivity, constant, given by the
  `horizontal_diffusivity_m2_s` attribute of `meteorology.nc`.
* `tau` is a single effective NOx loss time, shared by the whole domain and all
  episodes. It is **not** an NO2 photochemical lifetime; it is the bulk loss
  time of the reduced model.
* `E` is the total NOx emission flux in `mol m-2 s-1`, defined in section 3.

Episodes are independent. Each is integrated from `episode_start` to
`episode_end` (`meteorology.nc`). No graded observation falls before
`analysis_start`; the interval before it is spin-up.

## 2. Wind, rotation conventions and boundary treatment

`meteorology.nc` supplies `u_east` and `v_north` on four layers whose
interfaces are in `level_bounds`, together with `blh`.

**Transport wind.** Form the layer-thickness weighted mean over the part of
each layer that lies below `blh`:

    w_k = max(min(blh, z_top_k) - z_bottom_k, 0)        (w_0 is floored at 1 m)
    u_supplied = sum_k w_k u_k / sum_k w_k

Form this mean **on the 12 km meteorological grid at the meteorological
times**, then interpolate the two resulting components bilinearly in space and
linearly in time onto the inference grid. Outside the hull of the
meteorological cell centres, clamp to the nearest edge value.

**Wind correction.** Two unknown corrections act on the supplied wind, in the
true east/north frame:

    [u'; v'] = a * R(delta) * [u_east; v_north],
    R(delta) = [[cos delta, -sin delta], [sin delta, cos delta]]

so a positive `delta` rotates the wind vector counter-clockwise when viewed
with east to the right and north upward. `delta` acts on the direction the wind
blows **toward**, not on a meteorological "from" bearing.

**Grid axes.** The projected grid axes are not aligned with east/north. Use
`grid_convergence` from `domain.nc`, the angle `g` from grid north to true
north, positive counter-clockwise:

    u_grid =  cos(g) u' + sin(g) v'
    v_grid = -sin(g) u' + cos(g) v'

Transport is carried out with `u_grid`, `v_grid` on the grid axes.

**Background and boundaries.** The inflow NOx column is

    B(x, y) = b0 + bx * xt + by * yt,     xt = x / Lx,  yt = y / Ly

with `Lx = 240000 m`, `Ly = 180000 m`, `x` and `y` the cell-centre coordinates
of `domain.nc`, and `B` in `mol m-2`. `B` must be non-negative everywhere in
the domain. It is used in exactly two places:

* as the initial condition, `C = B` at `episode_start`;
* as the exterior value at every domain boundary face where the flow enters.
  At outflow faces the adjacent interior value is used.

The diffusive flux across every domain boundary is zero. `B` is not added
anywhere else; there is no free post-processing offset.

## 3. Emissions, mass basis and time modulation

`emissions.nc` reports fluxes as mass per area per time, on **two different
mass bases**:

* `road_prior(region, y, x)` in `kg km-2 h-1` of **NO2-equivalent** mass;
* `fixed_sources(y, x)` in `kg km-2 h-1` of **nitrogen** mass.

Convert each with its own molar mass (`molar_mass_no2_kg_per_mol` and
`molar_mass_n_kg_per_mol`, both attributes of `emissions.nc`) to reach
`mol NOx m-2 s-1`. The two conversions are not interchangeable.

The total emission at time `t` in episode `e` is

    E = fixed_time_factor[e] * fixed_flux
      + road_time_factor[e] * d(t) * sum_r s_r * road_flux_r

where `s_r` are the six unknown non-negative region scale factors and `d(t)` is
`road_diurnal_factor` evaluated at the UTC hour of `t`, interpolated linearly
between whole-hour nodes and wrapped cyclically. `fixed_sources` is prescribed
and is never scaled.

## 4. Vertical shape and observation operators

The prescribed normalised vertical shape inside the boundary layer is

    g(z; x, y, t) = phi(z / h) / h,    h = blh(x, y, t)
    phi(zeta) = exp(-zeta / zeta0) / (zeta0 * (1 - exp(-1 / zeta0)))   for 0 <= zeta <= 1
    phi(zeta) = 0                                                       otherwise

with `zeta0` given by the `vertical_shape_zeta0` attribute of `meteorology.nc`.
`phi` integrates to one over `zeta` in `[0, 1]`, so `g` integrates to one over
height. The NO2 number density is `f * C * g`, with `f` the prescribed NO2/NOx
molar fraction `f_no2`, interpolated like the other meteorological fields.

**Satellite.** For footprint `i` with corners `corner_x`, `corner_y`:

    y_i = m_i * < f * C >_i

where `< . >_i` is the area-weighted average over the footprint polygon,
evaluated on the inference grid, `m_i` is `vertical_sensitivity` (the integral
of the averaging kernel against `g`, dimensionless), and `C` is taken at the
footprint's acquisition time. Every retained footprint lies wholly inside the
domain, so the overlap weights sum to the full polygon area. Units: `mol m-2`.

**Surface station.** For a site at `(x, y)` with inlet height `z_inlet`, the
model value is `f * C * g(z_inlet)` in the grid cell containing the site,
averaged over the reported interval, converted to `ug m-3` of NO2 with the NO2
molar mass. All inlets are inside the boundary layer at all reported times.

## 5. Data handling rules

* Times are seconds since 1970-01-01T00:00:00Z. There is a single time
  convention across all files.
* `no2_column` in the swath files is packed. Decode with `scale_factor` and
  `add_offset` and drop `_FillValue` entries **before** applying the QA rule.
* Accept a retrieval only when `qa_value >= 0.75` (the
  `qa_acceptance_threshold` attribute). Rejected retrievals are cloud-affected
  and biased; they are not usable by rescaling.
* Valid retrievals with negative values are genuine noise realisations of a
  small positive quantity. Retain them. Clipping them biases the inversion.
* Retrieval errors are independent between footprints, and station errors are
  independent between records. This is a disclosed simplification.
* `obs_id` is unique across the whole dataset.

## 6. Error budget

Total error for an observation is

    sigma_total^2 = sigma_measurement^2 + sigma_representation^2

`sigma_measurement` is the per-observation uncertainty in the swath files,
`stations.csv` and `prediction_queries.nc`. `sigma_representation` is given in
`data_manifest.json` under `representation_error`, separately for each
instrument. It was measured as the root-mean-square difference between the
generating model and the same physical state evaluated on the 4 km inference
grid.

The withheld values used for grading are noise-free model truth evaluated at
the query geometry, so a prediction is scored on model error alone. The
`sigma_measurement` reported for each query is still the measurement term of the
normalisation defined above.

That representation error applies to a **conservative, second-order or better**
transport scheme run at a time step of 60 s or less on the 4 km grid. A
first-order upwind scheme on this grid adds numerical diffusion well above the
quoted value and will not reproduce the plume structure the data contain.
