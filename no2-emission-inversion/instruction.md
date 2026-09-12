# Regional NOx emission inversion from synthetic NO2 observations

`/app/data` holds a synthetic observing system for a 240 km by 180 km
air-quality domain: satellite NO2 column footprints and surface concentration
records from twelve observed episodes, the prior road NOx inventory, the
meteorology, and the geometry and timing of six further episodes whose
measurements are withheld.

Infer the fifteen unknown quantities listed below from the observed episodes,
and predict the withheld observations.

`/app/data/model_spec.md` defines the physical model, the unit system, the wind
and rotation conventions, the boundary treatment, the two observation
operators, the data-quality rules and the error budget.
`/app/data/data_manifest.json` carries the constants, the file checksums and
the representation-error budget. The numerical method is your choice; the
specification constrains the physics, not the discretisation.

## What is unknown

Fifteen quantities, shared across every episode:

* six non-negative road emission scale factors, one per source group, within
  0.30 to 2.50;
* the low-NOx reference loss time of the saturating chemical sink, between
  1800 s and 28800 s;
* the saturation column of that sink, between 1.0e-5 and 1.0e-3 mol m-2;
* one wind speed multiplier, between 0.70 and 1.30;
* one wind direction correction, between -20 and +20 degrees;
* three coefficients of the affine inflow background field, which must keep
  that field non-negative everywhere in the domain;
* one scale factor on the non-road source inventory, between 0.40 and 2.20;
* the shape parameter of the normalised vertical profile, between 0.12 and
  0.80.

The spatial pattern of each road source group and of the non-road field is
exact; only the amplitudes are unknown.

## What you must deliver

Three files, at these absolute paths.

### `/app/result.json`

The inferred parameter set:

| key | meaning |
| --- | --- |
| `schema_version` | string, `"1.0"` |
| `region_ids` | the six region identifiers from `domain.nc` |
| `emission_scale` | six scale factors, ordered to match `region_ids` |
| `reference_loss_time_s` | seconds |
| `loss_saturation_column` | `mol m-2` |
| `loss_saturation_units` | the string `mol m-2` |
| `wind_speed_scale` | dimensionless |
| `wind_rotation_deg` | degrees, sign as defined in the specification |
| `background_coefficients` | `[b0, bx, by]` in `mol m-2` |
| `background_units` | the string `mol m-2` |
| `fixed_source_scale` | dimensionless |
| `vertical_shape_zeta0` | dimensionless |
| `integrated_road_mol_s` | mapping from episode id (`"0"` to `"17"`) to the domain-integrated corrected road NOx emission rate at that episode's reference time, in `mol s-1` |

The episode reference time is the `overpass_time` of `meteorology.nc`, so the
reported total carries both the episode day-type factor and the diurnal factor
at that instant.

### `/app/posterior.nc`

The inferred inventory on the inference grid of `domain.nc`: `region_id`, `x`,
`y`, their cell bounds `x_bnds` and `y_bnds` with dimensions `(x, nv)` and
`(y, nv)`, `corrected_road_flux` with dimensions `(region, y, x)`
and `total_source_flux` with dimensions `(y, x)`, both in `mol m-2 s-1` and
both at the base period, that is with all time factors set to one.
`corrected_road_flux` is the supplied prior converted into those units and
multiplied by your scale factor for that region; `total_source_flux` adds the
non-road field, converted and multiplied by your `fixed_source_scale`, to their
sum. The file must declare its units.

### `/app/predicted_observations.csv`

One row for every query listed in `/app/data/prediction_queries.nc`, with
header `obs_id,instrument_type,predicted_value,unit`. `instrument_type` is
`satellite` or `station`; satellite predictions are NO2 columns in `mol m-2`
and station predictions are NO2 concentrations in `ug m-3`. Every query id must
appear exactly once, no other id may appear, and every value must be finite.
Rows may be in any order.

## How the work is judged

A sealed verifier reads only those three files, and grading is all-or-nothing:
every check must pass. It checks that the schemas and units are right; that the
parameters are admissible and that the exported inventory and the reported
totals follow from them; that an independent forward run of the reported
parameters reproduces the submitted predictions; and that those predictions
match the withheld measurements.

Predictive skill is the root mean square of
`(prediction - withheld value) / sigma_total`, with
`sigma_total^2 = sigma_measurement^2 + sigma_representation^2` as the
specification defines it. The withheld values are the noise-free model truth at
the query geometry. The per-query `sat_uncertainty` and `sta_uncertainty` in
`prediction_queries.nc` are the measurement terms of that normalisation.
Satellite and station scores are graded separately, and each of the three
wind regimes is graded against its own threshold.

You have 28800 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
