# Regional NOx emission inversion from synthetic NO2 observations

You are given a synthetic but physically consistent observing system for a
240 km by 180 km air-quality domain, in `/app/data`. Six road-transport source
groups have been perturbed away from their reported inventory by unknown
constant factors. The meteorology delivered with the dataset is also imperfect:
the wind field carries an unknown speed bias and an unknown constant direction
error, and the chemical loss of NOx is represented by a single unknown
effective loss time. A smooth inflow field of unknown amplitude and tilt enters
the domain from outside.

Your job is the inverse problem an emission analyst actually faces. Separate
the emission signal from transport, chemical loss and inflow, using satellite
column footprints and surface concentration records from twelve observed
episodes, and then show that the state you inferred predicts observations from
six further episodes that you have never seen.

`/app/data/model_spec.md` is the authoritative statement of the physical model,
the unit system, the wind and rotation conventions, the boundary treatment, the
two observation operators, the data-quality rules and the error budget. Read it
before you touch the data: several conventions in it are not the ones you would
guess, and a wrong guess about any of them produces a fit that looks healthy
and predicts badly. `/app/data/data_manifest.json` carries the constants,
checksums and the representation-error budget. The numerical method is entirely
your choice; the specification constrains the physics, not the discretisation.

## What is unknown

Twelve quantities, all shared across every episode:

* six non-negative road emission scale factors, one per source group, within
  0.30 to 2.50;
* one effective NOx loss time, between 3600 s and 28800 s;
* one wind speed multiplier, between 0.70 and 1.30;
* one wind direction correction, between -20 and +20 degrees;
* three coefficients of the affine inflow background field, which must keep
  that field non-negative everywhere in the domain.

The prescribed non-road sources are not adjusted. The spatial pattern of each
road source group is exact; only its amplitude is in question.

## What you must deliver

Three files, at these absolute paths.

**`/app/result.json`** — the inferred parameter set:

| key | meaning |
| --- | --- |
| `schema_version` | string |
| `region_ids` | the six region identifiers from `domain.nc` |
| `emission_scale` | six scale factors, ordered to match `region_ids` |
| `effective_lifetime_s` | seconds |
| `wind_speed_scale` | dimensionless |
| `wind_rotation_deg` | degrees, sign as defined in the specification |
| `background_coefficients` | `[b0, bx, by]` in `mol m-2` |
| `background_units` | string |
| `integrated_road_mol_s` | mapping from episode id (`"0"` to `"17"`) to the domain-integrated corrected road NOx emission rate at that episode's reference time, in `mol s-1` |

The episode reference time is the `overpass_time` of `meteorology.nc`, so the
reported total must carry both the episode day-type factor and the diurnal
factor at that instant.

**`/app/posterior.nc`** — the inferred inventory on the inference grid of
`domain.nc`: `region_id`, `x`, `y` with cell bounds, `corrected_road_flux`
with dimensions `(region, y, x)` and `total_source_flux` with dimensions
`(y, x)`, both in `mol m-2 s-1` and both at the base period, that is with all
time factors set to one. `corrected_road_flux` is the supplied prior converted
into those units and multiplied by your scale factor for that region;
`total_source_flux` adds the prescribed non-road sources to their sum. The file
must declare its units.

**`/app/predicted_observations.csv`** — one row for every query listed in
`/app/data/prediction_queries.nc`, with header
`obs_id,instrument_type,predicted_value,unit`. `instrument_type` is `satellite`
or `station`; satellite predictions are NO2 columns in `mol m-2` and station
predictions are NO2 concentrations in `ug m-3`. Every query id must appear
exactly once, no other id may appear, and every value must be finite. Rows may
be in any order.

## How the work is judged

A sealed verifier reads only those three files. It checks that the schemas and
units are right; that your parameters are admissible and that the exported
inventory and the reported totals really follow from them; that an independent
forward run of your own reported parameters reproduces your own predicted
observations; and that those predictions match the withheld measurements.

Predictive skill is measured as the root mean square of
`(prediction - withheld value) / sigma_total`, with
`sigma_total^2 = sigma_measurement^2 + sigma_representation^2` as the
specification defines it. The withheld values are the noise-free model truth at
the query geometry, not noisy realisations of it, so the score measures your
model error directly in units of the observation error and there is no noise
floor to hide behind. The per-query `sat_uncertainty` and `sta_uncertainty` in
`prediction_queries.nc` are the measurement terms of that normalisation. Satellite and station scores are graded separately,
and each of the three wind regimes must meet its own threshold, so a good
average over regimes will not rescue a regime you have modelled badly. The
domain-integrated road emission is also compared against the value used to
generate the data. The thresholds are frozen and were calibrated to sit well
below what the unadjusted prior inventory and the uncorrected wind field
achieve. Grading is all-or-nothing: every check must pass.

You have 28800 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
