# Provenance of the dataset and the ground truth

Nothing in this directory is ever mounted into a container. It records where the
data came from and how the sealed evaluation values were produced.

## Generator

`generator/` holds the complete, seeded truth generator:

| file | role |
| --- | --- |
| `config.py` | frozen geometry, constants, episode layout and the latent truth |
| `fields.py` | emission patterns, meteorology, station network, swath geometry |
| `physics.py` | truth-side transport, vertical shape and footprint operators |
| `generate.py` | driver: runs the truth, measures representation error, writes files |
| `writers.py` | NetCDF and CSV serialisation, noise, quality flags, manifest |
| `refresh_manifest.py` | recompute `data_manifest.json` after a dataset edit |

Reproduce the whole dataset with

    cd authoring/provenance/generator
    python3 generate.py --out <bundle-root>
    python3 ../refresh_manifest.py

`MASTER_SEED = 20260911` in `config.py` drives every random draw. That seed is
never available to the agent: it appears only here and the environment contains
no generator code. The instruction and `model_spec.md` both disclose that the
data are synthetic without disclosing how to reconstruct them.

## The latent state

The twelve latent parameters in `config.py` (`TRUE_SOURCE_SCALE`,
`TRUE_LIFETIME_S`, `TRUE_WIND_SPEED_SCALE`, `TRUE_WIND_ROTATION_DEG`,
`TRUE_BACKGROUND`) were chosen before any inversion was attempted and were not
adjusted afterwards. None sits on a published bound. The published bounds are
wide enough that the truth is not inferable from them.

The generator integrates the state on a **2 km** grid with a 30 s time step,
using a flux-form finite-volume scheme with a van Leer limited reconstruction.
The published inference grid in `domain.nc` is **4 km**. Observations are drawn
from the 2 km state, so a solver working on the published grid faces a genuine
representation error.

The road source patterns and the fixed-source field are built on the 2 km grid
and coarse-grained to 4 km by area-weighted averaging before publication. That
makes the published spatial patterns exact in the coarse-cell sense, so the only
emission unknowns are the six amplitudes. This is stated in `model_spec.md`; it
is what makes the scale factors a well-posed target rather than an arbitrary one.

## Representation error

`generate.py` runs the identical latent state a second time on the 4 km grid
with a 60 s step and reports the root-mean-square difference in observation
space:

| instrument | representation error |
| --- | --- |
| satellite column | 1.974e-06 mol m-2 |
| surface station | 1.402 ug m-3 |

Those values are published in `data_manifest.json` and enter the grading
normalisation as `sigma_representation`. They are measured, not assumed, and
they are measured independently of the reference solution.

## Observation error and quality flags

Satellite noise is Gaussian and independent between footprints, with a
per-footprint 1-sigma drawn uniformly from 5.5e-06 to 1.15e-05 mol m-2. About
16 percent of footprints are marked cloud-affected: their quality value is drawn
below the acceptance threshold and their reported column carries a large
positive bias, so a solver that ignores the quality rule inherits that bias. A
further 3 percent carry the fill value. The remaining quality values come from a
Beta(7, 1) draw, so a further tenth of otherwise clean retrievals fall below the
threshold, as in a real conservative quality screen.

Station noise is Gaussian and independent, with a per-site 1-sigma between 1.8
and 3.6 ug m-3, modulated per record.

Columns are stored as packed 16-bit integers with `scale_factor = 2e-08` and
`_FillValue = -32767`, written without library auto-scaling so that the file
contains exactly the intended integers.

## Split

Episodes 0-8 and 12-14 are published with full observations. Episodes 9, 10, 11,
15, 16 and 17 are withheld, two from each of the three wind regimes, and appear
in `prediction_queries.nc` with complete geometry, timing and meteorology but no
measured values. The split was frozen before thresholds were calibrated. Every
wind regime is represented in both parts, so no regime has to be extrapolated
from nothing.

## What the verifier holds

`tests/truth/evaluation.npz` carries the noise-free truth at each withheld query,
the per-query measurement sigma, the regime label, the representation errors, the
latent parameters and the true domain-integrated road emission per episode.
`tests/inputs/` carries copies of the four input files the verifier needs to
rerun the forward model. Neither is reachable from the agent's container.
