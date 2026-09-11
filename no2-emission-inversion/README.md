# afterquery/no2-emission-inversion

Meteorology-aware regional NOx emission inversion from synthetic satellite NO2
column footprints and surface concentration records, graded on withheld
episodes.

Domain `earth-sciences` / field `atmospheric-sciences` / subcategory
`Atmospheric & Climate Sciences`.

## Difficulty

An expert who already writes transport and retrieval code needs roughly a
day: the work is a long chain of dependent modelling decisions, each stated in
`/app/data/model_spec.md`, none of which is the default choice an agent reaches
for, and every one of which must be right at the same time.

The chain is: decode packed satellite columns and apply the quality rule before
screening, keeping valid negative retrievals; convert two emission fields that
are published on different mass bases, NO2-equivalent for the road inventory and
nitrogen for the fixed sources, each with its own molar mass; form the
boundary-layer mean wind on the meteorological grid before interpolating it,
not after; compose the unknown wind rotation with the spatially varying grid
convergence in the right sense; integrate a conservative advection-diffusion
equation with a single effective loss time, with the unknown inflow field used
as both initial condition and inflow boundary value and zero diffusive flux
across the boundary; average the model field over irregular quadrilateral
footprints by area-weighted overlap and apply the per-pixel vertical
sensitivity; and evaluate the surface operator through the prescribed
normalised in-layer vertical shape.

What makes this unforgiving rather than merely long is that the twelve free
parameters absorb a broken link. A solver that inverts the rotation sign, or
that uses the NO2 molar mass for the nitrogen-basis field, or that averages
footprints by nearest cell centre instead of by area, still fits the twelve
training episodes at a plausible chi-square, because the scale factors, the
loss time and the background silently compensate. The compensation is
regime-dependent, so it collapses on the six withheld episodes and on at least
one of the three wind regimes, which are graded separately.

The inference is a twelve-parameter joint estimation with real structural
trade-offs: emission amplitude against effective loss time, wind speed against
loss time, broad sources against inflow background. They separate only when
several distinct wind regimes are used together. Individual region factors
remain more weakly constrained than their domain total, and the total is what
is graded.

**The data are synthetic**, and the instruction says so. They were produced by
the seeded generator in `authoring/provenance/generator/`, which is implemented
separately from the reference solution, integrates on a 2 km grid while the
published inference grid is 4 km, and was run once with latent parameters fixed
in advance. The representation error between the two grids is measured and
published in `data_manifest.json`, so a fair solution is never penalised for
discretisation.

## Reference solution

`solution/no2_inversion.py`, driven by `solution/solve.sh`.

For a fixed loss time and fixed wind corrections the stated transport model is
linear in the six road scale factors and the three background coefficients. The
solution exploits that: it integrates ten basis states per episode, one for the
prescribed non-road sources, six for the road groups and three for a corner
parametrisation of the background field, assembles the design matrix against the
accepted observations weighted by the total error, and solves the inner problem
with a bounded linear least squares. Only three nonlinear parameters are
searched, with a coarse grid on one episode per wind regime, Nelder-Mead on that
subset, and a final Nelder-Mead pass over all twelve training episodes. This is
one valid route, not a required one; the instruction leaves the numerical method
open.

Transport uses a flux-form finite-volume scheme with a van Leer limited
reconstruction at a 60 s time step, first order at boundary faces, the
background field in the ghost cells and zero diffusive flux across the boundary.
Fitted parameters are then propagated through the six withheld episodes to
produce the graded predictions. The solution never reads the generator, its seed
or the sealed evaluation values.

Measured oracle behaviour and the baselines it is calibrated against are in
`authoring/evidence/`.

## Verification

`tests/test_no2_inversion.py` reads only the three declared artifacts and awards
reward 1 only if every gate passes.

1. **Schema and completeness.** All three files present, parseable and finite;
   correct dimensions and region identifiers; declared units; exactly one
   prediction per query id and no foreign ids.
2. **Physical admissibility.** Parameters inside the published bounds; the
   affine background field non-negative at every domain corner, hence
   everywhere.
3. **Internal consistency of the inventory.** The exported `corrected_road_flux`
   and `total_source_flux` must equal the supplied priors converted by the
   verifier's own unit handling and scaled by the reported factors, and the
   reported per-episode totals must follow from those same factors and the
   published time modulation. This is where a wrong emission mass basis is
   caught.
4. **Prediction consistency.** `tests/verifier_forward.py` re-runs the forward
   model from the submitted parameters with a different discretisation, Strang
   directional splitting instead of an unsplit update, and requires the
   submitted predictions to agree with it.
5. **Held-out skill and regime robustness.** Error-normalised RMSE against the
   withheld satellite and surface measurements, each below a frozen threshold,
   and each of the three wind regimes below its own threshold so that a good
   average cannot conceal a failed regime.
6. **Identifiable recovery.** The domain-integrated road NOx emission rate is
   compared with the value used to generate the data. Individual region factors
   are not graded, because the identifiability study in
   `authoring/evidence/` shows the public data do not constrain all six
   separately.

Ground truth is the generator's own withheld output, baked into the verifier
image. Nothing in `environment/` references the solution or the tests, and the
sealed values never appear in the agent's container. Thresholds were frozen from
measured reference performance and documented baselines before any agent run.
