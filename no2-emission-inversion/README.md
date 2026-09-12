# afterquery/no2-emission-inversion

Meteorology-aware regional NOx emission inversion from synthetic satellite NO2
column footprints and surface concentration records, graded on withheld
episodes.

Domain `earth-sciences` / field `atmospheric-sciences` / subcategory
`Atmospheric & Climate Sciences`.

## Difficulty

An expert who already writes transport and retrieval code needs on the order of
a day. The work is a long chain of dependent modelling decisions, each stated in
`/app/data/model_spec.md`, plus one thing the specification deliberately does not
state: which retrievals are fit to use.

**The chemistry does not superpose.** The NOx sink saturates, because the
oxidant that removes NOx is suppressed as NOx rises:
`L(C) = C P / (tau_0 (1 + C / C_ref))`. Two consequences follow. The column from
two sources together is not the sum of the columns each produces alone, so there
is no basis decomposition and no inner linear least squares to hide behind:
every one of the fifteen unknowns has to be carried through a nonlinear fit.
And the inflow background lengthens the lifetime of the emitted plumes, so
background, emissions and loss are entangled rather than merely correlated.
`tau_0` and `C_ref` separate only because the data span a wide range of column
amounts; a solver that does not check that will find a ridge instead of a
minimum.

**The quality flag is not an artefact detector.** A contiguous band of
across-track positions develops a gain and offset error partway through the
record, as real ultraviolet imagers do. It passes the quality screen, and the
reported per-pixel uncertainty does not cover it. The specification says the
flag is not exhaustive and that establishing which retrievals are fit to use is
part of the analysis; it does not say which positions or from when. Fitting the
accepted set as delivered gives a chi-square per observation near 1.5 rather
than 1, which is the only signal that anything is wrong. Finding the band means
fitting once, binning normalised residuals by across-track index and episode,
and looking. Missing it biases the emission scales, the background and the wind
correction together.

**The withheld episodes are not a rerun of the observed ones.** They span a
wider range of day-type emission factors than the twelve observed episodes, so a
saturation column calibrated only over the observed concentration range
extrapolates badly. This is where a fit that looks converged comes apart.

**The remaining links still bite.** Two emission fields published on different
mass bases, NO2-equivalent and nitrogen, each needing its own molar mass. A
boundary-layer mean wind formed on the meteorological grid before interpolating,
not after. An unknown rotation composed with the spatially varying grid
convergence in the right sense. A conservative transport solve with the unknown
inflow field as both initial condition and inflow boundary value. Area-weighted
polygon footprint averaging with a per-pixel vertical sensitivity. A surface
operator through a vertical profile whose shape parameter is itself unknown,
which is what forces the column and surface data to be used together.

What makes this unforgiving rather than merely long is that the fifteen free
parameters absorb a broken link. Every candidate mistake was refitted end to end
with that one link broken, then scored exactly as an agent would be; the table
is in `authoring/evidence/EVIDENCE.md`.

**The data are synthetic**, and the instruction says so. They were produced by
the seeded generator in `authoring/provenance/generator/`, which is implemented
separately from the reference solution, integrates on a 2 km grid while the
published inference grid is 4 km, and was run once with latent parameters fixed
in advance. The representation error between the two grids is measured and
published in `data_manifest.json` as a floor and a relative coefficient, so a
fair solution is never penalised for discretisation.

## Reference solution

`solution/no2_inversion.py`, driven by `solution/solve.sh`.

The saturating sink removes any basis decomposition, so all fifteen unknowns are
fitted jointly by bounded nonlinear least squares with a finite-difference
Jacobian, with the forward runs for the twelve observed episodes distributed
over the available cores. The instrument artefact is found the way an analyst
finds it, and without reading the generator: fit once over the whole accepted
set, bin the normalised residuals by across-track index within each episode,
flag the positions carrying a significant systematic offset, drop them and fit
again. A restart from a distant point checks that the optimiser has not stopped
in a local minimum. This is one valid route, not a required one; the instruction
leaves the numerical method open.

Transport uses a flux-form finite-volume scheme with a van Leer limited
reconstruction at a 60 s time step, first order at boundary faces, the
background field in the ghost cells and zero diffusive flux across the boundary.
The loss coefficient is evaluated at the current column and applied as an
exponential update over the step.
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
   verifier's own unit handling and scaled by the reported road factors and
   `fixed_source_scale`, and the reported per-episode totals must follow from
   those same factors and the published time modulation. This is where a wrong
   emission mass basis is caught.
4. **Prediction consistency.** `tests/verifier_forward.py` re-runs the forward
   model from the submitted parameters with a different discretisation, Strang
   directional splitting instead of an unsplit update, and requires the
   submitted predictions to agree with it.
5. **Held-out skill and regime robustness.** Error-normalised RMSE against the
   withheld satellite and surface measurements, each below a frozen threshold,
   and each of the three wind regimes below its own threshold so that a good
   average cannot conceal a failed regime.
There is deliberately no sixth gate comparing inferred emissions with the
truth. The identifiability study in `authoring/evidence/EVIDENCE.md` shows the
public data do not constrain the absolute road total or the individual
chemistry parameters: emission amplitude, reference loss time and saturation
column trade against one another with almost no change in the fit, while the
wind correction and the vertical shape parameter are recovered to a fraction of
a percent. Grading a quantity the data do not determine would fail correct
work, so prediction agreement carries the evidence.

Ground truth is the generator's own withheld output, baked into the verifier
image. Nothing in `environment/` references the solution or the tests, and the
sealed values never appear in the agent's container. Thresholds are a fixed margin above measured reference performance, 1.75 times
on satellite and 1.45 times on station, and were then validated against the
broken variants: each of the four that must fail is rejected by at least one
gate. They were frozen before any agent run.
