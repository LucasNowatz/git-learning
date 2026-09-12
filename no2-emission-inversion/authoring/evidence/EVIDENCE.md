# Measured evidence

Everything here was measured on this bundle before any frontier-agent run.
Regenerate with `score_output.py`, `baselines.py`, `cheat_attempts.py`,
`freeze_thresholds.py` and this script.

## Reference solution

| quantity | value |
| --- | --- |
| held-out satellite WRMSE | 0.3049 |
| held-out station WRMSE | 0.5428 |
| NE_continental satellite / station WRMSE | 0.3715 / 0.6330 |
| SW_advective satellite / station WRMSE | 0.2707 / 0.4706 |
| S_light_variable satellite / station WRMSE | 0.2611 / 0.5116 |
| prediction consistency, satellite / station | 0.0943 / 0.1011 |
| domain-integrated road emission error | +4.02 % |
| wind rotation error | +0.053 deg |
| wind speed scale error | +0.63 % |
| vertical shape parameter error | -1.09 % |
| reference loss time error | -27.4 % |
| saturation column error | -35.9 % |
| non-road scale error | +25.3 % |

The split in that table is the identifiability result. The wind
correction and the vertical shape parameter are recovered to a fraction
of a percent. The reference loss time, the saturation column and the
non-road scale are not: they lie along a degenerate direction in which
emission amplitude, loss time and saturation column trade against one
another with almost no change in the fit. That is why no gate compares
them, or the absolute road total, with the truth. Oracle wall time on
four cores is about twenty-three minutes, of which the artefact
screening loop is three passes of the full nonlinear fit.

## Deliberately broken variants

Each variant refits every free parameter with one link of the modelling
chain broken, predicts the withheld episodes with that same flawed model,
and is scored exactly as an agent would be.

| variant | sat WRMSE | sta WRMSE | consistency | train chi2/n | verdict |
| --- | --- | --- | --- | --- | --- |
| *reference solution* | 0.305 | 0.543 | 0.101 | 0.960 | **pass** |
| first_order_loss | 0.495 | 0.739 | 0.092 | 1.072 | rejected: SW_advective regime, S_light_variable regime |
| fixed_sources_unscaled | 0.315 | 0.579 | 0.101 | 0.968 | passes the graded gates |
| no_artefact_screening | 0.929 | 0.920 | 0.101 | 2.188 | rejected: NE_continental regime, SW_advective regime, S_light_variable regime, satellite skill, station skill |
| no_wind_correction | 1.701 | 1.667 | 0.067 | 3.157 | rejected: NE_continental regime, SW_advective regime, S_light_variable regime, satellite skill, station skill |
| prior_unadjusted | 1.058 | 1.578 | 0.076 | 1.504 | rejected: NE_continental regime, SW_advective regime, S_light_variable regime, satellite skill, station skill |
| vertical_shape_assumed | 0.370 | 0.629 | 0.099 | 0.990 | passes the graded gates |

The training chi-square column is what the solver itself sees. It is the
only warning that anything is wrong, and it is loud for the artefact
(2.19 against 0.96) and quiet for everything else.

* `no_artefact_screening` is the central trap. Fitting the accepted
  retrievals as delivered costs a factor of three on held-out satellite
  skill. The specification says the quality flag is not exhaustive and
  that establishing which retrievals are fit to use is part of the
  analysis; it does not say which across-track positions or from when.
* `first_order_loss` pins the saturation column at the top of its range,
  which is what a solver that treats the sink as first order would have
  to report. It survives the global skill gates and is rejected by two of
  the three regime gates, which is what those gates are for.
* `fixed_sources_unscaled` and `vertical_shape_assumed` are **not**
  rejected. Assuming the non-road inventory is unbiased costs almost
  nothing, and assuming a plausible vertical shape costs little. Both are
  genuine identifiability limits, reported as such rather than claimed as
  traps.

## Adversarial attempts

Each entry is the laziest output that might satisfy a weak verifier.

| attempt | reward | rejected by |
| --- | --- | --- |
| background_units_wrong | 0 | background_units must be 'mol m-2', found 'molecules cm-2' |
| constant_predictions | 0 | submitted satellite predictions disagree with an independent forward run of the submitted parameters (normalis |
| duplicate_rows | 0 | duplicate obs_id 'SAT-E09-00000' |
| emission_scale_doubled | 0 | corrected_road_flux does not equal emission_scale times the supplied prior converted to mol m-2 s-1 (max relat |
| fixed_scale_reported_as_one | 0 | total_source_flux does not equal the corrected road flux plus the prescribed non-road sources (max relative de |
| foreign_obs_ids | 0 | unknown obs_id 'SAT-E09-00000-X' in predicted_observations.csv |
| garbage_nonlinear_parameters | 0 | submitted satellite predictions disagree with an independent forward run of the submitted parameters (normalis |
| loss_time_outside_bounds | 0 | reference_loss_time_s = 900.0 is outside the published range [1800.0, 28800.0] |
| missing_rows | 0 | 50 query ids have no prediction, e.g. ['STA-E17-ST04-4', 'STA-E17-ST04-5', 'STA-E17-ST05-0'] |
| nan_predictions | 0 | non-finite prediction for 'SAT-E09-00000' |
| negative_background_field | 0 | the background field is negative somewhere in the domain: corners [np.float64(1e-06), np.float64(-7e-06), np.f |
| no_posterior_file | 0 | posterior.nc missing or empty |
| nothing_written | 0 | result.json missing or empty |
| posterior_on_wrong_mass_basis | 0 | corrected_road_flux does not equal emission_scale times the supplied prior converted to mol m-2 s-1 (max relat |
| posterior_without_cell_bounds | 0 | posterior.nc is missing cell bounds 'x_bnds' |
| reported_rotation_sign_flipped | 0 | submitted satellite predictions disagree with an independent forward run of the submitted parameters (normalis |
| scales_pinned_at_bound | 0 | corrected_road_flux does not equal emission_scale times the supplied prior converted to mol m-2 s-1 (max relat |
| schema_version_missing | 0 | schema_version must be the string "1.0", found None |
| totals_missing_diurnal_factor | 0 | integrated_road_mol_s is inconsistent with the reported scale factors (worst relative deviation 2.944e-01); it |
| training_mean_predictions | 0 | submitted satellite predictions disagree with an independent forward run of the submitted parameters (normalis |
| vertical_shape_reported_as_default | 0 | submitted station predictions disagree with an independent forward run of the submitted parameters (normalised |
| zero_predictions | 0 | submitted satellite predictions disagree with an independent forward run of the submitted parameters (normalis |
| *reference solution* | 1 | - |

## Frozen thresholds

There is no gate on the absolute road emission total, and none on the
individual chemistry parameters. The identifiability result above is why.

```json
{
  "bounds": {
    "emission_scale": [
      0.3,
      2.5
    ],
    "fixed_source_scale": [
      0.4,
      2.2
    ],
    "loss_saturation_column": [
      1e-05,
      0.001
    ],
    "reference_loss_time_s": [
      1800.0,
      28800.0
    ],
    "vertical_shape_zeta0": [
      0.12,
      0.8
    ],
    "wind_rotation_deg": [
      -20.0,
      20.0
    ],
    "wind_speed_scale": [
      0.7,
      1.3
    ]
  },
  "consistency_max": {
    "satellite": 0.556,
    "station": 0.556
  },
  "regime": {
    "NE_continental": {
      "satellite": 0.65,
      "station": 0.918
    },
    "SW_advective": {
      "satellite": 0.474,
      "station": 0.682
    },
    "S_light_variable": {
      "satellite": 0.457,
      "station": 0.742
    }
  },
  "skill": {
    "satellite": 0.534,
    "station": 0.787
  }
}
```

Frozen by authoring/evidence/freeze_thresholds.py from measured reference performance, before any agent run. Each gate is a fixed margin above what the reference solution achieves, validated against the broken variants in EVIDENCE.md. Do not edit by hand.
