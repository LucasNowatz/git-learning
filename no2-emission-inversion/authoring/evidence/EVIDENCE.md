# Measured evidence

Everything here was measured on this bundle before any frontier-agent run.
Regenerate with `score_output.py`, `baselines.py`, `cheat_attempts.py`,
`freeze_thresholds.py` and this script.

## Reference solution

| quantity | value |
| --- | --- |
| held-out satellite WRMSE | 0.2437 |
| held-out station WRMSE | 0.4952 |
| NE_continental satellite / station WRMSE | 0.2785 / 0.4855 |
| SW_advective satellite / station WRMSE | 0.2763 / 0.2559 |
| S_light_variable satellite / station WRMSE | 0.1536 / 0.6592 |
| prediction consistency, satellite / station | 0.0714 / 0.0859 |
| domain-integrated road emission error | +5.84 % |
| recovered effective loss time | 4.385 h (true 4.600 h) |
| recovered wind speed scale | 1.2053 (true 1.1800) |
| recovered wind rotation | -13.358 deg (true -13.500 deg) |

Individual region scale factors are recovered to between 2 and 20 percent,
which is why only their domain total is graded. Oracle wall time on four
cores is about eleven minutes.

## Deliberately broken variants

Each variant refits every free parameter with one link of the modelling
chain broken, predicts the withheld episodes with that same flawed model,
and is scored exactly as an agent would be.

| variant | sat WRMSE | sta WRMSE | consistency | total err | train chi2/n | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| *reference solution* | 0.244 | 0.495 | 0.086 | 0.058 | 0.964 | **pass** |
| clipped_negatives | 0.245 | 0.494 | 0.086 | 0.049 | 0.868 | passes the graded gates |
| coarse_time_step_300s | 1.700 | 2.858 | 0.105 | 0.515 | 26.607 | rejected: NE_continental regime, SW_advective regime, S_light_variable regime, emission total, satellite skill, station skill |
| first_order_upwind | 0.706 | 1.044 | 0.850 | 0.029 | 1.340 | rejected: NE_continental regime, SW_advective regime, S_light_variable regime, satellite skill, station skill |
| grid_convergence_ignored | 0.293 | 0.525 | 0.176 | 0.041 | 0.994 | passes the graded gates |
| grid_convergence_sign_flipped | 0.399 | 0.600 | 0.336 | 0.024 | 1.071 | passes the graded gates |
| nearest_cell_footprint | 0.345 | 0.490 | 0.320 | 0.026 | 1.019 | rejected: S_light_variable regime |
| no_diurnal_factor | 0.365 | 0.942 | 2.276 | 0.122 | 1.123 | rejected: SW_advective regime, S_light_variable regime, consistency, emission total, station skill |
| no_qa_screening | 1.149 | 1.149 | 0.088 | 0.023 | 11.242 | rejected: NE_continental regime, SW_advective regime, S_light_variable regime, satellite skill, station skill |
| no_rotation | 1.499 | 1.780 | 0.082 | 0.034 | 3.234 | rejected: NE_continental regime, SW_advective regime, S_light_variable regime, satellite skill, station skill |
| no_wind_correction | 1.499 | 1.780 | 0.081 | 0.009 | 3.235 | rejected: NE_continental regime, SW_advective regime, S_light_variable regime, satellite skill, station skill |
| prior_unadjusted | 1.674 | 2.909 | 0.062 | 0.253 | 4.410 | rejected: NE_continental regime, SW_advective regime, S_light_variable regime, emission total, satellite skill, station skill |
| wrong_mass_basis | 0.283 | 0.532 | 0.166 | 0.061 | 0.978 | passes the graded gates |

The table scores only the graded prediction and parameter gates, because
the variants do not write a posterior file. Four entries need a note.

* `wrong_mass_basis` is not caught by the skill gates: the free parameters
  absorb most of a mis-scaled non-road source term. It is caught by the
  exported-inventory gate. Completing that variant with the posterior file
  such a solver would write, and running the sealed verifier on the whole
  triple, gives reward 0 with `total_source_flux does not equal the
  corrected road flux plus the prescribed non-road sources (max relative
  deviation 2.456e-01); check the nitrogen mass basis of fixed_sources`.
* `clipped_negatives` is genuinely harmless at this signal level and is
  reported as such rather than claimed as a trap.
* The grid convergence angle spans about six degrees across the domain, so
  ignoring it, or taking it with the wrong sign, is largely absorbed by the
  free rotation parameter. Both variants pass. The convention is documented
  because it is needed for a correct model, not because it is a trap.
* `first_order_upwind` and `coarse_time_step_300s` both fail. `model_spec.md`
  states in advance that a first-order scheme and a time step above 60 s are
  inadequate on the 4 km grid, so neither was used to set a threshold; they
  fail on their own merits.

## Adversarial attempts

Each entry is the laziest output that might satisfy a weak verifier.

| attempt | reward | rejected by |
| --- | --- | --- |
| constant_predictions | 0 | submitted satellite predictions disagree with an independent forward run of the submitted parameters (normalis |
| duplicate_rows | 0 | duplicate obs_id 'SAT-E09-00000' |
| emission_scale_doubled | 0 | corrected_road_flux does not equal emission_scale times the supplied prior converted to mol m-2 s-1 (max relat |
| foreign_obs_ids | 0 | unknown obs_id 'SAT-E09-00000-X' in predicted_observations.csv |
| garbage_nonlinear_parameters | 0 | submitted satellite predictions disagree with an independent forward run of the submitted parameters (normalis |
| missing_rows | 0 | 50 query ids have no prediction, e.g. ['STA-E17-ST04-4', 'STA-E17-ST04-5', 'STA-E17-ST05-0'] |
| nan_predictions | 0 | non-finite prediction for 'SAT-E09-00000' |
| negative_background_field | 0 | the background field is negative somewhere in the domain: corners [np.float64(1e-06), np.float64(-7e-06), np.f |
| no_posterior_file | 0 | posterior.nc missing or empty |
| nothing_written | 0 | E   KeyError: 'params' |
| posterior_on_wrong_mass_basis | 0 | corrected_road_flux does not equal emission_scale times the supplied prior converted to mol m-2 s-1 (max relat |
| reported_rotation_sign_flipped | 0 | submitted satellite predictions disagree with an independent forward run of the submitted parameters (normalis |
| scales_pinned_at_bound | 0 | corrected_road_flux does not equal emission_scale times the supplied prior converted to mol m-2 s-1 (max relat |
| totals_missing_diurnal_factor | 0 | integrated_road_mol_s is inconsistent with the reported scale factors (worst relative deviation 2.864e-01); it |
| training_mean_predictions | 0 | submitted satellite predictions disagree with an independent forward run of the submitted parameters (normalis |
| zero_predictions | 0 | submitted satellite predictions disagree with an independent forward run of the submitted parameters (normalis |
| *reference solution* | 1 | - |

## Frozen thresholds

```json
{
  "bounds": {
    "effective_lifetime_s": [
      3600.0,
      28800.0
    ],
    "emission_scale": [
      0.3,
      2.5
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
    "satellite": 0.874,
    "station": 0.874
  },
  "regime": {
    "NE_continental": {
      "satellite": 0.604,
      "station": 0.739
    },
    "SW_advective": {
      "satellite": 0.6,
      "station": 0.39
    },
    "S_light_variable": {
      "satellite": 0.333,
      "station": 1.004
    }
  },
  "skill": {
    "satellite": 0.529,
    "station": 0.754
  },
  "total_emission_rel_tol": 0.117
}
```

Frozen by authoring/evidence/freeze_thresholds.py from measured reference and baseline scores, before any agent run. Each global gate is the geometric mean of what the reference solution achieves and the best score reached by a deliberately broken variant; each regime gate is the global gate rescaled by the reference score in that regime. Do not edit by hand.
