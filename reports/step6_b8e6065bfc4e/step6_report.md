# Step 6 training results

Run: `step6_b8e6065bfc4e`

All candidates were ranked by mean grouped cross-validation MAE. Selection was frozen
before final evaluation. No holdout or Na/K labels were used for fitting or selection.

| Population | Family | Chosen representation | CV MAE (V) | Holdout MAE (V) | Na MAE (V) | K MAE (V) |
|---|---|---|---:|---:|---:|---:|
| mixed_ions | svr | scaled | 0.6729 | 0.6973 | 0.9115 | 1.3500 |
| mixed_ions | krr | scaled | 0.7575 | 0.8126 | 0.9973 | 1.4287 |
| mixed_ions | dnn | scaled | 0.6401 | 0.6553 | 0.7431 | 1.2093 |

mixed_ions: CV-selected family = **dnn**. Median baseline holdout MAE = 1.4013 V.

| li_only | svr | scaled | 0.6879 | 0.7802 | 0.9587 | 1.2083 |
| li_only | krr | scaled | 0.7856 | 0.8602 | 1.0010 | 1.2907 |
| li_only | dnn | scaled | 0.6672 | 0.7872 | 0.9240 | 1.0039 |

li_only: CV-selected family = **dnn**. Median baseline holdout MAE = 0.9742 V.


See step6_report.json for RMSE, direct predictive R2, bias, per-ion metrics,
known/new-group Na/K transfer and priority-review/negative-voltage subgroups.

CV fold standard deviations describe fold variation, not prediction intervals.

## Limitations

- Current Materials Project adaptation, not an exact reproduction of the original paper.
- The feature scheme uses composition and symmetry; it has no atomic coordinates, ionic radii or electronegativity.
- The search is small and predeclared. Selected CV scores are model-selection statistics, not unbiased test estimates.
- All original eligible labels are retained, including the 14 previously flagged priority-review records. Source-energy verification is pending.
- Na/K come from the same retrieval and are not the original paper sodium test set or an independent experimental benchmark.
- Known-group versus new-group transfer is relative to the development partition used to fit that bundle.
- No fine-tuning on Na/K and no calibrated uncertainty estimation are performed.
- Li-only direct working-ion identity columns are constant during training and cannot teach a Na/K identity effect.
- Metrics are in volts; R2 is computed directly on predictions and can be negative. MAE is not percentage accuracy.
- Holdout/transfer results must not be used to choose new hyperparameters while still calling the same sets untouched tests.
