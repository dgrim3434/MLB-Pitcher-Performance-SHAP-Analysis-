# Week 15 Capstone: Explainable AI with SHAP

## Project Overview

This notebook is part of **Week 15 of my 52-week data science training plan**, focused on **explainable AI with SHAP**.

The goal of this project is to predict whether an MLB pitcher will produce an above-average next-season **FIP-** using prior-season pitching statistics, then use explainability tools to understand *why* the model makes its predictions.

Rather than treating the model as a black box, this project focuses on interpreting feature impact through SHAP summary plots, SHAP force plots, Partial Dependence Plots, ICE plots, and permutation importance.

## Business Problem

Front offices need ways to identify pitchers who are likely to perform well in the following season. A predictive model can help narrow the list of pitchers worth deeper scouting review, especially players who may not stand out from prior-season FIP- alone.

The target is whether a pitcher records a next-season **FIP- below 100**. Since FIP- is scaled so that 100 is league average, a value below 100 indicates above-average fielding-independent pitching performance.

## Data Source

All data used in this notebook was extracted from **FanGraphs**.

The notebook loads standard and plus pitcher statistics from 2018 through 2025 using files named:

```text
standard_pitcher_stats_{year}.csv
plus_pitcher_stats_{year}.csv
```

## Model Summary

The model uses prior-season pitcher statistics to predict whether a pitcher will have a next-season FIP- below 100.

The final XGBoost model used:

* `ERA`
* `K/9+`
* `BB/9+`
* `HR/9+`
* `GB%+`
* `LOB%+`

`prior_season_FIP-` was used as the naive baseline, then removed from the model so the XGBoost classifier would not simply recreate the benchmark. `BABIP+` was also removed after SHAP showed it had almost no meaningful impact.

## Results

The naive baseline using prior-season FIP- achieved:

| Model                            |   AUC |
| -------------------------------- | ----: |
| Naive prior-season FIP- baseline | 0.672 |
| Final XGBoost model              | 0.698 |

Final test performance:

| Metric            | Value |
| ----------------- | ----: |
| ROC AUC           | 0.698 |
| Class 0 Precision |  0.55 |
| Class 0 Recall    |  0.64 |
| Class 1 Precision |  0.71 |
| Class 1 Recall    |  0.63 |
| Class 0 F1        |  0.59 |
| Class 1 F1        |  0.66 |

The model slightly outperformed the baseline, but the improvement was modest. The strongest value of the notebook is the explainability analysis, not raw predictive performance.

## Key Findings

`K/9+` was the strongest driver of model predictions. Higher strikeout ability generally increased the probability of above-average next-season performance.

`HR/9+` showed signs of regression to the mean. Extreme home run rates did not always create the worst model impact, suggesting the model learned that some home-run spikes are unstable year to year.

`GB%+` provided useful signal, but its interpretation requires caution because ground-ball rate is connected to pitcher type and correlated with other traits.

`BABIP+` added little value and was removed after SHAP inspection.

`BB/9+` appeared more important in permutation importance than in average SHAP importance, suggesting walk rate may matter heavily for specific edge-case pitchers even if its average impact looks smaller.

## Explainability Methods Used

This notebook applies several interpretability techniques:

* **SHAP summary plots** to identify global feature importance
* **SHAP force plots** to inspect individual predictions
* **Partial Dependence Plots** to understand average feature effects
* **ICE plots** to inspect individual-level feature behavior
* **Permutation importance** to compare against SHAP-based importance

## Final Conclusion

This Week 15 capstone successfully applies explainable AI techniques to a real baseball modeling problem.

The XGBoost model slightly improves over the prior-season FIP- baseline, but the bigger value comes from the explainability layer. SHAP, PDP, ICE, force plots, and permutation importance expose how the model thinks, which features drive predictions, and where the model may be misleading.

The honest conclusion is that this model is not a standalone decision system. It is a useful analytical filter that can support scouting and player evaluation when paired with deeper baseball context.
::: 
