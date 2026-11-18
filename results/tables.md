## Development CV (dev split only)

### PFI

| Feature set | Cox C (mean ± SD, 15 folds) | Cox Δ vs clinical | NN C (mean ± SD) | NN Δ vs clinical |
|---|---|---|---|---|
| clinical | 0.682 ± 0.064 | ref | 0.614 ± 0.056 | ref |
| clinical_rna | 0.720 ± 0.082 | +0.039 ± 0.055 (10/15 folds > 0) | 0.654 ± 0.073 | +0.040 ± 0.068 (12/15 folds > 0) |
| clinical_text | 0.688 ± 0.058 | +0.006 ± 0.030 (9/15 folds > 0) | 0.628 ± 0.054 | +0.014 ± 0.032 (10/15 folds > 0) |
| text | 0.601 ± 0.050 | -0.081 ± 0.053 (0/15 folds > 0) | 0.583 ± 0.042 | -0.031 ± 0.053 (4/15 folds > 0) |
| clinical_rna_text | 0.713 ± 0.071 | +0.031 ± 0.047 (11/15 folds > 0) | 0.684 ± 0.063 | +0.070 ± 0.053 (13/15 folds > 0) |
| rna | 0.662 ± 0.080 | -0.020 ± 0.085 (5/15 folds > 0) | 0.613 ± 0.057 | -0.001 ± 0.056 (9/15 folds > 0) |

### OS

| Feature set | Cox C (mean ± SD, 15 folds) | Cox Δ vs clinical | NN C (mean ± SD) | NN Δ vs clinical |
|---|---|---|---|---|
| clinical | 0.737 ± 0.071 | ref | 0.729 ± 0.062 | ref |
| clinical_rna | 0.731 ± 0.082 | -0.005 ± 0.033 (6/15 folds > 0) | 0.701 ± 0.065 | -0.028 ± 0.050 (5/15 folds > 0) |
| clinical_text | 0.737 ± 0.063 | +0.001 ± 0.038 (8/15 folds > 0) | 0.699 ± 0.058 | -0.030 ± 0.043 (4/15 folds > 0) |
| text | 0.595 ± 0.055 | -0.142 ± 0.082 (1/15 folds > 0) | 0.581 ± 0.061 | -0.148 ± 0.092 (1/15 folds > 0) |
| clinical_rna_text | 0.739 ± 0.076 | +0.003 ± 0.044 (10/15 folds > 0) | 0.716 ± 0.072 | -0.013 ± 0.050 (6/15 folds > 0) |
| rna | 0.626 ± 0.092 | -0.110 ± 0.068 (1/15 folds > 0) | 0.579 ± 0.056 | -0.150 ± 0.074 (1/15 folds > 0) |

Site-only Cox (PFI, dev CV): C = 0.471 ± 0.055
Text PCs → tissue source site: accuracy 0.893 vs majority baseline 0.173 (13 sites)

## Test — PFI — cox (n=255, events=35)

| Feature set | Harrell C [95% CI] | Δ vs clinical [95% CI] | Uno C (τ=5y) |
|---|---|---|---|
| clinical | 0.736 [0.633, 0.834] | ref | 0.658 |
| clinical_rna | 0.733 [0.630, 0.818] | -0.003 [-0.083, 0.074] | 0.636 |
| clinical_text | 0.726 [0.632, 0.807] | -0.010 [-0.078, 0.065] | 0.657 |
| text | 0.601 [0.499, 0.699] | -0.135 [-0.282, 0.013] | 0.565 |
| clinical_rna_text | 0.733 [0.648, 0.807] | -0.003 [-0.084, 0.087] | 0.645 |
| rna | 0.613 [0.509, 0.705] | -0.123 [-0.256, 0.008] | 0.541 |

clinical_rna_text − clinical_rna: +0.000 [-0.040, 0.042]

## Test — PFI — nn (n=255, events=35)

| Feature set | Harrell C [95% CI] | Δ vs clinical [95% CI] | Uno C (τ=5y) |
|---|---|---|---|
| clinical | 0.634 [0.573, 0.691] | ref | 0.594 |
| clinical_rna | 0.644 [0.563, 0.720] | +0.010 [-0.061, 0.070] | 0.619 |
| clinical_text | 0.707 [0.630, 0.775] | +0.073 [0.001, 0.138] | 0.667 |
| text | 0.622 [0.519, 0.715] | -0.012 [-0.131, 0.098] | 0.574 |
| clinical_rna_text | 0.725 [0.646, 0.800] | +0.091 [0.021, 0.158] | 0.691 |
| rna | 0.587 [0.489, 0.685] | -0.047 [-0.161, 0.065] | 0.555 |

clinical_rna_text − clinical_rna: +0.081 [0.042, 0.125]

### Sensitivity — PFI — cox, flagged reports removed (n=214, events=31)

| Feature set | Harrell C [95% CI] | Δ vs clinical [95% CI] |
|---|---|---|
| clinical | 0.708 [0.583, 0.815] | ref |
| clinical_rna | 0.714 [0.602, 0.810] | +0.006 [-0.087, 0.096] |
| clinical_text | 0.704 [0.610, 0.790] | -0.004 [-0.078, 0.082] |
| text | 0.579 [0.471, 0.684] | -0.129 [-0.288, 0.045] |
| clinical_rna_text | 0.720 [0.633, 0.806] | +0.012 [-0.079, 0.117] |
| rna | 0.603 [0.480, 0.701] | -0.105 [-0.260, 0.040] |

clinical_rna_text − clinical_rna: +0.006 [-0.041, 0.057]

## Test — OS — cox (n=255, events=37)

| Feature set | Harrell C [95% CI] | Δ vs clinical [95% CI] | Uno C (τ=5y) |
|---|---|---|---|
| clinical | 0.756 [0.665, 0.840] | ref | 0.680 |
| clinical_rna | 0.750 [0.655, 0.842] | -0.006 [-0.059, 0.049] | 0.712 |
| clinical_text | 0.778 [0.687, 0.861] | +0.022 [-0.039, 0.084] | 0.723 |
| text | 0.615 [0.492, 0.737] | -0.141 [-0.295, 0.009] | 0.646 |
| clinical_rna_text | 0.772 [0.677, 0.859] | +0.016 [-0.060, 0.091] | 0.754 |
| rna | 0.630 [0.499, 0.757] | -0.126 [-0.285, 0.020] | 0.682 |

clinical_rna_text − clinical_rna: +0.022 [-0.027, 0.073]

## Test — OS — nn (n=255, events=37)

| Feature set | Harrell C [95% CI] | Δ vs clinical [95% CI] | Uno C (τ=5y) |
|---|---|---|---|
| clinical | 0.756 [0.673, 0.838] | ref | 0.689 |
| clinical_rna | 0.683 [0.591, 0.773] | -0.073 [-0.173, 0.018] | 0.683 |
| clinical_text | 0.764 [0.681, 0.841] | +0.008 [-0.052, 0.067] | 0.719 |
| text | 0.616 [0.519, 0.710] | -0.140 [-0.263, -0.016] | 0.631 |
| clinical_rna_text | 0.721 [0.637, 0.806] | -0.035 [-0.133, 0.055] | 0.721 |
| rna | 0.542 [0.480, 0.607] | -0.214 [-0.321, -0.117] | 0.567 |

clinical_rna_text − clinical_rna: +0.038 [0.004, 0.079]

### Sensitivity — OS — cox, flagged reports removed (n=214, events=32)

| Feature set | Harrell C [95% CI] | Δ vs clinical [95% CI] |
|---|---|---|
| clinical | 0.768 [0.659, 0.857] | ref |
| clinical_rna | 0.768 [0.671, 0.852] | -0.000 [-0.050, 0.056] |
| clinical_text | 0.787 [0.700, 0.867] | +0.019 [-0.039, 0.082] |
| text | 0.624 [0.514, 0.744] | -0.145 [-0.293, 0.021] |
| clinical_rna_text | 0.788 [0.708, 0.867] | +0.020 [-0.051, 0.099] |
| rna | 0.641 [0.503, 0.767] | -0.127 [-0.293, 0.037] |

clinical_rna_text − clinical_rna: +0.020 [-0.034, 0.083]

## Decision

```json
{
  "rule": "Text is reported as adding detectable held-out information only if, for the primary Cox family on PFI, the paired bootstrap 95% CI of C(clinical_text) - C(clinical) excludes 0 on the positive side. C(clinical_rna_text) - C(clinical_rna) is the key secondary contrast under the same rule. Otherwise report: no detectable gain in this cohort. Point estimates alone are not evidence.",
  "primary_contrast": {
    "name": "clinical_text - clinical",
    "estimate": -0.010067114093959773,
    "ci95": [
      -0.0776365599915941,
      0.06534644880865918
    ],
    "fraction_boot_gt_0": 0.3795
  },
  "key_secondary_contrast": {
    "name": "clinical_rna_text - clinical_rna",
    "estimate": 0.00023969319271333056,
    "ci95": [
      -0.040288912628347,
      0.04207055085776231
    ],
    "fraction_boot_gt_0": 0.5115
  },
  "text_adds_detectable_information": false,
  "text_adds_beyond_clinical_rna": false
}
```
