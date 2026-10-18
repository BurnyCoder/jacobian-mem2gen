| arm | no-patch (q) | oracle (q) | rescued of failed | rescuing cells / failed q | share s>t | mean dMRR (all / s>t) | CV single pair (held-out) | LRSD pair net flips |
|---|---|---|---|---|---|---|---|---|
| copy | 6.1% | 39.1% | 35.2% (384 failed) | 9.4 | 56.9% | +0.0194 / +0.0227 | 2.3% [(2, 5), (3, 10)] | +7 / -13 = -6 |
| jlens | 6.1% | 33.7% | 29.4% (384 failed) | 9.1 | 68.6% | +0.0152 / +0.0240 | 2.9% [(10, 3), (22, 10)] | +10 / -6 = +4 |
| noise_copy | 6.1% | 48.2% | 44.8% (384 failed) | 9.1 | 58.4% | +0.0156 / +0.0211 | 4.2% [(9, 3), (18, 0)] | +8 / -20 = -12 |
| noise_jlens | 6.1% | 44.0% | 40.4% (384 failed) | 7.2 | 66.8% | +0.0101 / +0.0154 | 3.9% [(9, 0), (10, 2)] | +7 / -10 = -3 |
| norm | 6.1% | 42.1% | 38.3% (384 failed) | 9.6 | 58.3% | +0.0175 / +0.0206 | 2.3% [(1, 10), (20, 1)] | +11 / -13 = -2 |
| jlens_pp | 6.1% | 15.2% | 9.6% (384 failed) | 2.7 | 87.0% | +0.0031 / +0.0046 | 1.6% [(21, 0), (20, 0)] | +2 / -3 = -1 |
| noise_jlens_pp | 6.1% | 15.9% | 10.4% (384 failed) | 2.1 | 75.6% | +0.0022 / +0.0029 | 1.3% [(23, 5), (20, 3)] | +2 / -1 = +1 |

| contrast (per failed question) | mean | 95% CI |
|---|---|---|
| copy_minus_noise_copy | -0.096 | [-0.135, -0.057] |
| jlens_minus_noise_jlens | -0.109 | [-0.151, -0.068] |
| jlens_pp_minus_noise_jlens_pp | -0.008 | [-0.034, +0.018] |
| mrr_gain_all_copy_minus_noise_copy | +0.004 | [+0.001, +0.007] |
| mrr_gain_back_copy_minus_noise_copy | +0.002 | [-0.005, +0.009] |
| mrr_gain_all_jlens_minus_noise_jlens | +0.005 | [+0.002, +0.008] |
| mrr_gain_back_jlens_minus_noise_jlens | +0.009 | [+0.002, +0.016] |
| mrr_gain_all_jlens_minus_copy | -0.004 | [-0.009, +0.000] |
| mrr_gain_back_jlens_minus_copy | +0.001 | [-0.005, +0.008] |
| mrr_gain_all_jlens_pp_minus_noise_jlens_pp | +0.001 | [-0.000, +0.002] |
| mrr_gain_back_jlens_pp_minus_noise_jlens_pp | +0.002 | [-0.001, +0.005] |
| mrr_gain_all_jlens_pp_minus_copy | -0.016 | [-0.024, -0.009] |
| mrr_gain_back_jlens_pp_minus_copy | -0.018 | [-0.027, -0.010] |
| jlens_minus_copy | -0.057 | [-0.091, -0.023] |
| mcnemar_copy_vs_jlens | only jlens 12, only copy 34 | p = 0.00164 |
| jlens_gain_minus_copy_gain | -0.013 | [-0.065, +0.039] |
| jlens_pp_minus_copy | -0.255 | [-0.299, -0.214] |
| mcnemar_copy_vs_jlens_pp | only jlens_pp 0, only copy 98 | p = 6.31e-30 |
| jlens_pp_gain_minus_copy_gain | +0.089 | [+0.042, +0.135] |
