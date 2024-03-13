# Description
This repo includes the code for Gaussian Blur algorithm running on Hammerblade 1 pod.

# File Description
- Baseline implementation:
    - `kernel0-baseline.cpp`
    - `main0_baseline.c`
- Baseline with marginal improvement
    - `kernel0-baseline_mlp.cpp`
    - `main0_baseline.c`
- Optimization #1 padding:
    - `kernel1-Opt1_padding.cpp`
    - `main1_2_3_opt1_2_3.c`
- Optimization #2 mlp:
    - `main1_2_3_opt1_2_3.c`
    - `kernel2-Opt2_mlp.cpp`
- Optimization #3 warm cache:
    - `kernel3-Opt3_cache_1.cpp`
    - `main1_2_3_opt1_2_3.c`
- Optimization #3-2 warm cache with filter preload:
    - `kernel3-Opt3_cache2_filter_a_copy.cpp`
    - `main3_opt3_cache_a_copy.c`
