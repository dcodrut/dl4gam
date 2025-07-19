## DL4GAM: a multi-modal Deep Learning-based framework for Glacier Area Monitoring

DL4GAM was developed to monitor glacier area changes using Deep Learning models trained on multi-modal data — currently including optical imagery, DEMs, and elevation change data.

The framework is glacier-centric, meaning that data is processed and analyzed on a glacier-by-glacier basis. 
This allows us to download the best images independently for each glacier, minimizing cloud/shadow coverage and seasonal snow as much as possible. 
This approach also makes the parallelization straightforward.

This repository extends the codebase of [DL4GAM-Alps](https://github.com/dcodrut/dl4gam_alps) by making it more modular and easier to apply to other regions or setups.