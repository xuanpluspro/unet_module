# Learning-rate settings

`python train_medical.py --lr 5e-5 ...` now passes the requested peak learning rate to Adam and the existing warmup/cosine scheduler. Previously this option was ignored and the peak was clamped to 1e-4 for every batch size.

The default peak remains 1e-4. The minimum remains 1% of the peak (1e-6 by default). The existing scheduler still starts with a short warmup before cosine decay; --lr denotes the peak, not necessarily the first epoch's value. Its existing warmup floor of 1e-6 is retained.

Batch-size changes do not automatically scale learning rate. Choose --lr explicitly for a new experiment. Default LR curves remain unchanged. Invalid nonpositive/nonfinite learning rates fail before optimizer construction.

Verification: dependency-light regression tests compare the full default curve, custom-LR propagation, batch-size independence and invalid inputs. No GPU training or accuracy verification was performed.
