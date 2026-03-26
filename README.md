# Search for Supermassive Binary Black Holes Systems: A brief presentation of the improved methodology

This work is based on <a href="#ref1">[1]</a> as part of my bachelor's thesis. There, singular spectrum analysis (SSA) is employed for the first time on data from the Fermi-LAT Space Telescope to study periodicities on the light curves of several blazars. The aim of using SSA is to isolate the periodic behaviour of the emissions from long-term trends and noise, and then compute the Lomb-Scargle periodogram (LSP) on the isolated oscillatory component to determine the most significant periods.
As a result of this multi-stage pipeline, 46 blazars have been identified as high-priority candidates for Quasi-Periodic Oscillations (QPOs). these findings provide a robust foundation for future investigations into the detection of Binary Supermassive Black Hole (SMBH) systems. To strengthen the evidence for these candidates, our methodology introduced several analytical advancements:

- Innovative Time Series Analysis: A refined approach to component identification using Singular Spectrum Analysis (SSA) to isolate trends and preserve high-frequency information.
- Fast Template Periodogram (FTP) Implementation: The first application of the FTP framework for blazar periodicity, allowing for the detection of non-sinusoidal morphologies (e.g., Gaussian and empirical archetypes).
- Machine Learning Validation: A rigorous periodicity confirmation step using a Random Forest classifier trained on a synthetic universe of $5 \times 10^4$ simulated light curves.

The implementation of this study is structured into four primary functional modules:

1. Statistical Validation (blazar_simulation_fap.py): Processing of raw light curves and calibration of False Alarm Probabilities (FAP) via Monte Carlo simulations.
2. Synthetic Data Generation (ML_simulations.py): Execution of large-scale simulations to generate the training set for the classifier.
3. Model Architecture (ML_build.py): Construction, hyperparameter tuning, and training of the Random Forest model.
4. Results Synthesis (ML_analysis.py): Evaluation of model performance and final classification of the blazar candidates.

A comprehensive explanation of the entire analysis is provided in the accompanying Jupyter Notebook.

## References

<a name="ref1">[1]</a> A. Rico et al., «Singular spectrum analysis of Fermi-LAT blazar light curves: A systematic search for periodicity and trends in the time domain», A&A, vol. 697, p. A35, may 2025, doi: 10.1051/0004-6361/202452495.

<a name="ref1">[2]</a> J. Hoffman, J. Vanderplas, J. Hartman, y G. Bakos, «A Fast Template Periodogram for Detecting Non-sinusoidal Fixed-shape Signals in Irregularly Sampled Time Series», 7 de febrero de 2021, arXiv: arXiv:2101.12348. doi: 10.48550/arXiv.2101.12348.


```python

```
