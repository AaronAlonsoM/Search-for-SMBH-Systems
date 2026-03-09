# Search for Supermassive Binary Black Holes Systems: A brief presentation of the improved methodology using Machine Learning

This work is based on <a href="#ref1">[1]</a> as part of my bachelor's thesis. There, singular spectrum analysis (SSA) is employed for the first time on data from the Fermi-LAT Space Telescope to study periodicities on the light curves of several blazars. The aim of using SSA is to isolate the periodic behaviour of the emissions from long-term trends and noise, and then compute the Lomb-Scargle periodogram (LSP) on the isolated oscillatory component to determine the most significant periods.
As a result, 46 blazars are identified as potential candidates for quasi-periodic oscillations (QPOs), which provides a foundation for future investigations on the detection of binary supermassive black hole (SMBHs) systems. 

Here, some alternatives are presented to try to get more evidence on the previous results. This includes:

- Time series analysis to identify the relevant components using a new approach.
- First implementation of Fast Template Periodogram (FTP) <a href="#ref1">[2]</a> for blazar periodicity detection, using the most promising expermental data to confirm the new possible detections.
- Periodicity detection using a Machine Learning model trained with simulated light curves (LCs), obtaining an improvement of 11% with respect the most recent results on periodicity detection against noise and a great reduction in computational work.

An applied example of the method and the model is presented as a notebook.

## References

<a name="ref1">[1]</a> A. Rico et al., «Singular spectrum analysis of Fermi-LAT blazar light curves: A systematic search for periodicity and trends in the time domain», A&A, vol. 697, p. A35, may 2025, doi: 10.1051/0004-6361/202452495.

<a name="ref1">[2]</a> J. Hoffman, J. Vanderplas, J. Hartman, y G. Bakos, «A Fast Template Periodogram for Detecting Non-sinusoidal Fixed-shape Signals in Irregularly Sampled Time Series», 7 de febrero de 2021, arXiv: arXiv:2101.12348. doi: 10.48550/arXiv.2101.12348.

```python

```
