# Prony series identification for linear viscoelastic material models

Linear viscoelastic materials are often described with a Generalized Maxwell model. Herein, the experimental data need to be fitted to a Prony series expansion to determine the necessary model parameter. This python repository allows the Prony series parameter identificaiton from experimental data measured in either the frequency domain (via Dynamic Mechanical Thermal Analysis) or time domain (via relaxation measurements). A jupyter notebook is provided that allows for direct upload of the raw measurement data and the application of the time-temperature superposition principle to create a master curve for which the Prony series parameters are identified. Additionally, shift functions are fitted for the determined shift factors during creation of the master curve. Alternatively, prepared master curves can be uploaded and the shift procedure skipped to directly identify the Prony series pareameters for the provided master curve. An optional minimization routine is provided to minimize the number of Prony elements used in the Generalized Maxwell model. This routine is helpful for reducing the computational complexity of the linear viscoelastic material model when used in Finite Element simulations.

An interactive version of the notebook can be found here. 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/martin-springer/LinViscoFit/HEAD?urlpath=voila%2Frender%2FLinViscoFit.ipynb)

The Python implementation has been verified by comparison of the obtained results with the curve fitting routine provided by ANSYS APDL 2020 R1. Jupyter notebooks showcasing this verification can be found below.
[Example input files](examples) TBD!

