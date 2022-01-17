# Prony series identification for linear viscoelastic material models

> Note: This repository is currently in development!

## Overview
Linear viscoelastic materials are often described with a Generalized Maxwell model. The model parameter are typically identified by fitting a Prony series to the experimental data. This python repository allows the Prony series parameter identificaiton from experimental data measured in either the frequency domain (via Dynamic Mechanical Thermal Analysis) or time domain (via relaxation measurements). 

## Usage
The easiest way of using this code is through an interactive Jupyter notebook that provides a graphical user interface to upload the experimental data, perform the curve fitting process, and download the obtained Prony series parameters. Click the link below to start the web application.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/martin-springer/LinViscoFit/HEAD?urlpath=voila%2Frender%2FLinViscoFit.ipynb)

Alternatively, all required functions are provided in `LinViscoFit.py`. Download the file, copy it into your working directory, and import the module in your Python code (`import LinViscoFit as visco`{.python}). The verification notebooks provided below showcase the usage of the module functions to perform the curve fitting procedure.

## Functionality
A jupyter notebook is provided that allows for direct processing of the raw measurement data and the application of the time-temperature superposition principle to create a master curve for which the Prony series parameters are identified. Additionally, shift functions are fitted for the determined shift factors during creation of the master curve. Alternatively, prepared master curves can be uploaded and the shift procedure skipped to directly identify the Prony series pareameters for the provided master curve. An optional minimization routine is provided to minimize the number of Prony elements used in the Generalized Maxwell model. This routine is helpful for reducing the computational complexity of the linear viscoelastic material model when used in Finite Element simulations.


## Verification
The Python implementation has been verified by comparison of the obtained results with the curve fitting routine provided by ANSYS APDL 2020 R1. Jupyter notebooks showcasing this verification can be found here. [Verification notebooks](verification) TBD!

## Acknowledgements
TBD...
