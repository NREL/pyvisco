# Prony series identification for linear viscoelastic material models

> Note: This repository is currently in development!

## Overview
Linear viscoelastic materials are often described with a Generalized Maxwell model. The necessary model parameters are identified by fitting a Prony series to the experimental measurement data. 

This Python repository allows for the identification of Prony series parameters from experimental data measured in either the frequency-domain (via Dynamic Mechanical Thermal Analysis) or time-domain (via relaxation measurements). The experimental data can be provided as raw measurement sets at different temperatures or as pre-processed master curves.

* If raw measurement data are provided, the time-temperature superposition principle is applied to create a master curve and obtain the shift functions prior to the Prony series parameters identification. 

* If master curves are provided, the shift procedure can be skipped, and the Prony series parameters can be directly identified. 

An optional minimization routine is provided to reduce the number of Prony elements. This routine is helpful for Finite Element simulations where reducing the computational complexity of the linear viscoelastic material models can shorten the simulation time.

## Usage
The easiest way of using this code is through an interactive Jupyter notebook that provides a graphical user interface to upload the experimental data, perform the curve fitting procedure, and download the obtained Prony series parameters. Click the link below to start the web application.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/martin-springer/LinViscoFit/HEAD?urlpath=voila%2Frender%2FLinViscoFit.ipynb)

Alternatively, all required functions are provided in `LinViscoFit.py`. Download the file, copy it into your working directory, and import the module in your Python code (`import LinViscoFit as visco`). The verification notebooks provided below showcase the usage of the module functions to perform the curve fitting procedure.

## Verification
The Python implementation has been verified by comparing the obtained results with the curve fitting routine implemented in the commercial software package ANSYS APDL 2020 R1. Jupyter notebooks showcasing this verification can be found here. [Verification notebooks](verification) TBD!

## Acknowledgements
TBD...
