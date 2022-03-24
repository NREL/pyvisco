# pyvisco

Pyvisco is a Python library that supports the identification of Prony series parameters in Generalized Maxwell models describing linear viscoelastic materials. 

> Note: This repository is under development!

## Overview
The mechanical response of linear viscoelastic materials is often described with Generalized Maxwell models. The necessary material model parameters are typically identified by fitting a Prony series to the experimental measurement data in either the frequency-domain (via Dynamic Mechanical Thermal Analysis) or time-domain (via relaxation measurements). Pyvisco performs the necessary data processing of the experimental measurements, mathematical operations, and curve-fitting routines to identify the Prony series parameters. The experimental data can be provided as raw measurement sets at different temperatures or as pre-processed master curves.

* If raw measurement data are provided, the time-temperature superposition principle is applied to create a master curve and obtain the shift functions prior to the Prony series parameters identification. 

* If master curves are provided, the shift procedure can be skipped, and the Prony series parameters identified directly. 

An optional minimization routine is provided to reduce the number of Prony elements. This routine is helpful for Finite Element simulations where reducing the computational complexity of the linear viscoelastic material models can shorten the simulation time.

## Usage
The easiest way of using pyvisco is through an interactive Jupyter notebook that provides a graphical user interface to upload the experimental data, perform the curve fitting procedure, and download the obtained Prony series parameters. Currently, the Jupyter notebook is rendered with voila and can be accessed either through binder or Heroku. Click one of the below links to start the web application.

[Heroku](https://pyvisco.herokuapp.com/)  
[Binder](https://mybinder.org/v2/gh/NREL/pyvisco/HEAD?urlpath=voila%2Frender%2FLinViscoFit.ipynb)  



Alternatively, the library can be installed as (PyPi installation coming soon...):
1. Clone or download the pyvisco repository.
2. Navigate to repository: `cd pyvisco`
3. Install via pip: `pip install .`

A full API documentation is available at: [![Documentation Status](https://readthedocs.org/projects/pyvisco/badge/?version=latest)](https://pyvisco.readthedocs.io/en/latest/?badge=latest)   
Additionally, the [verification subfolder](https://github.com/NREL/pyvisco/tree/main/verification) contains example Jupyter notebooks on how to use the library.

## Verification
The Python implementation was verified by comparing the obtained Prony series parmaters with the curve fitting routine implemented in the commercial software package ANSYS APDL 2020 R1. Jupyter notebooks showcasing the comparision and supplementary files can be found in the [verification subfolder](https://github.com/NREL/pyvisco/tree/main/verification).

## Citing
If you are using pyvisco in your published work, please cite the DOI corresponding to the specific version that you are using. Pyvisco DOIs are listed at Zenodo.org (TBD).
