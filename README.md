# Forest Light Environmental Simulator (FLiES) Radiative Transfer Model Look-Up Table (LUT) Implementation in Python

[![CI](https://github.com/gregory-halverson-jpl/FLiESLUT/actions/workflows/ci.yml/badge.svg)](https://github.com/gregory-halverson-jpl/FLiESLUT/actions/workflows/ci.yml)

This package is a look-up table emulator for the Forest Light Environmental Simulator (FLiES) model in Python. This model is used to estimate solar radiation for the Breathing Earth Systems Simulator (BESS) model used to estimate evapotranspiration (ET) and gross primary productivity (GPP) for the ECOsystem Spaceborne Thermal Radiometer Experiment on Space Station (ECOSTRESS) and Surface Biology and Geology (SBG) thermal remote sensing missions.

## Contributors

[Gregory H. Halverson](https://github.com/gregory-halverson-jpl) (they/them)<br>
[gregory.h.halverson@jpl.nasa.gov](mailto:gregory.h.halverson@jpl.nasa.gov)<br>
Lead developer<br>
NASA Jet Propulsion Laboratory 329G

Hideki Kobayashi (he/him)<br>
FLiES algorithm inventor<br>
Japan Agency for Marine-Earth Science and Technology

## Installation

The `FLiESLUT` package is available as a [pip package](https://pypi.org/project/FLiESLUT/) from PyPi:

```
pip install FLiESLUT
```

## Usage

Import the `FLiESLUT` package as `FLiESLUT`:

```
import FLiESLUT
```

## References

If you use the **Forest Light Environmental Simulator (FLiES)** model in your work, please cite the following references:

1. Kobayashi, H., & Iwabuchi, H. (2008). *A coupled 1-D atmospheric and 3-D canopy radiative transfer model for canopy reflectance, light environment, and photosynthesis simulation in a heterogeneous landscape*. **Remote Sensing of Environment**, 112(1), 173-185.  
   [https://doi.org/10.1016/j.rse.2007.04.010](https://doi.org/10.1016/j.rse.2007.04.010)

2. Kobayashi, H., Ryu, Y., & Baldocchi, D. D. (2012). *A framework for estimating vertical profiles of canopy reflectance, light environment, and photosynthesis in discontinuous canopies*. **Agricultural and Forest Meteorology**, 150(5), 601-619.  
   [https://doi.org/10.1016/j.agrformet.2010.12.001](https://doi.org/10.1016/j.agrformet.2010.12.001)
