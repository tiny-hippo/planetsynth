## planetsynth

planetsynth is a python package to generate cooling tracks for giant planets.

### Installation
Installation is simple: download or clone this repository, navigate into the directory and execute
```
pip install .
```
which is the preferred way to install planetsynth. Alternatively, you can also run
```
python setup.py install
```
Note that the package requires numpy and scipy to function.

### Usage
```python
from planetsynth import PlanetSynth

logt = 9  # log(planetary age) in yrs
M = 0.4  # mass in Jupiter masses
Z = 0.2  # bulk heavy-element content (mass fraction)
Ze = 0.02  # atmospheric metallicity (mass fraction)
logF = 5  # log(incident stellar irradiation) in erg/s/cm2
planet_params = [M, Z, Ze, logF]

pls = PlanetSynth()
pred = pls.predict(logt, planet_params)
```
See examples/example.ipynb for more examples and in-depth explanations.

### Reference
https://ui.adsabs.harvard.edu/abs/2021MNRAS.507.2094M/abstract
