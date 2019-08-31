# dataDrivenRamen
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
 

## Chefs

- Daril Brown
- Shiva Farashahi
- Emily Gelfand
- Roman Levin
- Courtnie Paschall

## Recipe

```
├── README.md          <- The top-level README for developers using this project.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── scripts            <- The requirements file for reproducing the analysis environment
│
├── daRamen            <- Source code for use in this project.
│   ├── __init__.py    <- Makes daRamen a Python module
│   │
│   ├── data           <- Functions to download or generate data
│   │   └── make_dataset.py (Example)
│   │
│   ├── features       <- Functions to turn raw data into features for modeling
│   │   └── build_features.py (Example)
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
└── requirements.txt   <- The requirements file for reproducing the analysis environment
```
## How to Prepare Ramen

1. Code & Style
    * All code should be written in Python 3.4+
    * Code should follow the [PEP8](https://www.python.org/dev/peps/pep-0008/) style guide

2. Documentation
    * Docstrings for public functions should be in
[Numpy docs](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt) format.
At minimum, there should be a sentence describing what the function does and a list of
parameters and returns.
    * Private functions should be indicated with a leading underscore, and should still include a
docstrings including at least a sentence describition what the function does.
    * If you add any new public functions, note this function in the doc/api.rst file,
so that this function gets included in the documentation site API listing.

3. Dependencies
    * Any dependencies outside of the standard Anaconda distribution should be avoided if possible.
    * If any more packages are needed, they should be added to the `requirements.txt` file.

4. API & Naming Conventions
    * Try to keep the API consistent across daRamen in naming and parameter ordering, for example:
        * `convention_1` [Insert Convention Description Here]
        * `convention_2` [Insert Convention Description Here]
    * Try to keep naming conventions consistent with other modules, for example:
        * Function names are in all lowercase with underscores

5. Tests
    * All code within daRamen should require a test code that executes that code when it is ready to be commited
    * These tests, at a minimum, must be 'smoke tests' that execute the
code and check that it runs through, without erroring out, and returning appropriate variables.
    * If possible, including more explicit test code that checks more stringently for accuracy is encouraged,
but not strictly required.


