# Installation

This program is written in Python and requires a number of external libraries. Dependencies can be installed either
using [conda](https://docs.conda.io/en/latest/) or [venv](https://docs.python.org/3/library/venv.html).

### Installation using Conda

```console
conda create -n mgraph python=3.8 -y
conda activate mgraph
conda install --file requirements_conda.txt -y
python -m pip install -r requirements.txt
```

### Installation using Venv

```console
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

If not using `conda` you may encounter problems with the GDAL library installation. It requires the gdal binary to be
installed on the system and to the corresponding version of the Python library.

On Debian based Linux distros (Ubuntu) you may use:

```console
apt install -y gdal-bin libgdal-dev
pip install gdal==`gdal-config --version`
```

For further info please refer
to [https://mothergeo-py.readthedocs.io/en/latest/development/how-to/gdal-ubuntu-pkg.html](https://mothergeo-py.readthedocs.io/en/latest/development/how-to/gdal-ubuntu-pkg.html)
for help.
