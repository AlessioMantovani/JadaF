# JadaF - Just Another Data Analysis Framework

![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)


**JadaF** is a Python package designed to simplify data analysis workflows. 

## Installation

### Dependencies

JadaF requires the following Python packages:

- **plotly express**: For creating interactive visualizations.
- **scikit-learn**: For machine learning algorithms and data preprocessing tools.
- **polars**: A fast DataFrame library.
- **pandas**: For handling structured data and manipulation.
- **numpy**: For handling vectors, matrices and other math related data structures

#### Installation for developement

To install the dependencies, you need to download the repo and use requirements.txt (it's recommended to use a separate env if developing):

```bash
git clone https://github.com/yourusername/JadaF.git@dev
cd JadaF
pip install -r requirements.txt
```

Then to install the library locally in dev mode you can run:

```bash
pip install -e .
```
This will install the library in editable mode, which means that any changes made to the source code will be reflected without needing to reinstall the package.

---

#### Normal installation

If you don't want to develop, and just use the library you can install JadaF directly from GitHub:

```bash
pip install git+https://github.com/AlessioMantovani/jada.git@main
```

## Docs

You can easly read from files and create a jadaf Dataframe with:
```
import jadaf as jd

df = jd.load_csv("path_to_your_csv_file")
df = jd.load_json("path_to_your_json_file")
df = jd.load_excell("path_to_your_excell_file")
```

If you already have a loaded polars dataframe you can also wrap it around JDF:
```
import jadaf as jd

pl_dataframe = pl.Dataframe()
jd_dataframe = JDF(pl_dataframe)
```
This will keep basic functionalities like:
- Slicing 
- Shape 
- Columns names
- dtypes
- Head and Tails
```
import jadaf as jd

jd_dataframe[:2]
jd_dataframe[:2, :2]
jd_dataframe.shape
jd_dataframe.dtype
jd_dataframe.columns
jd_dataframe.head(5)
jd_dataframe.tail(10)
```
