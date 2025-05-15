# JadaF - Just Another Data Analysis Framework

![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Version](https://img.shields.io/badge/version-0.1.0-green)

**JadaF** is a Python package designed to simplify data analysis workflows by providing an enhanced DataFrame implementation with intuitive slicing, pattern matching, and data profiling capabilities.

## Features

- **Enhanced DataFrame Operations** - Intuitive indexing, slicing, and data manipulation
- **Flexible Data Loading** - Easy loading from CSV, JSON, and Excel files
- **Pattern Matching** - Select columns using wildcard patterns
- **Data Profiling** - Quickly generate comprehensive data profiles
- **Integrated Scaling** - Built-in scaling utilities powered by scikit-learn

## Installation

### Dependencies

JadaF requires the following Python packages:

- **polars**: Fast DataFrame library as the core engine
- **pandas**: For structured data manipulation and Excel support
- **scikit-learn**: For machine learning algorithms and data preprocessing
- **numpy**: For numerical operations

### Development Installation

To install for development (recommended if you want to contribute):

```bash
git clone https://github.com/yourusername/JadaF.git
cd JadaF
pip install -r requirements.txt
pip install -e .
```

This installs the library in editable mode, allowing code changes to be immediately reflected without reinstallation.

### User Installation

For regular usage, install directly from GitHub:

```bash
pip install git+https://github.com/AlessioMantovani/jada.git@main
```

## Documentation

### Importing the Library

```python
import jadaf as jd
from jadaf import JDF  # The core DataFrame class
```

### Loading Data

JadaF provides convenient functions to load data from various file formats:

```python
# Load from CSV
df = jd.load_csv("path/to/data.csv", delimiter=",", has_header=True)

# Load from JSON
df = jd.load_json("path/to/data.json", orient="records")

# Load from Excel
df = jd.load_excel("path/to/data.xlsx", sheet_name=0, has_header=True)
```

### Creating a JDF from Existing DataFrames

You can wrap existing Polars or Pandas DataFrames:

```python
import polars as pl
import pandas as pd

# From Polars DataFrame
pl_df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
jdf = JDF(pl_df)

# From Pandas DataFrame (automatically converted to Polars)
pandas_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
jdf = JDF(pandas_df)
```

### Basic Properties and Methods

```python
# Basic properties
jdf.shape        # Dimensions (rows, columns)
jdf.columns      # List of column names
jdf.dtypes       # Dictionary mapping column names to data types

# Preview data
jdf.head(5)      # First 5 rows
jdf.tail(5)      # Last 5 rows
```

### Advanced Indexing and Slicing

JadaF provides an enhanced, intuitive syntax for accessing data:

```python
# Single column access
col_data = jdf["column_name"]

# Multiple columns with comma-separated string
subset = jdf["col1, col2, col3"]

# Multiple columns with list
subset = jdf[["col1", "col2"]]

# Row slicing
first_10_rows = jdf[:10]

# Row and column slicing
subset = jdf[:10, ["col1", "col2"]]
subset = jdf[:10, "col1, col2"]

# Column range by name
subset = jdf["start_col":"end_col"]  # Inclusive of end_col

# Using wildcard patterns for column selection
date_columns = jdf["date_*"]  # All columns starting with "date_"
id_columns = jdf["*_id"]      # All columns ending with "_id"
```

### Filtering and Conditions

```python
# Filter with lambda function
adults = jdf[lambda df: df["age"] > 18]

# Filter with boolean series
high_income = jdf[jdf["income"] > 100000]

# Using the loc method
seniors = jdf.loc(lambda df: df["age"] > 65, ["name", "age", "income"])
```

### Column Pattern Matching

```python
# Get all columns matching a pattern
date_cols = jdf.columns_like("*date*")
```

### Column Grouping

```python
# Create a named group of columns
jdf.group_columns("demographics", ["age", "gender", "location"])

# Access the group (returns a new JDF with only those columns)
demo_data = jdf.demographics
```

### Data Analysis Utilities

```python
# Count distinct values with percentages
class_counts = jdf.count_classes(["gender", "occupation"])

# Check for missing values
missing_report = jdf.missing()

# Generate a comprehensive profile
profile = jdf.profile()
profile = jdf.profile(subset=["age", "income", "education"])
```

### Data Export

```python
# Convert to dictionary
records = jdf.to_dict(orient="records")

# Export to JSON
json_str = jdf.to_json()
jdf.to_json("output.json")

# Export to CSV
csv_str = jdf.to_csv()
jdf.to_csv("output.csv", sep=",", header=True)
```

### Using the Scaler Utility

JadaF provides a convenient wrapper for scikit-learn's scaling utilities:

```python
from jadaf.transform.scalers import Scaler, ScalerType

# Create a scaler (default: StandardScaler)
scaler = Scaler()

# Or specify the type
robust_scaler = Scaler(ScalerType.ROBUST)
minmax_scaler = Scaler("m")  # Short form for MinMaxScaler

# Fit and transform
X_scaled = scaler.fit_transform(jdf[["height", "weight", "age"]])

# Transform new data
X_new_scaled = scaler.transform(new_data)

# Inverse transform
X_original = scaler.inverse_transform(X_scaled)
```

## Examples

### Basic Data Exploration

```python
import jadaf as jd

# Load data
df = jd.load_csv("customer_data.csv")

# Quick overview
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns}")

# Check missing values
missing = df.missing()
print("Columns with missing values:")
print(missing.loc(lambda df: df["missing_count"] > 0))

# Explore categorical columns
cat_cols = ["gender", "customer_type", "membership_level"]
for col in cat_cols:
    print(f"\nDistribution of {col}:")
    print(df.count_classes([col]))
```

### Data Preprocessing and Filtering

```python
import jadaf as jd
from jadaf.transform.scalers import Scaler

# Load data
df = jd.load_csv("sales_data.csv")

# Group columns for easier access
df.group_columns("dates", ["order_date", "ship_date", "delivery_date"])
df.group_columns("amounts", ["order_amount", "tax", "shipping_cost", "total"])

# Filter active high-value customers
active_customers = df.loc(
    lambda df: (df["status"] == "active") & (df["lifetime_value"] > 1000),
    ["customer_id", "name", "email", "lifetime_value"]
)

# Scale numerical features
numeric_cols = ["order_amount", "frequency", "recency", "lifetime_value"]
scaler = Scaler("s")  # StandardScaler
scaled_data = scaler.fit_transform(df[numeric_cols])
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
