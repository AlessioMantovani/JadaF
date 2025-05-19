# JadaF - Just Another Data Analysis Framework

![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Version](https://img.shields.io/badge/version-0.1.0-green)
![Polars](https://img.shields.io/badge/powered%20by-Polars-orange)

**JadaF** is a modern Python library that provides an enhanced DataFrame implementation built on top of Polars. It offers intuitive indexing, powerful pattern matching, SQL-like queries, and comprehensive data profiling capabilities, making data analysis workflows more efficient and enjoyable.

## 🚀 Key Features

- **🔍 Enhanced DataFrame Operations** - Intuitive indexing, slicing, and data manipulation with Pythonic syntax
- **📁 Flexible Data Loading** - Seamless loading from CSV, JSON, and Excel files with smart type inference
- **🔎 Pattern Matching** - Select columns using wildcard patterns (`*`, `?`) for efficient column selection
- **📊 Data Profiling** - Generate comprehensive statistical profiles with missing value analysis
- **🔧 Integrated Scaling** - Built-in scaling utilities powered by scikit-learn
- **🛠️ SQL-like Queries** - Filter data using familiar SQL syntax with the `query()` method
- **⚡ High Performance** - Leverages Polars' lightning-fast in-memory processing
- **🎯 Column Grouping** - Organize related columns into named groups for easier access

## 📦 Installation

### Prerequisites

JadaF requires Python 3.8+ and the following dependencies:

- **polars** ≥ 0.18.0 - Fast DataFrame library as the core engine
- **pandas** ≥ 1.3.0 - For structured data manipulation and Excel support
- **scikit-learn** ≥ 1.0.0 - For machine learning algorithms and data preprocessing
- **numpy** ≥ 1.21.0 - For numerical operations

### Development Installation

For contributors and developers:

```bash
git clone https://github.com/yourusername/JadaF.git
cd JadaF
pip install -r requirements.txt
pip install -e .
```

This installs the library in editable mode, allowing code changes to be immediately reflected.

### User Installation

For end users:

```bash
pip install git+https://github.com/AlessioMantovani/jada.git@main
```

Or if published to PyPI:

```bash
pip install jadaf
```

## 🏃‍♂️ Quick Start

```python
import jadaf as jd

# Load your data
df = jd.load_csv("sales_data.csv")

# Quick data exploration
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns}")

# Filter data with SQL-like syntax
high_value_sales = df.query("amount > 1000 and status == 'completed'")

# Select columns using patterns
date_columns = df["*_date"]

# Generate a comprehensive profile
profile = df.profile()
```

## 📚 Documentation

### Importing and Basic Usage

```python
import jadaf as jd
from jadaf import JDF  # The core DataFrame class

# Create from existing data
import polars as pl
df = JDF(pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
```

### 📂 Data Loading

JadaF provides convenient functions for loading data from various sources:

```python
# CSV files with customizable options
df = jd.load_csv("data.csv", delimiter=",", has_header=True)

# JSON files with orient support
df = jd.load_json("data.json", orient="records")

# Excel files with sheet selection
df = jd.load_excel("data.xlsx", sheet_name="sales", has_header=True)
```

### 🎯 Advanced Indexing and Slicing

JadaF's indexing system supports multiple paradigms for maximum flexibility:

#### Single Column Access
```python
# Single column as Series
prices = df["price"]

# Multiple columns as comma-separated string
subset = df["name, price, quantity"]

# Multiple columns as list
subset = df[["name", "price", "quantity"]]
```

#### Pattern-Based Column Selection
```python
# Wildcard patterns
date_columns = df["*_date"]        # All columns ending with "_date"
id_columns = df["id_*"]           # All columns starting with "id_"
temp_columns = df["temp_*_avg"]   # Complex patterns

# Get matching column names
date_cols = df.columns_like("*date*")
```

#### Row Selection and Filtering
```python
# Row slicing by position
first_100 = df[:100]
last_50 = df[-50:]

# Boolean filtering
adults = df[df["age"] > 18]
high_earners = df[lambda df: df["income"] > 100000]

# Combined row and column selection
subset = df[:100, ["name", "age"]]
subset = df[df["active"] == True, "name, email, phone"]
```

#### Column Range Selection
```python
# Select columns from 'start' to 'end' (inclusive)
range_cols = df["col_a":"col_z"]

# With functions
numeric_cols = df[lambda cols: [c for c in cols if c.startswith("num_")]]
```

### 🔍 Position-Based Indexing

For pandas-style iloc functionality:

```python
# Single value
value = df.iloc[0, 0]

# Row and column slices
subset = df.iloc[0:5, [0, 2, 4]]

# All rows, specific columns
cols_subset = df.iloc[:, 1:4]
```

### 🗂️ Column Grouping

Organize related columns for easier access:

```python
# Create column groups
df.group_columns("demographics", ["age", "gender", "location"])
df.group_columns("financial", ["income", "expenses", "savings"])

# Access groups as attributes
demo_data = df.demographics  # Returns JDF with only demographic columns
financial_data = df.financial
```

### 🔎 SQL-like Queries

Filter data using familiar SQL syntax:

```python
# Basic comparisons
active_users = df.query("status == 'active' and age > 25")

# Range queries
mid_range = df.query("price between 100 and 500")

# Pattern matching
smiths = df.query("name like '%Smith%'")

# List membership
categories = df.query("category in ('A', 'B', 'Premium')")

# Complex nested conditions
filtered = df.query("(age > 30 or income > 50000) and status != 'inactive'")

# NULL checking
complete_records = df.query("email is not null and phone is not null")
```

### 📊 Data Analysis and Profiling

#### Class Counting
```python
# Count unique combinations with percentages
gender_counts = df.count_classes(["gender"])
category_distribution = df.count_classes(["category", "region"])
```

#### Missing Value Analysis
```python
# Comprehensive missing value report
missing_report = df.missing()
print(missing_report)

# Example output:
#    column    missing_count  missing_percentage
# 0  name      0              0.0
# 1  email     45             4.5
# 2  phone     120            12.0
```

#### Comprehensive Profiling
```python
# Full dataset profile
profile = df.profile()

# Profile specific columns
numerical_profile = df.profile(subset=["age", "income", "score"])
```

The profile includes:
- Data types and missing value statistics
- Numerical statistics (min, max, mean, median, std, skewness)
- Categorical value counts (for columns with < 20 unique values)
- Memory usage estimation

### 🔧 Data Transformation and Scaling

```python
from jadaf.transform.scalers import Scaler, ScalerType

# Standard scaling (default)
scaler = Scaler()
scaled_data = scaler.fit_transform(df[["height", "weight", "age"]])

# Different scaling methods
robust_scaler = Scaler(ScalerType.ROBUST)
minmax_scaler = Scaler("m")  # Short form

# Transform new data
new_scaled = scaler.transform(new_data)

# Inverse transformation
original_data = scaler.inverse_transform(scaled_data)
```

### 💾 Data Export

#### Dictionary Export
```python
# Convert to list of dictionaries
records = df.to_dict(orient="records")
```

#### JSON Export
```python
# Return as JSON string
json_string = df.to_json()

# Save to file
df.to_json("output.json")
```

#### CSV Export
```python
# Return as CSV string
csv_string = df.to_csv()

# Save to file with custom options
df.to_csv("output.csv", sep=";", header=True)
```

## 💡 Examples

### Complete Data Analysis Workflow

```python
import jadaf as jd
from jadaf.transform.scalers import Scaler

# 1. Load and explore data
df = jd.load_csv("customer_data.csv")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {', '.join(df.columns)}")

# 2. Check data quality
missing_info = df.missing()
problematic_cols = missing_info.query("missing_percentage > 10")
print(f"Columns with >10% missing: {problematic_cols['column'].to_list()}")

# 3. Group related columns
df.group_columns("personal", ["name", "age", "gender", "email"])
df.group_columns("behavior", ["login_frequency", "purchase_count", "last_activity"])
df.group_columns("financial", ["income", "spending", "credit_score"])

# 4. Filter and analyze specific segments
high_value_customers = df.query(
    "income > 75000 and purchase_count > 10 and credit_score > 700"
)

# 5. Analyze patterns in different segments
segment_analysis = high_value_customers.count_classes(["gender", "age_group"])

# 6. Scale numerical features for ML
scaler = Scaler("standard")
scaled_features = scaler.fit_transform(df.financial)

# 7. Export results
high_value_customers.to_json("high_value_segment.json")
segment_analysis.to_csv("segment_analysis.csv")
```

### Sales Data Analysis

```python
import jadaf as jd

# Load sales data
sales = jd.load_csv("sales_2023.csv")

# Group sales columns
sales.group_columns("dates", ["order_date", "ship_date", "delivery_date"])
sales.group_columns("amounts", ["subtotal", "tax", "shipping", "total"])
sales.group_columns("location", ["customer_city", "customer_state", "warehouse"])

# Analyze Q4 performance
q4_sales = sales.query("order_date >= '2023-10-01' and order_date <= '2023-12-31'")

# Find high-value orders
high_value = q4_sales.query("total > 1000")

# Geographic analysis
regional_performance = q4_sales.count_classes(["customer_state"])

# Export insights
regional_performance.to_csv("q4_regional_analysis.csv")
high_value.location.to_json("high_value_locations.json")
```

### Data Cleaning Pipeline

```python
import jadaf as jd

# Load raw data
raw_data = jd.load_csv("raw_customer_data.csv")

# Identify data quality issues
profile = raw_data.profile()
missing_summary = raw_data.missing()

# Clean data step by step
# 1. Remove rows with critical missing data
clean_data = raw_data.query("email is not null and customer_id is not null")

# 2. Filter out invalid entries
clean_data = clean_data.query("age > 0 and age < 120")

# 3. Focus on active customers
active_customers = clean_data.query("last_login >= '2023-01-01'")

# 4. Select relevant columns for analysis
analysis_cols = active_customers["customer_id, name, age, email, *_score, last_*"]

# 5. Export cleaned data
analysis_cols.to_csv("cleaned_customer_data.csv")

print(f"Original records: {raw_data.shape[0]}")
print(f"Cleaned records: {analysis_cols.shape[0]}")
print(f"Retention rate: {analysis_cols.shape[0]/raw_data.shape[0]*100:.1f}%")
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature-amazing-feature`)
3. **Make your changes** and add tests
4. **Run the test suite** (`pytest`)
5. **Commit your changes** (`git commit -m 'Add amazing feature'`)
6. **Push to your branch** (`git push origin feature-amazing-feature`)
7. **Open a Pull Request**

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/yourusername/JadaF.git
cd JadaF

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .

# Run tests
pytest

# Run linting
black src/
flake8 src/
```

## 📋 Roadmap

- [ ] **Integration with more data sources** (databases, APIs)
- [ ] **Advanced statistical functions** (correlation matrices, hypothesis testing)
- [ ] **Visualization helpers** (integration with plotting libraries)
- [ ] **Performance optimizations** (lazy evaluation, parallel processing)
- [ ] **Extended SQL support** (window functions, CTEs)
- [ ] **Data validation framework** (schema validation, constraint checking)

## 🐛 Bug Reports and Feature Requests

Please use [GitHub Issues](https://github.com/yourusername/JadaF/issues) to:
- Report bugs
- Request new features
- Ask questions

When reporting bugs, please include:
- Your operating system and Python version
- JadaF version
- Minimal code example that reproduces the issue
- Expected vs. actual behavior

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Polars Team** - for creating an amazing DataFrame library
- **Contributors** - for making JadaF better
- **Community** - for feedback and feature suggestions

---

<div align="center">

**Made with ❤️ for the data science community**

[Documentation](https://github.com/yourusername/JadaF#documentation) • 
[Examples](https://github.com/yourusername/JadaF#examples) • 
[Contributing](https://github.com/yourusername/JadaF#contributing) • 
[Issues](https://github.com/yourusername/JadaF/issues)

</div>