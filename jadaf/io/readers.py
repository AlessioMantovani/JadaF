"""
Data loading functions for jadaf.
"""

import polars as pl
import pandas as pd
from typing import Optional, Union
from pathlib import Path

from jadaf.core.jdf import JDF


def load_csv(
    filepath: Union[str, Path],
    delimiter: str = ",",
    has_header: bool = True,
    ignore_errors: bool = False,
    **kwargs
) -> JDF:
    """
    Load a CSV file into a JDF object.
    
    Args:
        filepath: Path to the CSV file
        delimiter: Character used to separate values
        has_header: Whether the file has a header row
        ignore_errors: Whether to ignore parsing errors
        **kwargs: Additional arguments to pass to polars.read_csv
        
    Returns:
        A JDF object containing the data from the CSV file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file can't be parsed as CSV
    """
    try:
        df = pl.read_csv(
            filepath,
            separator=delimiter,
            has_header=has_header,
            ignore_errors=ignore_errors,
            **kwargs
        )
        return JDF(df)
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            raise FileNotFoundError(f"File not found: {filepath}")
        else:
            raise ValueError(f"Error reading CSV file: {e}")
        
def load_json(
    filepath: Union[str, Path],
    **kwargs
) -> JDF:
    """
    Load a JSON file into a JDF object.
    
    Args:
        filepath: Path to the JSON file
        **kwargs: Additional arguments to pass to polars.read_json
        
    Returns:
        A JDF object containing the data from the JSON file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file can't be parsed as JSON
    """
    try:
        df = pl.read_json(filepath, **kwargs)
        return JDF(df)
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            raise FileNotFoundError(f"File not found: {filepath}")
        else:
            raise ValueError(f"Error reading JSON file: {e}")

def load_excel(
    filepath: Union[str, Path],
    sheet_name: Optional[Union[str, int]] = 0,
    has_header: bool = True,
    **kwargs
) -> JDF:
    """
    Load an Excel file into a JDF object.
    
    Args:
        filepath: Path to the Excel file
        sheet_name: Name or index of the sheet to load (0 is the first sheet)
        has_header: Whether the sheet has a header row
        **kwargs: Additional arguments to pass to pandas.read_excel
        
    Returns:
        A JDF object containing the data from the Excel file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file can't be parsed as Excel
    """
    try:
        df_pd = pd.read_excel(
            filepath,
            sheet_name=sheet_name,
            header=0 if has_header else None,
            **kwargs
        )
        df = pl.from_pandas(df_pd)
        return JDF(df)
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            raise FileNotFoundError(f"File not found: {filepath}")
        else:
            raise ValueError(f"Error reading Excel file: {e}")