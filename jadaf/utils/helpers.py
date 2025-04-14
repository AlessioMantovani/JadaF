"""
Utility functions for jadaf.
"""

import os
from typing import List, Optional, Union
from pathlib import Path


def list_csv_files(directory: Union[str, Path]) -> List[str]:
    """
    List all CSV files in a directory.
    
    Args:
        directory: Path to the directory
        
    Returns:
        List of CSV filenames in the directory
        
    Raises:
        FileNotFoundError: If the directory doesn't exist
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    return [str(f) for f in directory.glob("*.csv")]


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory
        
    Returns:
        Path object representing the directory
    """
    path = Path(path)
    os.makedirs(path, exist_ok=True)
    return path