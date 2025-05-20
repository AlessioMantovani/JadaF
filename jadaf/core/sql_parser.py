import operator
import re
import polars as pl
from typing import Optional, List, Dict, Any, Union, Tuple, Callable


class _SQLExpressionParser:
    """Helper class for parsing SQL expressions using recursive descent."""

    def __init__(self, expr: str, columns: List[str], df: pl.DataFrame = None):
        self.expr = expr
        self.columns = columns
        self.df = df  # Store reference to dataframe for type checking
        self.pos = 0  # Current position in the expression

        # Define operators and their corresponding functions
        self.comparison_ops = {
            '==': operator.eq,
            '=': operator.eq,  # SQL-style equals
            '!=': operator.ne,
            '<>': operator.ne,  # SQL-style not equals
            '>=': operator.ge,
            '<=': operator.le,
            '>': operator.gt,
            '<': operator.lt
        }

        # Logical operators (will be handled separately)
        self.logical_ops = {
            'AND': operator.and_,
            'OR': operator.or_
        }

        # Special operators
        self.special_ops = {'BETWEEN', 'IN', 'LIKE', 'IS', 'NOT'}

    def parse_expression(self) -> pl.Expr:
        """Parse a complete expression."""
        return self._parse_logical_expression()

    def _parse_logical_expression(self) -> pl.Expr:
        """Parse expressions with AND/OR operators."""
        # Parse the first part (left of any AND/OR)
        left_expr = self._parse_comparison_expression()

        # Look ahead for logical operators
        remainder = self.expr.strip()

        while remainder:
            # Look for AND/OR at word boundaries
            and_match = re.match(r'\s*AND\s+', remainder, re.IGNORECASE)
            or_match = re.match(r'\s*OR\s+', remainder, re.IGNORECASE)

            if and_match:
                # Skip 'AND' and parse the next expression
                self.expr = remainder[and_match.end():].strip()
                right_expr = self._parse_comparison_expression()
                left_expr = left_expr & right_expr
                remainder = self.expr.strip()
            elif or_match:
                # Skip 'OR' and parse the next expression
                self.expr = remainder[or_match.end():].strip()
                right_expr = self._parse_comparison_expression()
                left_expr = left_expr | right_expr
                remainder = self.expr.strip()
            else:
                # No more logical operators
                break

        return left_expr

    def _parse_comparison_expression(self) -> pl.Expr:
        """Parse a comparison expression (possibly in parentheses)."""
        # Check for parentheses
        if self.expr.strip().startswith('('):
            return self._parse_parenthesized_expression()

        # Handle NOT operator
        if self.expr.strip().upper().startswith('NOT '):
            self.expr = self.expr.strip()[4:].strip()
            expr = self._parse_comparison_expression()
            return ~expr

        # Extract the column name and operator
        col_name, op, value_str = self._extract_comparison_parts()

        if not col_name or not op:
            raise ValueError(f"Invalid comparison expression: {self.expr}")

        # Validate column exists
        if col_name not in self.columns:
            raise KeyError(f"Column '{col_name}' not found in DataFrame")

        # Handle special operators
        if op.upper() in self.special_ops:
            return self._parse_special_operator(col_name, op.upper(), value_str)

        # Handle standard comparison operators
        if op not in self.comparison_ops:
            raise ValueError(f"Unsupported operator: {op}")

        # Parse value and create expression with proper type handling
        value = self._parse_value_with_type_checking(col_name, value_str)

        # Create the comparison expression using Polars column and parsed value
        return self.comparison_ops[op](pl.col(col_name), value)

    def _extract_comparison_parts(self) -> Tuple[str, str, str]:
        """Extract column name, operator, and value string from a comparison expression."""
        expr = self.expr.strip()

        # Special handling for IS NULL and IS NOT NULL first
        is_null_match = re.search(r'^(\w+)\s+IS\s+(NOT\s+)?NULL', expr, re.IGNORECASE)
        if is_null_match:
            col_name = is_null_match.group(1)
            not_op = is_null_match.group(2)
            self.expr = expr[is_null_match.end():].strip()
            return col_name, "IS", "NOT NULL" if not_op else "NULL"

        # Try to find the longest matching operator
        ops = sorted(self.comparison_ops.keys() | self.special_ops, key=lambda x: (-len(x), x))

        for op in ops:
            # Use regex to find the operator with word boundaries for special ops
            if op.upper() in self.special_ops:
                pattern = rf'^(.+?)\s+{re.escape(op)}\s+(.+?)(?=\s+(?:AND|OR)\s+|$)'
                match = re.match(pattern, expr, re.IGNORECASE)
                if match:
                    col_part = match.group(1).strip()
                    val_part = match.group(2).strip()

                    # Find where the current expression ends (before AND/OR)
                    remaining_match = re.search(r'\s+(AND|OR)\s+', expr, re.IGNORECASE)
                    if remaining_match:
                        self.expr = expr[remaining_match.start():].strip()
                    else:
                        self.expr = ''

                    return col_part, op, val_part
            else:
                # For regular operators, use improved logic that stops at AND/OR
                op_pattern = re.escape(op)
                # Use positive lookahead to stop before AND/OR or end of string
                pattern = rf'^(.+?)\s*{op_pattern}\s*(.+?)(?=\s+(?:AND|OR)\s+|$)'
                match = re.match(pattern, expr)
                if match:
                    col_part = match.group(1).strip()
                    val_part = match.group(2).strip()

                    # Find where the current expression ends (before AND/OR)
                    remaining_match = re.search(r'\s+(AND|OR)\s+', expr, re.IGNORECASE)
                    if remaining_match:
                        self.expr = expr[remaining_match.start():].strip()
                    else:
                        self.expr = ''

                    return col_part, op, val_part

        return None, None, None

    def _parse_parenthesized_expression(self) -> pl.Expr:
        """Parse an expression enclosed in parentheses."""
        # Find the matching closing parenthesis
        open_count = 0
        close_idx = -1
        expr = self.expr.strip()

        for i, char in enumerate(expr):
            if char == '(':
                open_count += 1
            elif char == ')':
                open_count -= 1
                if open_count == 0:
                    close_idx = i
                    break

        if close_idx == -1:
            raise ValueError(f"Unmatched parentheses in: {expr}")

        # Extract the inner expression
        inner_expr = expr[1:close_idx].strip()

        # Update expression to consume the parsed part
        self.expr = expr[close_idx + 1:].strip()

        # Parse the inner expression
        inner_parser = _SQLExpressionParser(inner_expr, self.columns, self.df)
        return inner_parser.parse_expression()

    def _parse_special_operator(self, col_name: str, op: str, value_str: str) -> pl.Expr:
        """Parse expressions with special operators like BETWEEN, IN, LIKE."""
        op = op.upper()

        if op == 'BETWEEN':
            # Format: BETWEEN lower AND upper
            and_split = re.split(r'\s+AND\s+', value_str, 1, re.IGNORECASE)
            if len(and_split) != 2:
                raise ValueError(f"Invalid BETWEEN expression: {col_name} BETWEEN {value_str}")

            lower_str, upper_str = and_split
            lower = self._parse_value_with_type_checking(col_name, lower_str.strip())
            upper = self._parse_value_with_type_checking(col_name, upper_str.strip())

            return (pl.col(col_name) >= lower) & (pl.col(col_name) <= upper)

        elif op == 'IN':
            # Format: IN (val1, val2, ...)
            if not (value_str.startswith('(') and value_str.endswith(')')):
                raise ValueError(f"Invalid IN expression: {col_name} IN {value_str}")

            values_str = value_str[1:-1].strip()
            values = self._parse_in_values_with_type_checking(col_name, values_str)

            return pl.col(col_name).is_in(values)

        elif op == 'LIKE':
            # Format: LIKE pattern
            if not (value_str.startswith("'") and value_str.endswith("'")) and \
                    not (value_str.startswith('"') and value_str.endswith('"')):
                raise ValueError(f"Invalid LIKE pattern: {value_str}. Must be quoted")

            like_pattern = value_str[1:-1]
            regex_pattern = self._like_to_regex(like_pattern)

            return pl.col(col_name).str.contains(regex_pattern)

        elif op == 'IS':
            # Format: IS [NOT] NULL
            if value_str.upper() == 'NULL':
                return pl.col(col_name).is_null()
            elif value_str.upper() == 'NOT NULL':
                return ~pl.col(col_name).is_null()
            else:
                raise ValueError(f"Invalid IS expression: {col_name} IS {value_str}")

        else:
            raise ValueError(f"Unsupported operator: {op}")

    def _get_column_dtype(self, col_name: str) -> pl.DataType:
        """Get the data type of a column from the DataFrame."""
        if self.df is not None:
            col_index = self.df.columns.index(col_name)
            return self.df.dtypes[col_index]
        return None

    def _parse_value_with_type_checking(self, col_name: str, val_str: str) -> Any:
        """Parse string values into appropriate Python types with column type awareness."""
        val_str = val_str.strip()

        # Handle quoted strings - always return as string
        if (val_str.startswith("'") and val_str.endswith("'")) or \
                (val_str.startswith('"') and val_str.endswith('"')):
            return val_str[1:-1]

        # Handle NULL value
        if val_str.upper() == 'NULL':
            return None

        # Handle boolean values
        if val_str.upper() == 'TRUE':
            return True
        if val_str.upper() == 'FALSE':
            return False

        # Get column type for smart conversion
        col_dtype = self._get_column_dtype(col_name)

        # If we have column type information, use it to guide conversion
        if col_dtype is not None:
            if col_dtype in [pl.String, pl.Utf8]:
                # For string columns, convert numeric literals to strings
                return str(val_str)
            elif col_dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                               pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
                # For integer columns, try to convert to int
                try:
                    return int(val_str)
                except ValueError:
                    raise ValueError(f"Cannot convert '{val_str}' to integer for column '{col_name}'")
            elif col_dtype in [pl.Float32, pl.Float64]:
                # For float columns, try to convert to float
                try:
                    return float(val_str)
                except ValueError:
                    raise ValueError(f"Cannot convert '{val_str}' to float for column '{col_name}'")
            elif col_dtype == pl.Boolean:
                # For boolean columns
                if val_str.lower() in ['true', '1']:
                    return True
                elif val_str.lower() in ['false', '0']:
                    return False
                else:
                    raise ValueError(f"Cannot convert '{val_str}' to boolean for column '{col_name}'")

        # Fallback to automatic type detection
        # Handle numeric values
        try:
            # Try to convert to integer first
            if '.' not in val_str:
                return int(val_str)
        except ValueError:
            pass

        try:
            # Then try float
            return float(val_str)
        except ValueError:
            pass

        # If all else fails, return as string
        return val_str

    def _parse_value(self, val_str: str) -> Any:
        """Parse string values into appropriate Python types (legacy method)."""
        val_str = val_str.strip()

        # Handle quoted strings
        if (val_str.startswith("'") and val_str.endswith("'")) or \
                (val_str.startswith('"') and val_str.endswith('"')):
            return val_str[1:-1]

        # Handle NULL value
        if val_str.upper() == 'NULL':
            return None

        # Handle boolean values
        if val_str.upper() == 'TRUE':
            return True
        if val_str.upper() == 'FALSE':
            return False

        # Handle numeric values
        try:
            # Try to convert to integer first
            return int(val_str)
        except ValueError:
            try:
                # Then try float
                return float(val_str)
            except ValueError:
                # If all else fails, return as string
                return val_str

    def _parse_in_values_with_type_checking(self, col_name: str, values_str: str) -> List[Any]:
        """Parse a comma-separated list of values for IN operator with type checking."""
        values = []
        current = ""
        in_quotes = False
        quote_char = None

        for char in values_str:
            if char in "\"'" and (not current or current[-1] != '\\'):
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False

            if char == ',' and not in_quotes:
                values.append(self._parse_value_with_type_checking(col_name, current.strip()))
                current = ""
            else:
                current += char

        if current:
            values.append(self._parse_value_with_type_checking(col_name, current.strip()))

        return values

    def _parse_in_values(self, values_str: str) -> List[Any]:
        """Parse a comma-separated list of values for IN operator (legacy method)."""
        values = []
        current = ""
        in_quotes = False
        quote_char = None

        for char in values_str:
            if char in "\"'" and (not current or current[-1] != '\\'):
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False

            if char == ',' and not in_quotes:
                values.append(self._parse_value(current.strip()))
                current = ""
            else:
                current += char

        if current:
            values.append(self._parse_value(current.strip()))

        return values

    def _like_to_regex(self, pattern: str) -> str:
        """Convert SQL LIKE pattern to regex pattern."""
        # Escape special regex characters
        escaped = re.escape(pattern)

        # Convert SQL LIKE wildcards to regex
        # % matches any sequence of characters
        # _ matches any single character
        regex = escaped.replace('\\%', '.*').replace('\\_', '.')

        # Add start and end anchors
        return f"^{regex}$"