import operator
import re
import polars as pl
from typing import Optional, List, Dict, Any, Union, Tuple, Callable
from datetime import datetime, date, time


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

        # Track column data types
        self.column_dtypes = self._get_column_dtypes()

    def _get_column_dtypes(self) -> Dict[str, pl.DataType]:
        """Get data types for all columns in the DataFrame."""
        if self.df is None:
            return {}

        return {col: dtype for col, dtype in zip(self.df.columns, self.df.dtypes)}

    def parse_expression(self) -> pl.Expr:
        """Parse a complete expression."""
        return self._parse_logical_expression()

    def _parse_logical_expression(self) -> pl.Expr:
        """Parse expressions with AND/OR operators."""
        left_expr = self._parse_comparison_expression()

        remainder = self.expr.strip()

        while remainder:
            and_match = re.match(r'\s*AND\s+', remainder, re.IGNORECASE)
            or_match = re.match(r'\s*OR\s+', remainder, re.IGNORECASE)

            if and_match:
                self.expr = remainder[and_match.end():].strip()
                right_expr = self._parse_comparison_expression()
                left_expr = left_expr & right_expr
                remainder = self.expr.strip()
            elif or_match:
                self.expr = remainder[or_match.end():].strip()
                right_expr = self._parse_comparison_expression()
                left_expr = left_expr | right_expr
                remainder = self.expr.strip()
            else:
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
        """Extract column name, operator, and value string from a comparison expression.
        Uses proper tokenization to avoid mis-parsing expressions."""
        expr = self.expr.strip()

        # Special handling for IS NULL and IS NOT NULL first
        is_null_match = re.search(r'^(\w+)\s+IS\s+(NOT\s+)?NULL', expr, re.IGNORECASE)
        if is_null_match:
            col_name = is_null_match.group(1)
            not_op = is_null_match.group(2)
            self.expr = expr[is_null_match.end():].strip()
            return col_name, "IS", "NOT NULL" if not_op else "NULL"

        # Step 1: Tokenize the expression to identify column name and operator
        # First, find the column name followed by an operator
        column_and_op_pattern = r'^\s*(\w+)\s*(' + '|'.join(re.escape(op) for op in
                                                            sorted(self.comparison_ops.keys() | self.special_ops,
                                                                   key=lambda x: (-len(x), x))) + r')\s*'

        column_op_match = re.match(column_and_op_pattern, expr, re.IGNORECASE)

        if not column_op_match:
            return None, None, None

        col_name = column_op_match.group(1).strip()
        op = column_op_match.group(2).strip()

        # Step 2: Extract the value part, being careful about logical operators
        # Position after the operator
        pos_after_op = column_op_match.end()

        # Handle special operators differently
        if op.upper() in self.special_ops:
            if op.upper() == 'BETWEEN':
                # For BETWEEN, we need to extract "value1 AND value2"
                # Find the position of the logical AND/OR that terminates this expression
                between_pattern = r'(.*?)\s+AND\s+(.*?)(?=\s+(?:AND|OR)\s+|$)'
                between_match = re.match(between_pattern, expr[pos_after_op:], re.IGNORECASE)

                if between_match:
                    value_str = f"{between_match.group(1).strip()} AND {between_match.group(2).strip()}"
                    end_pos = pos_after_op + between_match.end()
                else:
                    # If no terminating AND/OR, take all remaining text
                    value_str = expr[pos_after_op:].strip()
                    end_pos = len(expr)

            elif op.upper() == 'IN':
                # For IN, we need to extract the parenthesized list
                in_value = ""
                paren_count = 0
                i = pos_after_op

                # Skip whitespace
                while i < len(expr) and expr[i].isspace():
                    i += 1

                if i < len(expr) and expr[i] == '(':
                    in_value += '('
                    paren_count = 1
                    i += 1

                    while i < len(expr) and paren_count > 0:
                        if expr[i] == '(':
                            paren_count += 1
                        elif expr[i] == ')':
                            paren_count -= 1
                        in_value += expr[i]
                        i += 1

                    end_pos = i
                    value_str = in_value
                else:
                    return None, None, None

            elif op.upper() == 'LIKE':
                # For LIKE, we need to extract the quoted string pattern
                i = pos_after_op
                # Skip whitespace
                while i < len(expr) and expr[i].isspace():
                    i += 1

                if i < len(expr) and expr[i] in ["'", '"']:
                    quote_char = expr[i]
                    start_pos = i
                    i += 1
                    # Find the closing quote
                    while i < len(expr) and expr[i] != quote_char:
                        # Skip escaped quotes
                        if expr[i] == '\\' and i + 1 < len(expr) and expr[i + 1] == quote_char:
                            i += 2
                        else:
                            i += 1

                    if i < len(expr):  # Found closing quote
                        i += 1  # Include the closing quote
                        value_str = expr[start_pos:i]
                        end_pos = i
                    else:
                        return None, None, None
                else:
                    return None, None, None

            elif op.upper() == 'IS':
                # For IS, handle NULL and NOT NULL
                is_value_pattern = r'\s*(NOT\s+)?NULL'
                is_value_match = re.match(is_value_pattern, expr[pos_after_op:], re.IGNORECASE)

                if is_value_match:
                    not_part = is_value_match.group(1)
                    value_str = f"{'NOT ' if not_part else ''}NULL"
                    end_pos = pos_after_op + is_value_match.end()
                else:
                    return None, None, None
            else:
                # For other special operators, find the end of the value before any AND/OR
                logical_op_match = re.search(r'\s+(AND|OR)\s+', expr[pos_after_op:], re.IGNORECASE)

                if logical_op_match:
                    value_str = expr[pos_after_op:pos_after_op + logical_op_match.start()].strip()
                    end_pos = pos_after_op + logical_op_match.start()
                else:
                    value_str = expr[pos_after_op:].strip()
                    end_pos = len(expr)
        else:
            # For standard comparison operators, extract the value part carefully
            value_part = ""
            i = pos_after_op

            # Skip whitespace
            while i < len(expr) and expr[i].isspace():
                i += 1

            # Handle quoted values
            if i < len(expr) and expr[i] in ["'", '"']:
                quote_char = expr[i]
                value_part += quote_char
                i += 1

                while i < len(expr) and expr[i] != quote_char:
                    # Handle escaped quotes
                    if expr[i] == '\\' and i + 1 < len(expr) and expr[i + 1] == quote_char:
                        value_part += expr[i:i + 2]
                        i += 2
                    else:
                        value_part += expr[i]
                        i += 1

                if i < len(expr):  # Found closing quote
                    value_part += quote_char
                    i += 1
            else:
                # For non-quoted values, read until whitespace or logical operator
                while i < len(expr) and not re.match(r'\s+(AND|OR)\s+', expr[i:], re.IGNORECASE):
                    value_part += expr[i]
                    i += 1
                    if i >= len(expr):
                        break
                    # Stop if we encounter whitespace followed by AND/OR
                    if expr[i:i + 1].isspace():
                        next_word_match = re.match(r'\s+(\w+)', expr[i:])
                        if next_word_match and next_word_match.group(1).upper() in ["AND", "OR"]:
                            break

            # Extract just to the logical operator or end
            logical_op_match = re.search(r'\s+(AND|OR)\s+', expr[i:], re.IGNORECASE)

            if logical_op_match:
                end_pos = i
                value_str = value_part.strip()
            else:
                end_pos = len(expr)
                value_str = value_part.strip()

        # Update the remaining expression
        if end_pos < len(expr):
            self.expr = expr[end_pos:].strip()
        else:
            self.expr = ""

        return col_name, op, value_str

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
        return self.column_dtypes.get(col_name, None)

    def _parse_value_with_type_checking(self, col_name: str, val_str: str) -> Any:
        """Parse string values into appropriate Python types with column type awareness."""
        val_str = val_str.strip()

        # Handle quoted strings - always return as string
        if (val_str.startswith("'") and val_str.endswith("'")) or \
                (val_str.startswith('"') and val_str.endswith('"')):
            string_value = val_str[1:-1]

            # Check if it's a date column - attempt to parse date strings
            col_dtype = self._get_column_dtype(col_name)
            if col_dtype in [pl.Date, pl.Datetime]:
                return self._parse_date_or_datetime(string_value)
            return string_value

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
            elif col_dtype in [pl.Date, pl.Datetime]:
                # For date/datetime columns, try to parse as date
                try:
                    return self._parse_date_or_datetime(val_str)
                except ValueError:
                    raise ValueError(f"Cannot parse '{val_str}' as date/datetime for column '{col_name}'")

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

    def _parse_date_or_datetime(self, date_str: str) -> Union[date, datetime]:
        """Parse a string as a date or datetime object."""
        # Try common date formats
        formats = [
            '%Y-%m-%d',  # ISO format: 2023-01-05
            '%Y/%m/%d',  # Slash format: 2023/01/05
            '%d-%m-%Y',  # European format: 05-01-2023
            '%d/%m/%Y',  # European slash: 05/01/2023
            '%Y-%m-%d %H:%M:%S',  # ISO with time: 2023-01-05 12:34:56
            '%Y/%m/%d %H:%M:%S',  # Slash with time: 2023/01/05 12:34:56
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # If none of the formats match, raise error
        raise ValueError(f"Cannot parse '{date_str}' as a date/datetime. Use format like '2023-01-05'")

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