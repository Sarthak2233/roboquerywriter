import ast


def split_and_store(input_string):
    """
    Splits a string representing a Python list and stores it as a real Python list.

    Args:
      input_string: The string to be split.

    Returns:
      A Python list containing the elements from the input string.
    """
    try:
        return ast.literal_eval(input_string)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing string: {e}")
        return []

