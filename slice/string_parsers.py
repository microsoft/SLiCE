import re
from typing import Set, List, Optional


def parse_entity_names(entity_string: str, delimiters: List[str]=[',', ';']) -> Set[str]:
    """
    Parse a string containing column names or table names into a set of single column names or table names

    Args:
        entity_string (str): String contains column name(s) or table name(s)
        delimiters (List[str]): List of delimiters to split the string by

    Returns:
        Set[str]: A set of column name strings or table name strings with white space 
        
    Examples:
        >>> parse_names("col1, col2; col3")
        {'col1', 'col2', 'col3'}
        >>> parse_names("table1, table2; table3")
        {'table1', 'table2', 'table3'}
        >>> parse_names("single_name")
        {'single_name'}
    """
    if not entity_string or not entity_string.strip():
        return set()

    result = entity_string.strip()
    # remove any tabs/newlines from the string
    result = result.replace('\t', '')
    result = result.replace('\n', '')
    
    import uuid
    # Generate a unique delimiter that's extremely unlikely to exist in the input
    unique_delimiter = f"__SPLIT_{uuid.uuid4().hex}__"

    # interactively replace each delimiter with the unique delimiter
    for delimiter in delimiters:
        result = result.replace(delimiter, unique_delimiter)

    # split the results on the unique delimiter 
    # remove any spaces/tabs/newlines from the string

    entities = {
        clean_entity(col.strip())
        for col in result.split(unique_delimiter)
        if col.strip() # skip empty strings
    }
    
    return entities

def remove_special_characters(string: str, 
                            chars_to_remove: List[str],
                            replace_char: Optional[str]='_') -> str:
    """
    remove special characters from a string

    Args:
        string (str): The string to remove special characters from
        chars_to_remove (List[str]): The characters to remove from the string
        replace_char (Optional[str]): The character to replace the removed characters with

    Returns:
        str: The string with the special characters removed
    """
    if not string or not chars_to_remove:
        return string

    # Join the list of characters into a string and escape them
    chars_str = ''.join(chars_to_remove)
    pattern = re.compile(f'[{re.escape(chars_str)}]')

    # use the pattern to replace the characters with replace_char
    result = pattern.sub(replace_char, string)
    # remove consecutive replace_chars
    if replace_char:
        result = re.sub(f'{re.escape(replace_char)}+', replace_char, result)

    return result 

def clean_entity(entity_string: str, 
                chars_to_remove: Optional[List[str]]=None,
                replace_char: str='_') -> str:
    """
    clean an entity string. If chars_to_remove is not provided, default is only keep letters, numbers, and underscores. Replace any other characters with replace_char if provided.

    Args:
        entity_string (str): The entity string to clean
        chars_to_remove (Optional[List[str]]): The characters to remove from the entity string. If not provided, default is only keep letters, numbers, and underscores.
        replace_char (Optional[str]): The character to replace the characters to remove with. If not provided, default is '_'.

    Returns:
        str: The cleaned entity string
    """
    if not entity_string:
        return ""
        
    # If chars_to_remove is provided, use remove_special_characters
    if chars_to_remove is not None:
        cleaned = remove_special_characters(entity_string, chars_to_remove, replace_char)
        return cleaned
        
    # Default behavior: keep only letters, numbers, and underscores
    # Replace all other characters with replace_char
    pattern = re.compile(r'[^a-zA-Z0-9_]')
    cleaned = pattern.sub(replace_char, entity_string)
    
    # Remove consecutive replace_chars
    if replace_char:
        cleaned = re.sub(f'{re.escape(replace_char)}+', replace_char, cleaned)
        
    # Remove replace_char from start and end
    if replace_char:
        cleaned = cleaned.strip(replace_char)
        
    return cleaned

    