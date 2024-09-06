def read_text_file(file_path: str) -> str:
    """
    Read a text file and return its content as a string.
    
    Args:
    file_path (str): The path to the text file.
    
    Returns:
    str: The content of the text file as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except IOError as e:
        print(f"Error reading file: {e}")
        return ""
