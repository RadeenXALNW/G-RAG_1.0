def get_library_name() -> str:
    """Prompt the user for a library name."""
    while True:
        library_name = input("Please enter a name for the library: ").strip()
        if library_name:
            return library_name
        else:
            print("Library name cannot be empty. Please try again.")