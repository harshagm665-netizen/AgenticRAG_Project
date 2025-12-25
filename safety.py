# safety.py

def check_safety(text):
    """
    Checks if the text contains any banned keywords.
    Returns: (is_safe: bool, triggered_word: str or None)
    """
    # Define your banned keywords here
    banned_keywords = ["confidential", "secret", "classified", "internal use only", "restricted"]
    
    text_lower = text.lower()
    for word in banned_keywords:
        if word in text_lower:
            return False, word
    return True, None