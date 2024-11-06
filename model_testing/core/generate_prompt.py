def generate_prompt(components : str, context : str, instruction : str, example : str, finetuned: bool = False) -> str:
    """
    Generate a prompt for a text generation task based on the given components.

    This function constructs a prompt string using specified components, context,
    instruction, and examples, depending on whether the model is finetuned.

    Parameters
    ----------
    components : str
        A string containing the components to include in the prompt. The available 
        components are 'C', 'I', 'E', 'K', and 'X'.
    
    context : str
        The context to include in the prompt.
    
    instruction : str
        The instruction or task description to include in the prompt.
    
    example : str
        An example input-output pair to include in the prompt. If `finetuned` is True,
        this part will be excluded.
    
    finetuned : bool, optional
        A flag indicating whether the model is finetuned. If True, the example part 
        will be excluded. Defaults to False.

    Returns
    -------
    str
        The generated prompt as a string.

    Examples
    --------
    >>> generate_prompt('CIEKX', 'This is some context', 'Translate to French', 'Input: Hello, Output: Bonjour')
    'Below is an instruction that describes a task, paired with an input that 
    provides further context. Write a response that appropriately completes the 
    request.\n\n### Instruction: Translate to French\n\nThis is some context\n\n
    Input: Hello, Output: Bonjour\n\n### Response Formatting: Only answer in the 
    following XML format:\n<answer><classification><end_target>Target</end_target>
    </classification><extraction><end_target_year>Year</end_target_year></extraction>
    <quote>...</quote></answer>'
    """
    prompt_parts = {
        'C': f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.",
        'I': f"### Instruction: {instruction}\n\n",
        'E': f"### Examples: {example}\n\n" if not finetuned else "",
        'K': f"### Context: {context}\n\n",
        'X': f"### Response Formatting: Only answer in the following XML format:\n<answer><classification><end_target>Target</end_target></classification><extraction><end_target_year>Year</end_target_year></extraction><quote>...</quote></answer>",
    }
    return ''.join(prompt_parts[c] for c in components)