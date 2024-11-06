from openai import OpenAI
def generate_response_remote_model(text : str, client : OpenAI, model:str) -> str:
    """
    Generate a response using the provider API and return just the answer as a string.

    This function sends a user message to the API and retrieves the generated response.

    Parameters
    ----------
    text : str
        The input text to generate a response from.

    Returns
    -------
    str
        The generated response as a string.

    Examples
    --------
    >>> generate_response("Hello, how are you?")
    "I'm doing well, thank you! How can I assist you today?"
    """
    response = client.chat.completions.create(
        messages=[
            {
            "role": "user",
            "content": text,
            }
        ],
        model=model,
        temperature=0.0,
        top_p=0.9
    )
    return response.choices[0].message.content