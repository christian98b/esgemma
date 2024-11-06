from typing import Callable, Tuple, Dict, List, Optional
from core.generate_prompt import generate_prompt
from core.generate_response import generate_response_remote_model
from core.eval_function import *
import pandas as pd
from openai import OpenAI

def eval_function(instruction: str, example:str, iteration:int, ground_truth_dataframe : pd.DataFrame, target_column :str ,target_year_column: str, context_column : str ,prompt_structures:list[str] ,save_to_docx : bool = False, results_folder_path : str = 'results' ,finetuned:bool = False, client : OpenAI = None, model: str = None, generate_response: Optional[Callable[[str], str]] = None) -> Tuple[map,map]:
    """
    Evaluate language model outputs using given prompt structures and ground truth data.

    This function generates prompts based on the provided structures, instruction, and context,
    then collects and evaluates the language model's responses. It also allows saving the results
    to a DOCX file if `save_to_docx` is set to True.

    Parameters
    ----------
    instruction : str
        The instruction to be included in the prompt.
    example : str
        The example to be included in the prompt.
    iteration : int
        The current iteration number.
    ground_truth_dataframe : pd.DataFrame
        A DataFrame containing the ground truth data with columns specified by `target_column`,
        `target_year_column`, and `context_column`.
    target_column : str
        The column name in the ground truth DataFrame containing the target values.
    target_year_column : str
        The column name in the ground truth DataFrame containing the target year values.
    context_column : str
        The column name in the ground truth DataFrame containing the context text.
    prompt_structures : List[str]
        A list of prompt structures to be used for generating prompts.
    client : OpenAi, optional
        The openai client. Use when not providing a generate_response
    model : str, optional
        The model that should be used on the api. Use when not providing a generate_response
    save_to_docx : bool, optional
        Whether to save the results to a DOCX file, by default False.
    finetuned : bool, optional
        Whether the language model is finetuned, by default False.
        If Finetuned the examples are left out in the prompt structure
    generate_response : Optional[Callable[[str], str]], optional
        A custom response generation function to replace `generate_response_remote_model`. If not provided,
        the default `generate_response_remote_model` function will be used, by default None.

    Returns
    -------
    Tuple[Dict[str, Dict[str, float]], Dict[str, List[str]]]
        A tuple containing two dictionaries:
        - `scores`: A dictionary where keys are prompt structures and values are dictionaries containing
          evaluation scores for each structure.
        - `llm_outputs`: A dictionary where keys are prompt structures and values are lists of language model outputs.

    Notes
    -----
    This function currently assumes that the ground truth DataFrame has columns specified by `target_column`,
    `target_year_column`, and `context_column`, and that the evaluation is based on these columns.
    If using this function for other tasks, you may need to modify the following parts:

    - The way ground truth data is extracted and processed.
    - The `evaluate_llm_output` function, which is not defined here and should be modified to suit your specific evaluation needs.
    - The DataFrame columns used in the function.

    Additionally, the `generate_prompt` and `generate_response` functions, which are not defined here,
    may also need modification depending on your use case.

    Examples
    --------
    >>> ground_truth_dataframe = pd.DataFrame({'target': ['A', 'B', 'C'], 'target_year': [2020, 2021, 2022], 'context': ['Text 1', 'Text 2', 'Text 3']})
    >>> prompt_structures = ['CIEKX', 'CIEXK']
    
    """
    ground_truth = ground_truth_dataframe[[target_column, target_year_column, context_column]].to_dict('records')
    ground_truth = [(row[target_column], row[target_year_column], row[context_column]) for row in ground_truth]

    scores = {}
    llm_outputs = {}

    for i, structure in enumerate(prompt_structures):
        print(f"Structure: {i + 1}/{len(prompt_structures)}")
        llm_output = []
        for index, row in ground_truth_dataframe.iterrows():
            print(f"{index + 1}/{len(ground_truth_dataframe)}")
            text = generate_prompt(components=structure, instruction=instruction, context=row[context_column], example=example, finetuned=finetuned)
            if generate_response:
                response = generate_response(text)
            else:
                response = generate_response_remote_model(text, client=client, model=model)
            llm_output.append(f"Ground truth: {row[target_column]} {row[target_year_column]} Response: {response}")
        llm_outputs[structure] = llm_output
        scores[structure] = evaluate_llm_output(ground_truth, llm_output)

        if save_to_docx:
            for i, value in enumerate(prompt_structures):
                save_results_to_docx(scores[value], folder_path=results_folder_path, file_name=f'iteration_{iteration}')

    return scores, llm_outputs
