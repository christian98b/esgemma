import gradio as gr
import fitz  # PyMuPDF
from openai import OpenAI
import openai
import re
from tqdm import tqdm
import os

# Initialize the OpenAI client globally
openai_key = os.getenv('OPENAIAPI')
runpod_key = os.getenv('RUNPOD')
vllm_url = "https://api.runpod.ai/v2/vllm-jrfeqatakzy6pe/openai/v1"
#client = OpenAI(api_key=openai_key)
client= OpenAI(api_key=runpod_key, base_url=vllm_url)
#client= OpenAI(api_key='lm-studio', base_url='http://100.72.55.38:1234/v1')

instruction = """You are an expert ESG (Environmental, Social, and Governance) analyst who conducts ESG research by analyzing texts to identify the presence of climate balance targets. Your primary task is to classify identified targets into one of four predefined classes and determine the target year for the climate balance target. Only consider overall climate balance targets, meaning that they are company-wide.
The possible classes are “Carbon neutral(ity)”, “Emissions reduction target”, “Net zero”, and “No target”.
Each class has equal importance, and the correct classification should reflect the most explicit target mentioned in the text. In cases where multiple classes are present:
	•	“Net zero” should only be prioritized if explicitly mentioned as a company’s overarching target.
	•	“Carbon neutral(ity)” takes precedence over “Emissions reduction target” only if it is the primary focus of the text.
	•	“Emissions reduction target” should be classified if it is directly stated and not overshadowed by “Net zero” or “Carbon neutral(ity)” commitments.
	•	If no explicit target is mentioned, classify as “No target”.
Ensure the classification is based on explicit information from the text, without assuming that one target implies another unless clearly stated."""

examples= """
### Examples:

"""


def call_agent(text:str) -> tuple[str,str]:
    response = client.chat.completions.create(
        messages=[
            {
            "role": "user",
            "content": text
            }
        ],
        model='chris7374/esgemma-2b',
        temperature=0.0,
        max_tokens=200
    )
    return response.choices[0].message.content, response.usage.prompt_tokens

def process_pdf(text:str, page:int) -> str:
    message = f"""
    Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction: {instruction}\n\n

    ### Context: {text}\n\n

    ### Response Formatting: Only answer in the following XML format:\n<answer><classification><end_target>Target</end_target></classification><extraction><end_target_year>Year</end_target_year></extraction><quote>...</quote></answer>\n
    """

    answer, tokens = call_agent(message)

    # Extract information using regex
    target_match = re.search(r'<end_target>(.*?)</end_target>', answer)
    year_match = re.search(r'<end_target_year>(.*?)</end_target_year>', answer)
    quote_match = re.search(r'<quote>(.*?)</quote>', answer, re.DOTALL)

    target = target_match.group(1) if target_match else "N/A"
    year = year_match.group(1) if year_match else "N/A"
    quote = quote_match.group(1) if quote_match else "N/A"

    ###Extra check if year is in text in case Class is predicted
    if (year != 'No target' and year != 'N/A'):
        year_in_text : str =""
        quote_in_text : str=""
        if year in text:
            year_in_text = "Year in text"
        if quote in text:
            quote_in_text = "Quote is in text"        
        return f"Page:{page} Total Token: {tokens} Target: {target} Year: {year} Quote: {quote}\n{year_in_text}\n {quote_in_text} \n"
    return f"Page:{page} Total Token: {tokens} Target: {target} Year: {year} Quote: {quote}\n"

def extract_first_words(pdf_file):
    page_words = ""
    doc = fitz.open(pdf_file)
    page_words = f"Seitenzahl: {len(doc)} \n"
    yield page_words
    for index, page in enumerate(doc):
        text = page.get_text()
        page_words += process_pdf(text=text, page=index+1) + "\n"
        yield page_words

with gr.Blocks() as app:
    file = gr.File(file_types=['pdf'])
    output = gr.Text()
    submitButton = gr.Button('Analyse')
    clearButton = gr.Button('Clear')

    process_function = submitButton.click(extract_first_words, inputs=file, outputs=output, show_progress="minimal")
    clearButton.click(lambda: [None,None], outputs=[file, output], cancels=[process_function])

app.launch()
