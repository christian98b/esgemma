from openai import OpenAI
import pandas as pd
import random
import os
from dotenv import load_dotenv


from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAIAPI'))

def sanitize_sentence(original_sentence, end_target):
    try:

        prompt = (
            f"""Please prepare some data for me.
            Your task is to take a sentence and revise it. The revised sentence should meet the following criteria:
            Characters like " are removed.
            Source citations are removed.
            Links are removed.
            Page numbers are removed.
            Text in these brackets () or [] are removed, including the brackets.
            If the entire sentence you are supposed to revise is a URL, respond only with: ERROR-URL
            If the text is in a language other than English, translate it into English as well.
            It is very important that your response is only the revised sentence!
            Original sentence:\n"""
            f"{original_sentence}\n"
            )
        response = client.chat.completions.create(
           messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="gpt-3.5-turbo",
        )
        extended_text = response.choices[0].message.content
        return extended_text
    except Exception as e:
        raise Exception(f'An error occurred while extending the sentence: {e}')
        

def process_sentences(file_path,file_path_new):
    try:
        df = pd.read_excel(file_path, engine='openpyxl')

        if 'end_target_text' not in df.columns:
            raise Exception('The Excel file does not contain a Sentences column.')

        df['end_target_text_sanitized'] = None

        print(f'Rows to process: {len(df.index)}')

        for index, row in df.iterrows():
            print(f'Currently Processing: {index} of {len(df.index)}')
            original_sentence = row['end_target_text']
            end_target = row['end_target']
            extended_sentence = sanitize_sentence(original_sentence,end_target)
            df.at[index, 'end_target_text_sanitized'] = extended_sentence

        df.to_excel(file_path_new, index=False, engine='openpyxl')
        print("Processing complete. The file has been created.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    file_path = 'nach-zwischenpräsi/dataset_preparation/datasets/3-dataset-overall-emissions-reduction-only-country.xlsx' 
    file_path_new = 'nach-zwischenpräsi/dataset_preparation/datasets/4-dataset-overall-emissions-reduction-only-country-sanitized-text.xlsx' 
    process_sentences(file_path, file_path_new)