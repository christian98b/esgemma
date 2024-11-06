from openai import OpenAI
import pandas as pd
import random
import os
from dotenv import load_dotenv


from dotenv import load_dotenv

load_dotenv() 
client = OpenAI(api_key=os.getenv('OPENAIAPI'))

def extend_sentence(original_sentence, end_target):
    try:
        position = random.choice(['beginning', 'middle', 'end'])
        additional_prompting = 'The sentence you create should contain information about how you try to achieve the target.'
        additional_sentences_length = random.randint(8, 12)
        #The Corporate topic is needed in case there is not target.
        #The coporate topics a synthesized
        corporate_topic = random.choice([
            "Corporate Governance",
            "Financial Performance",
            "Human Resources Management",
            "Product Development",
            "Supply Chain Management",
            "Customer Service",
            "Mergers and Acquisitions",
            "Risk Management",
            "Cybersecurity",
            "Diversity and Inclusion",
            "Corporate Social Responsibility (non-environmental)",
            "Global Expansion",
            "Workplace Culture",
            "Employee Training and Development",
            "Data Analytics",
            "Entrepreneurship",
            "Digital Transformation"
            ])
        corporate_topic_year = random.choice([
            2020,
            2025,
            2030,
            2035,
            2040,
            2045,
            2050,
            2055,
            2060,
            2065,
            2070
        ])
        if(end_target == "No target"):
            prompt = f"""Write a {additional_sentences_length+1} sentences long corporate text about {corporate_topic} als well as in what year ({corporate_topic_year}) you try to achieve that and how"""
        else:
            prompt = (
                f"""Extend the given original sentence about {end_target} emissions, by adding {additional_sentences_length} sentences while incorporating the original essense of the text somewhere in the {position} of your answer.
                In your answer write about the reduction target and how its reached from the original sentence once.
                The rest of the answer should be about other topics.
                {additional_prompting}
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
                temperature=0.8
        )
        extended_text = response.choices[0].message.content
        return extended_text
    except Exception as e:
        raise Exception(f'An error occurred while extending the sentence: {e}')
        

def process_sentences(file_path,file_path_new):
    try:
        df = pd.read_excel(file_path, engine='openpyxl')

        if 'end_target_text_sanitized' not in df.columns:
            raise Exception('The Excel file does not contain a Sentences column.')

        df['end_target_text_extended'] = None

        print(f'Rows to process: {len(df.index)}')

        for index, row in df.iterrows():
            print(f'Currently Processing: {index} of {len(df.index)}')
            original_sentence = row['end_target_text_sanitized']
            end_target = row['end_target']
            extended_sentence = extend_sentence(original_sentence,end_target)
            df.at[index, 'end_target_text_extended'] = extended_sentence
            df.to_excel(file_path_new, index=False, engine='openpyxl')
        print("Processing complete. The file has been created.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    file_path = 'nach-zwischenpräsi/dataset_preparation/datasets/6-removed-everything-with-no-date-in-sanitzied-and-removed-classes-less-50.xlsx' 
    file_path_new = 'nach-zwischenpräsi/dataset_preparation/datasets/7-extendet-texts-and-generated-no-target-ones.xlsx' 
    process_sentences(file_path, file_path_new)