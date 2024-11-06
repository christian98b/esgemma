from openai import OpenAI
import pandas as pd
import random
import os
from dotenv import load_dotenv


from dotenv import load_dotenv

load_dotenv()  

client = OpenAI(api_key=os.getenv('OPENAIAPI'))

def extend_sentence(end_target):
    try:
        position = random.choice(['beginning', 'middle', 'end'])
        additional_prompting = 'The sentence you create should contain information about how you try to achieve the target.'
        additional_sentences_length = random.randint(8, 12)
        #The Corporate topic is needed in case there is not target.
        #The coporate topics a synthesized
        corporate_topic = random.choice([
            "Strategic Planning", 
            "Performance Metrics", 
            "Change Management", 
            "Corporate Governance", 
            "Supply Chain Optimization", 
            "Talent Acquisition", 
            "Market Analysis", 
            "Digital Transformation", 
            "Financial Forecasting", 
            "Brand Positioning", 
            "Customer Retention", 
            "Lean Manufacturing", 
            "Diversity and Inclusion", 
            "Data Analytics", 
            "Product Innovation", 
            "Mergers and Acquisitions", 
            "Regulatory Compliance", 
            "Employee Engagement", 
            "Quality Assurance", 
            "Sustainability Initiatives", 
            "Competitive Intelligence", 
            "Revenue Growth", 
            "Cost Reduction", 
            "Crisis Management", 
            "Intellectual Property", 
            "Stakeholder Relations", 
            "Operational Efficiency", 
            "Corporate Social Responsibility", 
            "Leadership Development", 
            "Risk Assessment", 
            "Business Continuity", 
            "Vendor Management", 
            "Cybersecurity", 
            "Customer Experience", 
            "Process Improvement", 
            "Global Expansion", 
            "Innovation Management", 
            "Workforce Planning", 
            "Asset Management", 
            "Market Segmentation", 
            "Corporate Culture", 
            "Inventory Management", 
            "Strategic Partnerships", 
            "Business Ethics", 
            "Performance Evaluation", 
            "Project Management", 
            "Knowledge Management",
            "Customer Relationship Management", 
            "Succession Planning", 
            "Agile Methodology",
            "Blockchain Integration", 
            "Remote Work Policies", 
            "Artificial Intelligence Adoption", 
            "Cash Flow Management", 
            "Sustainable Supply Chains", 
            "Workplace Wellness", 
            "Predictive Analytics", 
            "Corporate Restructuring", 
            "Employer Branding", 
            "Circular Economy", 
            "Data Governance", 
            "Cultural Intelligence", 
            "Demand Forecasting", 
            "Digital Marketing", 
            "Organizational Design", 
            "Emotional Intelligence", 
            "Business Process Outsourcing", 
            "Productivity Optimization", 
            "Continuous Improvement", 
            "Enterprise Resource Planning", 
            "Conflict Resolution", 
            "Customer Segmentation", 
            "Business Model Innovation", 
            "Negotiation Strategies", 
            "Six Sigma", 
            "Omnichannel Strategy", 
            "Cross-functional Collaboration", 
            "Green Technology", 
            "Thought Leadership", 
            "Scalability Planning", 
            "Total Quality Management",
            "Lean Startup", 
            "Return on Investment", 
            "Scenario Planning", 
            "Capacity Building", 
            "Corporate Philanthropy", 
            "Disruptive Innovation", 
            "Growth Hacking", 
            "Key Performance Indicators", 
            "Mindfulness in Leadership", 
            "Net Promoter Score", 
            "Organizational Agility", 
            "Design Thinking", 
            "Value Chain Analysis", 
            "Workforce Diversity", 
            "Blue Ocean Strategy", 
            "360-Degree Feedback", 
            "Gamification", 
            "Intrapreneurship", 
            "Zero-Based Budgeting"
            ])
        prompt = f"""Write a {additional_sentences_length+1} sentences long corporate text about {corporate_topic}, from the view of a company"""
        response = client.chat.completions.create(
           messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="gpt-3.5-turbo",
                temperature=1.0
        )
        extended_text = response.choices[0].message.content
        print(extended_text)
        print()
        return [extended_text,corporate_topic]
    except Exception as e:
        raise Exception(f'An error occurred while extending the sentence: {e}')
        

def process_sentences(file_path,file_path_new):
    try:
        df = pd.read_excel(file_path, engine='openpyxl')

        if 'end_target_text_extended' not in df.columns:
            raise Exception('The Excel file does not contain a Sentences column.')

        df['end_target_text_topic'] = None

        print(f'Rows to process: {len(df.index)}')

        for index, row in df.iterrows():
            print(f'Currently Processing: {index} of {len(df.index)}')
            if pd.isnull(row['end_target_text_extended']):
                extended_sentence, corporate_topic = extend_sentence(row['end_target'])
                df.at[index, 'end_target_text_extended'] = extended_sentence
                df.at[index, 'end_target_text_topic'] = corporate_topic
        df.to_excel(file_path_new, index=False, engine='openpyxl')
        print("Processing complete. The file has been created.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    file_path = 'nach-zwischenpräsi/dataset_preparation/datasets/7-extendet-texts-and-generated-no-target-ones.xlsx' 
    file_path_new = 'nach-zwischenpräsi/dataset_preparation/datasets/8-extendet-texts-and-generated-no-target-ones-with-totally-different-topic.xlsx' 
    process_sentences(file_path, file_path_new)