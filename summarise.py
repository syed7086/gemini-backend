from google import genai
import os

client = genai.Client(api_key= "AIzaSyCnsrjxjbT-kis56lARSvj7ivZxJF0a6eI")

symptoms = """
Patient is a 45-year-old male with persistent cough, mild fever, and occasional shortness of breath.
No known allergies. History of smoking for 10 years.
"""


prompt = f"""
You are a medical assistant. Given the patient's symptoms, suggest the 3 most likely diagnoses.
For each, provide:
- Name of the condition
- ICD-10 code (or SNOMED CT code if possible)
- Brief reasoning
- Confidence score from 0 to 1

Return the result as a JSON array.
Patient symptoms: {symptoms}
"""

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents= prompt
)

print(response.text)

