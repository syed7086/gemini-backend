import os
import json
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")


client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title= "AI Medical Diagnosis Assistant")

def extract_json(text):
    match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if match:
        return match.group(1)
    raise ValueError("No JSON found")

class SymptomsInput(BaseModel):
    symptoms : str
    top_k: int = 3

@app.post('/predict')
async def predict_diagnosis(data: SymptomsInput):
    prompt = f"""
    You are a medical assistant. Given the patient's symptoms, suggest the {data.top_k} most likely diagnoses.
    For each, provide:
    - Condition name
    - ICD-10 code (or SNOMED CT if possible)
    - Brief reasoning
    - Confidence score (0 to 1)

    Return only valid JSON with the following schema:
    [
      {{
        "condition": "string",
        "code": "string",
        "system": "ICD-10 or SNOMED CT",
        "reasoning": "string",
        "confidence": 0.0
      }}
    ]

    Symptoms: {data.symptoms}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents= prompt
        )
        
        raw_output = response.text
        json_text = extract_json(raw_output)
        
        try:
            predictions = json.loads(json_text)
        except json.JSONDecodeError as e:
            print("JSON parsing error: ", e)
            predictions = None
        
        
        return {"predictions":predictions}
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Gemini returned invalid JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 8000))  # Use Render's assigned port
        uvicorn.run(app, host="0.0.0.0", port=port)