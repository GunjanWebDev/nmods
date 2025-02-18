from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AzureOpenAI
from typing import Optional, Dict
from dotenv import load_dotenv
import os
import pandas as pd

from pdf_processor import PDFProcessor

# Load environment variables
load_dotenv()

# Global variable to store mappings
header_mappings: Dict[str, str] = {}

# Initialize PDF processor
pdf_processor = PDFProcessor(cache_dir="./cache")

def load_reference_mappings():
    """Load and cache the reference mappings from Excel"""
    global header_mappings
    try:
        excel_path = os.getenv("REFERENCE_EXCEL_PATH")
        if not excel_path:
            return
        
        df = pd.read_excel(excel_path)
        # Create a dictionary with RL_HEADER as key and SOV_HEADERS as value
        header_mappings = dict(zip(df['RL_HEADER'].str.lower(), df['SOV_HEADERS']))
    except Exception as e:
        print(f"Error loading reference mappings: {e}")

def initialize_processors():
    global header_mappings, pdf_processor
    try:
        # Load Excel mappings
        load_reference_mappings()
        
        # Initialize and process PDF
        pdf_path = os.getenv("REFERENCE_PDF_PATH")
        if pdf_path:
            pdf_processor.process_pdf(pdf_path)
    except Exception as e:
        print(f"Error initializing processors: {e}")

# Initialize when app starts
initialize_processors()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Azure OpenAI client
default_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

class ColumnRequest(BaseModel):
    column_name: str
    api_key: Optional[str] = None

@app.post("/suggest-column")
async def suggest_column(request: ColumnRequest):
    try:
        # Step 1: Check Excel mappings
        excel_mapping = header_mappings.get(request.column_name.lower())
        if excel_mapping:
            return {
                "suggested_name": excel_mapping, 
                "source": "excel_reference",
                "confidence": 1.0  # Direct matches from Excel get 100% confidence
            }

        # Step 2: Check PDF reference with new processor
        if pdf_processor:
            pdf_result = pdf_processor.get_column_mapping(request.column_name)
            if pdf_result:
                return {
                    "suggested_name": pdf_result['mapping'],
                    "source": "pdf_reference",
                    "confidence": pdf_result['confidence']
                }

        # Step 3: If no mapping found, use Azure OpenAI
        client = default_client  # Using Azure OpenAI client
        
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),  # This should be your GPT-4 deployment name
            messages=[
                {
                    "role": "system",
                    "content": "You are a data naming expert. For each column name suggestion, provide both a standardized name and a confidence score (0.0-1.0) indicating how certain you are about the mapping."
                },
                {
                    "role": "user",
                    "content": f'Suggest a standardized column name for "{request.column_name}" following SOV naming standards. Return in format: "suggested_name|confidence_score"'
                }
            ],
            temperature=0.2,
            max_tokens=50
        )

        # Parse the response to get both name and confidence
        result = response.choices[0].message.content.strip().split('|')
        suggested_name = result[0].strip()
        confidence = float(result[1]) if len(result) > 1 else 0.5

        return {
            "suggested_name": suggested_name, 
            "source": "openai",
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Add endpoint to refresh mappings
@app.post("/refresh-mappings")
async def refresh_mappings():
    """Endpoint to reload the reference mappings"""
    try:
        load_reference_mappings()
        return {"status": "success", "message": "Mappings refreshed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing mappings: {str(e)}")
