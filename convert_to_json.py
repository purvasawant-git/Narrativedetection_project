import os
import pandas as pd
import json

def convert_excel_to_json():
    excel_path = "data/project_outputs.xlsx"
    output_dir = "frontend/data"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading {excel_path}...")
    
    # Read all sheets
    xls = pd.ExcelFile(excel_path)
    
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Convert datetime columns to string to avoid serialization issues
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)
                
        # Fill NA values with None so JSON becomes null
        df = df.where(pd.notnull(df), None)
        
        json_path = os.path.join(output_dir, f"{sheet_name}.json")
        
        # Orient 'records' is best for JavaScript fetching (list of dictionaries)
        df.to_json(json_path, orient='records', indent=2)
        print(f"Saved {sheet_name} to {json_path}")

if __name__ == "__main__":
    convert_excel_to_json()
