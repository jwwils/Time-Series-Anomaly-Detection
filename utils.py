import pandas as pd
import os

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_uploaded_file(filepath):
    if filepath.endswith('.csv'):
        data = pd.read_csv(filepath)
    elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        data = pd.read_excel(filepath)
    elif filepath.endswith('.json'):
        data = pd.read_json(filepath)
    else:
        raise ValueError("Unsupported file type")
    return data