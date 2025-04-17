from flask import Flask, request, render_template, send_file, jsonify, redirect, url_for
import pandas as pd
import json
import google.generativeai as genai
import whisper
from datetime import datetime
import os
from fuzzywuzzy import fuzz
from typing import List, Dict, Any
import werkzeug
import tempfile

app = Flask(__name__)

# Configuration constants
MEDICAL_CODES_EXCEL = "medical_codes.xlsx"
OUTPUT_DIR = "output"
JSON_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "json")
UPLOAD_FOLDER = "uploads"
API_KEY = "AIzaSyDPmukhY7Ejs9TEwaRyxtCMiTZVAsJC2dk"  # Replace with your key

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global whisper model
whisper_model = None

def initialize_whisper():
    global whisper_model
    if whisper_model is None:
        print("Loading Whisper model...")
        whisper_model = whisper.load_model("base")
    return whisper_model

# Initialize Google Gemini
def initialize_gemini():
    genai.configure(api_key=API_KEY)
    return genai.GenerativeModel('gemini-1.5-pro')

def initialize_default_codes():
    """Create default medical codes file if it doesn't exist"""
    if not os.path.exists(MEDICAL_CODES_EXCEL):
        print("Creating default medical_codes.xlsx file...")
        default_codes = [
            {
                "Term": "complete blood count",
                "Description": "Complete Blood Count",
                "Code": "LAB023",
                "Type": "lab_test",
                "Alternate Terms": "CBC, blood panel"
            },
            # Add more default codes as needed
        ]
        pd.DataFrame(default_codes).to_excel(MEDICAL_CODES_EXCEL, index=False)

def load_medical_codes() -> dict:
    """Load medical codes from Excel with validation"""
    try:
        if not os.path.exists(MEDICAL_CODES_EXCEL):
            raise FileNotFoundError(f"Medical codes file not found at {MEDICAL_CODES_EXCEL}")

        df = pd.read_excel(MEDICAL_CODES_EXCEL)
        required_cols = {'Term', 'Description', 'Code', 'Type'}

        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        codes_db = {}
        for _, row in df.iterrows():
            primary_term = str(row['Term']).lower().strip()
            codes_db[primary_term] = {
                'description': row['Description'],
                'code': row['Code'],
                'type': row['Type']
            }

            if 'Alternate Terms' in df.columns and pd.notna(row['Alternate Terms']):
                for alt_term in str(row['Alternate Terms']).split(','):
                    alt_term_clean = alt_term.strip().lower()
                    if alt_term_clean:  # Ensure not empty
                        codes_db[alt_term_clean] = codes_db[primary_term]

        print(f"Loaded {len(codes_db)} medical codes")
        return codes_db

    except Exception as e:
        print(f"Error loading medical codes: {e}")
        return {}

def extract_medical_phrases(model, text):
    """Use Gemini to extract medical phrases from text"""
    prompt = f"""
    Extract all medical terms, procedures, diagnoses, and treatments from the following text.
    Return the results as a JSON array with objects having these fields:
    - "phrase": the exact medical phrase
    - "category": the category (diagnosis, procedure, medication, etc.)
    - "confidence": a number between 0-1 indicating confidence
    
    Text: {text}
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Find JSON array in response
        start = response_text.find('[')
        end = response_text.rfind(']') + 1
        
        if start >= 0 and end > start:
            json_str = response_text[start:end]
            parsed_json = json.loads(json_str)
            if isinstance(parsed_json, list):
                return parsed_json
        
        # Fallback if proper JSON wasn't found
        return [{"phrase": text, "category": "unknown", "confidence": 0.5}]
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        return [{"phrase": text, "category": "unknown", "confidence": 0.5}]

def match_phrase_to_code(phrase, codes_db):
    """Match a medical phrase to the codes database using fuzzy matching"""
    if not phrase:
        return {
            'matched_term': None,
            'code': None,
            'description': None,
            'type': None,
            'match_score': 0
        }
        
    best_match = None
    best_score = 0
    phrase_lower = phrase.lower()
    
    for term, details in codes_db.items():
        score = fuzz.ratio(phrase_lower, term)
        if score > best_score and score > 70:  # Only match if similarity > 70%
            best_score = score
            best_match = {
                'matched_term': term,
                'code': details['code'],
                'description': details['description'],
                'type': details['type'],
                'match_score': score
            }
    
    if best_match:
        return best_match
    else:
        return {
            'matched_term': None,
            'code': None,
            'description': None,
            'type': None,
            'match_score': 0
        }

def save_outputs(results: List[Dict[str, Any]], base_filename: str):
    """Save results to both Excel and JSON formats"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename_base = f"{base_filename}_{timestamp}"

    # Save to Excel
    excel_path = os.path.join(OUTPUT_DIR, f"{filename_base}.xlsx")
    df = pd.DataFrame(results)
    df.to_excel(excel_path, index=False)
    print(f"Excel report saved to: {excel_path}")

    # Save to JSON
    json_path = os.path.join(JSON_OUTPUT_DIR, f"{filename_base}.json")
    with open(json_path, 'w') as f:
        json.dump({
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'source_audio': base_filename,
                'total_matches': len(results)
            },
            'results': results
        }, f, indent=2)
    print(f"JSON results saved to: {json_path}")

    return excel_path, json_path

def process_audio(audio_path: str):
    """Main processing pipeline"""
    # 1. Initialize and validate
    initialize_default_codes()
    codes_db = load_medical_codes()
    if not codes_db:
        raise ValueError("No medical codes loaded - cannot continue")

    # 2. Transcribe audio
    print(f"Transcribing {audio_path}...")
    whisper_model = initialize_whisper()
    try:
        transcription = whisper_model.transcribe(audio_path)
        medical_text = transcription["text"]
    except Exception as e:
        raise ValueError(f"Audio transcription failed: {str(e)}")

    # 3. Extract medical phrases
    gemini_model = initialize_gemini()
    extracted_phrases = extract_medical_phrases(gemini_model, medical_text)

    # 4. Match to codes and prepare results
    results = []
    for phrase in extracted_phrases:
        phrase_text = phrase.get('phrase', '')
        match = match_phrase_to_code(phrase_text, codes_db)
        results.append({
            **phrase,
            **match,
            'source_text': medical_text,
            'timestamp': datetime.now().isoformat()
        })

    # 5. Save outputs
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    return save_outputs(results, base_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    temp_dir = None
    
    try:
        if file.filename == '':
            # Handle browser-recorded audio which might not have a filename
            if file.content_type and file.content_type.startswith('audio/'):
                # Create a temporary file for the recording
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, 'recording.wav')
                file.save(temp_path)
                filepath = temp_path
                filename = 'browser_recording'
            else:
                return jsonify({'error': 'No selected file or invalid file type'}), 400
        else:
            # Handle regular file upload
            filename = werkzeug.utils.secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            filename = os.path.splitext(filename)[0]
        
        excel_file, json_file = process_audio(filepath)
        
        # Read the JSON file to send back to the client
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        return jsonify({
            'success': True,
            'excel_file': os.path.basename(excel_file),
            'json_file': os.path.basename(json_file),
            'results': json_data
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up temp files if created
        if temp_dir:
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Error cleaning temp directory: {e}")

@app.route('/download/<filetype>/<filename>')
def download_file(filetype, filename):
    # Validate filename to prevent directory traversal
    filename = werkzeug.utils.secure_filename(filename)
    
    if filetype == 'excel':
        file_path = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        return send_file(file_path, as_attachment=True)
    
    elif filetype == 'json':
        file_path = os.path.join(JSON_OUTPUT_DIR, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        return send_file(file_path, as_attachment=True)
    
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/codes')
def view_codes():
    try:
        if not os.path.exists(MEDICAL_CODES_EXCEL):
            initialize_default_codes()
            
        codes_df = pd.read_excel(MEDICAL_CODES_EXCEL)
        codes = codes_df.to_dict('records')
        return render_template('codes.html', codes=codes)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/add_code', methods=['GET', 'POST'])
def add_code():
    if request.method == 'POST':
        try:
            # Validate form data
            term = request.form.get('term', '').strip()
            description = request.form.get('description', '').strip()
            code = request.form.get('code', '').strip()
            type_val = request.form.get('type', '').strip()
            
            if not all([term, description, code, type_val]):
                return render_template('error.html', 
                                      error="All required fields (Term, Description, Code, Type) must be filled")
            
            # Load existing codes
            if os.path.exists(MEDICAL_CODES_EXCEL):
                codes_df = pd.read_excel(MEDICAL_CODES_EXCEL)
            else:
                codes_df = pd.DataFrame(columns=['Term', 'Description', 'Code', 'Type', 'Alternate Terms'])
            
            # Add new code
            new_code = {
                'Term': term,
                'Description': description,
                'Code': code,
                'Type': type_val,
                'Alternate Terms': request.form.get('alternate_terms', '').strip()
            }
            
            # Append and save
            codes_df = pd.concat([codes_df, pd.DataFrame([new_code])], ignore_index=True)
            codes_df.to_excel(MEDICAL_CODES_EXCEL, index=False)
            
            return redirect(url_for('view_codes'))
        except Exception as e:
            return render_template('error.html', error=str(e))
    
    return render_template('add_code.html')

if __name__ == '__main__':
    # Initialize medical codes at startup
    initialize_default_codes()
    app.run(debug=True)
