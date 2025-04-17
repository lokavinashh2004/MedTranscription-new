from flask import Flask, request, render_template, send_file, jsonify, redirect, url_for
import pandas as pd
import json
import os
import logging
import tempfile
import werkzeug
import traceback
from datetime import datetime
from typing import List, Dict, Any
from fuzzywuzzy import fuzz

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get environment variables or use defaults
PORT = int(os.environ.get("PORT", 5000))
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDPmukhY7Ejs9TEwaRyxtCMiTZVAsJC2dk")  # Set this in Render environment variables

# Configuration constants
MEDICAL_CODES_EXCEL = os.environ.get("MEDICAL_CODES_PATH", "medical_codes.xlsx")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
JSON_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "json")
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "tiny")  # Use tiny model to reduce resource needs

# Global variables for lazy loading
whisper_model = None
gemini_model = None

# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist"""
    try:
        for directory in [OUTPUT_DIR, JSON_OUTPUT_DIR, UPLOAD_FOLDER]:
            os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directories: {OUTPUT_DIR}, {JSON_OUTPUT_DIR}, {UPLOAD_FOLDER}")
    except Exception as e:
        logger.error(f"Failed to create directories: {str(e)}")
        raise

def initialize_whisper():
    """Initialize Whisper model only when needed"""
    global whisper_model
    if whisper_model is None:
        try:
            import whisper
            logger.info(f"Loading Whisper model ({WHISPER_MODEL_SIZE})...")
            whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
            logger.info("Whisper model loaded successfully")
            return whisper_model
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {str(e)}")
            raise
    return whisper_model

def initialize_gemini():
    """Initialize Google Gemini only when needed"""
    global gemini_model
    if gemini_model is None:
        try:
            import google.generativeai as genai
            logger.info("Initializing Google Gemini...")
            genai.configure(api_key=API_KEY)
            gemini_model = genai.GenerativeModel('gemini-1.5-pro')
            logger.info("Google Gemini initialized successfully")
            return gemini_model
        except Exception as e:
            logger.error(f"Failed to initialize Google Gemini: {str(e)}")
            raise
    return gemini_model

def initialize_default_codes():
    """Create default medical codes file if it doesn't exist"""
    try:
        if not os.path.exists(MEDICAL_CODES_EXCEL):
            logger.info(f"Creating default {MEDICAL_CODES_EXCEL} file...")
            default_codes = [
                {
                    "Term": "complete blood count",
                    "Description": "Complete Blood Count",
                    "Code": "LAB023",
                    "Type": "lab_test",
                    "Alternate Terms": "CBC, blood panel"
                },
                {
                    "Term": "upper respiratory infection",
                    "Description": "Upper Respiratory Infection",
                    "Code": "J06.9",
                    "Type": "diagnosis",
                    "Alternate Terms": "URI, common cold"
                },
                {
                    "Term": "amoxicillin",
                    "Description": "Amoxicillin",
                    "Code": "MED001",
                    "Type": "medication",
                    "Alternate Terms": "amox"
                },
                # Add more default codes as needed
            ]
            df = pd.DataFrame(default_codes)
            df.to_excel(MEDICAL_CODES_EXCEL, index=False)
            logger.info(f"Default codes created in {MEDICAL_CODES_EXCEL}")
    except Exception as e:
        logger.error(f"Failed to create default codes: {str(e)}")
        raise

def load_medical_codes() -> dict:
    """Load medical codes from Excel with validation"""
    try:
        if not os.path.exists(MEDICAL_CODES_EXCEL):
            logger.warning(f"Medical codes file not found at {MEDICAL_CODES_EXCEL}, creating default codes")
            initialize_default_codes()

        df = pd.read_excel(MEDICAL_CODES_EXCEL)
        required_cols = {'Term', 'Description', 'Code', 'Type'}

        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            logger.error(f"Missing required columns in {MEDICAL_CODES_EXCEL}: {missing}")
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

        logger.info(f"Loaded {len(codes_db)} medical codes")
        return codes_db

    except Exception as e:
        logger.error(f"Error loading medical codes: {str(e)}")
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
        logger.info("Sending text to Gemini for medical phrase extraction")
        response = model.generate_content(prompt)
        response_text = response.text
        logger.info("Received response from Gemini")
        
        # Find JSON array in response
        start = response_text.find('[')
        end = response_text.rfind(']') + 1
        
        if start >= 0 and end > start:
            json_str = response_text[start:end]
            try:
                parsed_json = json.loads(json_str)
                if isinstance(parsed_json, list):
                    logger.info(f"Successfully extracted {len(parsed_json)} medical phrases")
                    return parsed_json
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON parsing error: {str(json_err)}")
        
        # Fallback if proper JSON wasn't found
        logger.warning("Could not extract proper JSON from Gemini response, using fallback")
        return [{"phrase": text, "category": "unknown", "confidence": 0.5}]
    except Exception as e:
        logger.error(f"Error in Gemini medical phrase extraction: {str(e)}")
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
        logger.debug(f"Matched '{phrase}' to '{best_match['matched_term']}' with score {best_match['match_score']}")
        return best_match
    else:
        logger.debug(f"No match found for '{phrase}'")
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

    try:
        # Save to Excel
        excel_path = os.path.join(OUTPUT_DIR, f"{filename_base}.xlsx")
        df = pd.DataFrame(results)
        df.to_excel(excel_path, index=False)
        logger.info(f"Excel report saved to: {excel_path}")

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
        logger.info(f"JSON results saved to: {json_path}")

        return excel_path, json_path
    except Exception as e:
        logger.error(f"Error saving outputs: {str(e)}")
        raise

def process_audio(audio_path: str):
    """Main processing pipeline"""
    try:
        # 1. Initialize and validate
        logger.info(f"Processing audio file: {audio_path}")
        codes_db = load_medical_codes()
        if not codes_db:
            raise ValueError("No medical codes loaded - cannot continue")

        # 2. Transcribe audio
        logger.info(f"Transcribing {audio_path}...")
        model = initialize_whisper()
        try:
            transcription = model.transcribe(audio_path)
            medical_text = transcription["text"]
            logger.info(f"Transcription complete: {len(medical_text)} characters")
        except Exception as e:
            logger.error(f"Audio transcription failed: {str(e)}")
            raise ValueError(f"Audio transcription failed: {str(e)}")

        # 3. Extract medical phrases
        logger.info("Extracting medical phrases...")
        model = initialize_gemini()
        extracted_phrases = extract_medical_phrases(model, medical_text)

        # 4. Match to codes and prepare results
        logger.info("Matching phrases to medical codes...")
        results = []
        for phrase in extracted_phrases:
            phrase_text = phrase.get('phrase', '')
            match = match_phrase_to_code(phrase_text, codes_db)
            results.append({
                **phrase,
                **match,
                'source_text': medical_text[:200] + ('...' if len(medical_text) > 200 else ''),  # Truncate long text
                'timestamp': datetime.now().isoformat()
            })

        # 5. Save outputs
        logger.info("Saving results...")
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        return save_outputs(results, base_name)
    
    except Exception as e:
        logger.error(f"Error in process_audio: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    logger.info("Upload endpoint called")
    
    try:
        if 'file' not in request.files:
            logger.warning("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        temp_dir = None
        
        if file.filename == '':
            # Handle browser-recorded audio which might not have a filename
            if file.content_type and file.content_type.startswith('audio/'):
                logger.info("Processing browser recording")
                # Create a temporary file for the recording
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, 'recording.wav')
                file.save(temp_path)
                filepath = temp_path
                filename = 'browser_recording'
            else:
                logger.warning("No selected file or invalid file type")
                return jsonify({'error': 'No selected file or invalid file type'}), 400
        else:
            # Handle regular file upload
            logger.info(f"Processing uploaded file: {file.filename}")
            filename = werkzeug.utils.secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            filename = os.path.splitext(filename)[0]
        
        logger.info(f"File saved, starting processing: {filepath}")
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
        error_trace = traceback.format_exc()
        logger.error(f"Error in upload_file: {str(e)}\n{error_trace}")
        return jsonify({
            'error': str(e),
            'trace': error_trace.split('\n')  # Split into lines for better display
        }), 500
    
    finally:
        # Clean up temp files if created
        if temp_dir:
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning temp directory: {str(e)}")

@app.route('/download/<filetype>/<filename>')
def download_file(filetype, filename):
    logger.info(f"Download requested for {filetype}/{filename}")
    
    # Validate filename to prevent directory traversal
    filename = werkzeug.utils.secure_filename(filename)
    
    try:
        if filetype == 'excel':
            file_path = os.path.join(OUTPUT_DIR, filename)
            if not os.path.exists(file_path):
                logger.warning(f"Excel file not found: {file_path}")
                return jsonify({'error': 'File not found'}), 404
            logger.info(f"Sending excel file: {file_path}")
            return send_file(file_path, as_attachment=True)
        
        elif filetype == 'json':
            file_path = os.path.join(JSON_OUTPUT_DIR, filename)
            if not os.path.exists(file_path):
                logger.warning(f"JSON file not found: {file_path}")
                return jsonify({'error': 'File not found'}), 404
            logger.info(f"Sending json file: {file_path}")
            return send_file(file_path, as_attachment=True)
        
        else:
            logger.warning(f"Invalid file type requested: {filetype}")
            return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        logger.error(f"Error in download_file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/codes')
def view_codes():
    try:
        logger.info("Codes page requested")
        if not os.path.exists(MEDICAL_CODES_EXCEL):
            logger.info("Medical codes file not found, initializing default codes")
            initialize_default_codes()
            
        codes_df = pd.read_excel(MEDICAL_CODES_EXCEL)
        codes = codes_df.to_dict('records')
        return render_template('codes.html', codes=codes)
    except Exception as e:
        logger.error(f"Error in view_codes: {str(e)}")
        return render_template('error.html', error=str(e))

@app.route('/add_code', methods=['GET', 'POST'])
def add_code():
    if request.method == 'POST':
        try:
            logger.info("Processing add_code form submission")
            # Validate form data
            term = request.form.get('term', '').strip()
            description = request.form.get('description', '').strip()
            code = request.form.get('code', '').strip()
            type_val = request.form.get('type', '').strip()
            
            if not all([term, description, code, type_val]):
                logger.warning("Missing required fields in add_code form")
                return render_template('error.html', 
                                      error="All required fields (Term, Description, Code, Type) must be filled")
            
            # Load existing codes
            if os.path.exists(MEDICAL_CODES_EXCEL):
                codes_df = pd.read_excel(MEDICAL_CODES_EXCEL)
            else:
                logger.info("Creating new medical codes file")
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
            logger.info(f"Added new code: {code} - {term}")
            
            return redirect(url_for('view_codes'))
        except Exception as e:
            logger.error(f"Error in add_code POST: {str(e)}")
            return render_template('error.html', error=str(e))
    
    logger.info("Add code form requested")
    return render_template('add_code.html')

# Add a health check endpoint for Render
@app.route('/health')
def health_check():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

@app.errorhandler(404)
def page_not_found(e):
    logger.warning(f"404 error: {request.path}")
    return jsonify({'error': 'Not found', 'path': request.path}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"500 error: {str(e)}")
    return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

# Initialize app
@app.before_first_request
def before_first_request():
    """Initialize necessary components before the first request"""
    try:
        logger.info("Initializing application...")
        ensure_directories()
        initialize_default_codes()
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")

if __name__ == '__main__':
    # Create a Procfile for Render
    with open("Procfile", "w") as f:
        f.write("web: gunicorn app:app")
    
    # Initialize directories and default codes
    ensure_directories()
    initialize_default_codes()
    
    # Run app
    app.run(host='0.0.0.0', port=PORT, debug=False)
