import sys
import os

# Add the project root to sys.path so ai_model can be imported from backend/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import allowed_file, secure_filename_with_uuid
import pandas as pd
from ai_model.image_preprocessing import load_and_preprocess_image
from tensorflow.keras.models import load_model

# ====== HYBRID CONFIGURATION ======
CONFIDENCE_THRESHOLD = 0.75  # Adjust as needed
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ai_model', 'plant_model.keras'))

class PlantSpeciesIdentifier:
    def __init__(self, model_path, class_indices):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        if not class_indices:
            raise ValueError("class_indices cannot be empty")
        self.model = load_model(model_path)
        self.class_indices = class_indices
        self.class_names = {v: k for k, v in class_indices.items()}

    def predict(self, image_path):
        img = load_and_preprocess_image(image_path)
        if img is None:
            raise ValueError(f"Could not process image: {image_path}")
        img = np.expand_dims(img, axis=0)
        preds = self.model.predict(img, verbose=0)
        class_id = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        species = self.class_names[class_id]
        return species, confidence

# --- Data Loads ---
with open(os.path.join(os.path.dirname(__file__), 'conservation_data.json'), encoding='utf-8') as f:
    CONSERVATION_DATA = json.load(f)

with open(os.path.join(os.path.dirname(__file__), 'class_indices.json'), encoding='utf-8') as f:
    CLASS_INDICES = json.load(f)

IUCN_DATA = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'taxon.txt'),
    sep='\t',
    usecols=['scientificName', 'iucnReference'],
    dtype=str,
    na_filter=False,
    header=0
).set_index('scientificName')['iucnReference'].to_dict()

print(f"Loaded {len(IUCN_DATA)} species from IUCN database")

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5500", "http://127.0.0.1:5500"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

app.config.update({
    'UPLOAD_FOLDER': 'uploads',
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,
    'ALLOWED_EXTENSIONS': {'png', 'jpg', 'jpeg', 'webp'}
})
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

PLANTNET_API_KEY = "2b10acfLLkhlHgUAN4JXMJkI"

def identify_with_plantnet(image_paths):
    url = f"https://my-api.plantnet.org/v2/identify/all?api-key={PLANTNET_API_KEY}"
    files = []
    try:
        for path in image_paths:
            files.append(('images', open(path, 'rb')))
        data = {'organs': ['leaf'] * len(image_paths)}
        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {'error': str(e)}
    finally:
        for file in files:
            file[1].close()

def search_iucn_species(species_name):
    print(f"🔍 Searching IUCN for: '{species_name}'")
    if species_name in IUCN_DATA:
        print(f"✅ Found exact match for '{species_name}'")
        return IUCN_DATA[species_name]
    if '(' in species_name:
        clean_name = species_name.split(' (')[0].strip()
        print(f"🔍 Trying without authorship: '{clean_name}'")
        if clean_name in IUCN_DATA:
            print(f"✅ Found match without authorship: '{clean_name}'")
            return IUCN_DATA[clean_name]
    parts = species_name.split()
    if len(parts) >= 2:
        genus_species = f"{parts[0]} {parts[1]}"
        print(f"🔍 Trying genus + species: '{genus_species}'")
        if genus_species in IUCN_DATA:
            print(f"✅ Found genus + species match: '{genus_species}'")
            return IUCN_DATA[genus_species]
    similar_species = [name for name in IUCN_DATA.keys() if name.lower().startswith(species_name.lower()[:10])]
    if similar_species:
        print(f"📋 Found {len(similar_species)} similar species starting with '{species_name[:10]}'")
        return IUCN_DATA[similar_species[0]]
    print(f"❌ No match found for '{species_name}' in IUCN database")
    return None

# --- Initialize Local Model ---
local_identifier = PlantSpeciesIdentifier(MODEL_PATH, CLASS_INDICES)

@app.route('/')
def home():
    return jsonify({'message': 'Plant Conservation API is running'})

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Backend is running'})

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200

    temp_files = []
    try:
        files = request.files.getlist('images')
        if not files:
            return jsonify({'error': 'No images uploaded'}), 400
        if len(files) > 5:
            return jsonify({'error': 'Maximum 5 images allowed'}), 400

        # Validate and save files
        for file in files:
            if file.filename == '':
                return jsonify({'error': 'Empty filename detected'}), 400
            if not allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
                return jsonify({'error': f'Invalid file type: {file.filename}'}), 400
            filename = secure_filename_with_uuid(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            temp_files.append(filepath)

        # --- HYBRID LOGIC: Try local model first ---
        local_species, local_confidence = local_identifier.predict(temp_files[0])
        print(f"Local model prediction: {local_species} (confidence: {local_confidence:.2f})")
        if local_species in CLASS_INDICES and local_confidence >= CONFIDENCE_THRESHOLD:
            best_species = local_species
            confidence = local_confidence
            common_name = best_species
            print("✅ Using local model prediction.")
            # For local model, no common_names or gbif_url available
            common_names = []
            gbif_url = ""
        else:
            # Fallback to Pl@ntNet API
            print("⚠️ Local model confidence too low or species not in local set. Using Pl@ntNet API.")
            api_result = identify_with_plantnet(temp_files)
            print("Pl@ntNet API Response:", json.dumps(api_result, indent=2))

            if 'error' in api_result:
                return jsonify({'error': api_result['error']}), 500
            if 'results' not in api_result or not api_result['results']:
                return jsonify({'error': 'No plant identified.'}), 404

            # Aggregate results
            species_scores = {}
            for result in api_result['results']:
                species = result['species'].get('scientificNameWithoutAuthor', 'Unknown')
                score = result.get('score', 0)
                if species:
                    species_scores[species] = species_scores.get(species, 0) + score

            if not species_scores:
                return jsonify({'error': 'No identifiable features found'}), 404

            best_species = max(species_scores, key=species_scores.get)
            confidence = species_scores[best_species] / len(files)

            print(f"🌱 Best identified species: '{best_species}' (confidence: {confidence:.3f})")

            # Get common name from class indices
            common_name = next(
                (cn for cn in CLASS_INDICES if cn.lower() == best_species.lower()),
                best_species
            )

            # Get additional details
            try:
                best_result = next(
                    r for r in api_result['results']
                    if r['species'].get('scientificNameWithoutAuthor') == best_species
                )
            except StopIteration:
                return jsonify({'error': 'Species details not found'}), 404

            common_names = best_result['species'].get('commonNames', [])
            gbif_url = best_result['species'].get('gbif', '')

        # --- Conservation lookup ---
        cons_status = "Unknown"
        cons_description = "No conservation data available"
        cons_url = gbif_url if 'gbif_url' in locals() else ""

        # 1. Check local conservation data
        search_terms = {common_name.lower(), best_species.lower()}
        if common_names:
            search_terms.update(name.lower() for name in common_names)
        
        print(f"🔍 Searching local conservation data for: {search_terms}")
        
        for key in CONSERVATION_DATA:
            if key.lower() in search_terms:
                cons_status = CONSERVATION_DATA[key]['status']
                cons_description = CONSERVATION_DATA[key]['description']
                cons_url = CONSERVATION_DATA[key]['details_url']
                print(f"✅ Found in local conservation data: {key} -> {cons_status}")
                break

        # 2. Enhanced IUCN data fallback
        if cons_status == "Unknown":
            print("🔍 Searching IUCN database...")
            iucn_ref = search_iucn_species(best_species)
            if iucn_ref:
                cons_status = "See IUCN Red List"
                cons_description = "Found in IUCN Red List - click for conservation status details"
                cons_url = iucn_ref
                print(f"✅ Found IUCN reference: {iucn_ref}")
            else:
                print("❌ Species not found in any conservation database")

        response_data = {
            'species': common_name,
            'scientific_name': best_species,
            'confidence': round(confidence, 4),
            'conservation_status': cons_status,
            'description': cons_description,
            'details_url': cons_url
        }
        print(f"📤 Response: {response_data}")
        return jsonify(response_data)

    except ValueError as ve:
        app.logger.error(f'Value error: {str(ve)}')
        return jsonify({'error': 'Identification failed due to invalid data'}), 500
    except Exception as e:
        app.logger.error(f'Prediction error: {str(e)}')
        return jsonify({'error': 'Plant identification failed', 'details': str(e)}), 500
    finally:
        # Cleanup temporary files
        for path in temp_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                app.logger.warning(f'Failed to remove {path}: {str(e)}')

if __name__ == "__main__":
    print("🚀 Starting Flask backend...")
    app.run(host='0.0.0.0', port=5000, debug=True)
