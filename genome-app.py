from flask import Flask, render_template, request, redirect, render_template_string, jsonify, url_for, jsonify
import joblib
import numpy as np
import pandas as pd
from pyngrok import ngrok
from sklearn.preprocessing import MinMaxScaler
import json
import gdown
import os

app = Flask(__name__)

# Define the directory to store models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Drive file IDs (replace with your actual file IDs)
MODEL_URLS = {
    "Genetic_Disorder_Stacked_Model.pkl": "https://drive.google.com/file/d/1-GBC7LW-Ofo-jf6Wp4noND0ysp3VrDpv/view?usp=drive_link",
    "Disorder_Subclass_Stacked_Model.pkl": "https://drive.google.com/file/d/1QlOlig0zDFIXdGixzctuBwUWjJx5K7Tk/view?usp=drive_link",
    "Combined_Genome_Disorder_Stacked_Model.pkl": "https://drive.google.com/file/d/1o_34CvFxlsH3Y03D0sYHWpPCVqOwAfWp/view?usp=drive_link",
}

# Download models if they don't exist locally
def download_models():
    for model_name, url in MODEL_URLS.items():
        model_path = os.path.join(MODEL_DIR, model_name)
        if not os.path.exists(model_path):
            print(f"Downloading {model_name} from Google Drive...")
            gdown.download(url, model_path, quiet=False)
        else:
            print(f"{model_name} already exists locally.")

# Load your trained models
def load_models():
    download_models()
    genetic_disorder_model = joblib.load(os.path.join(MODEL_DIR, "Genetic_Disorder_Stacked_Model.pkl"))
    disorder_subclass_model = joblib.load(os.path.join(MODEL_DIR, "Disorder_Subclass_Stacked_Model.pkl"))
    combined_disorder_model = joblib.load(os.path.join(MODEL_DIR, "Combined_Genome_Disorder_Stacked_Model.pkl"))
    return genetic_disorder_model, disorder_subclass_model, combined_disorder_model

# Load models at startup
genetic_disorder_model, disorder_subclass_model, combined_disorder_model = load_models()

# Load your trained models (replace with your actual paths)
# genetic_disorder_model = joblib.load('models\\Genetic_Disorder_Stacked_Model.pkl')
# disorder_subclass_model = joblib.load('models\\Disorder_Subclass_Stacked_Model.pkl')
# combined_disorder_model = joblib.load('models\\Combined_Genome_Disorder_Stacked_Model.pkl')

# Features lists for each model (in the order they were trained)
features_genetic = [
    "Blood cell count (mcL)", "White Blood cell count (thousand per microliter)", "Patient Age", "Father's age", "Mother's age", "No. of previous abortion", "Blood test result", "Gender", "Place of birth", "History of anomalies in previous pregnancies", "Folic acid details (peri-conceptional)", "Assisted conception IVF/ART", "Follow-up", "Heart Rate (rates/min)", "Respiratory Rate (breaths/min)", "Birth asphyxia", "Birth defects", "Status", "H/O radiation exposure (x-ray)", "H/O serious maternal illness", "H/O substance abuse", "Paternal gene", "Maternal gene", "Autopsy shows birth defect (if applicable)", "Genes in mother's side", "Inherited from father"
]

features_subclass = [
    "Blood cell count (mcL)", "White Blood cell count (thousand per microliter)", "Patient Age", "Father's age", "Mother's age", "No. of previous abortion", "Blood test result", "Gender", "History of anomalies in previous pregnancies", "Folic acid details (peri-conceptional)", "Place of birth", "Assisted conception IVF/ART", "Follow-up", "Heart Rate (rates/min)", "Birth asphyxia", "Respiratory Rate (breaths/min)", "H/O serious maternal illness", "H/O radiation exposure (x-ray)", "Birth defects", "Status", "H/O substance abuse", "Autopsy shows birth defect (if applicable)", "Maternal gene", "Paternal gene", "Genes in mother's side", "Inherited from father"
]

features_combined = [
    "Blood cell count (mcL)", "White Blood cell count (thousand per microliter)", "Patient Age", "Father's age", "Mother's age", "No. of previous abortion", "Blood test result", "Gender", "History of anomalies in previous pregnancies", "Folic acid details (peri-conceptional)", "Place of birth", "Assisted conception IVF/ART", "Follow-up", "Heart Rate (rates/min)", "Birth asphyxia", "Respiratory Rate (breaths/min)", "H/O serious maternal illness", "H/O radiation exposure (x-ray)", "Birth defects", "Status", "H/O substance abuse", "Autopsy shows birth defect (if applicable)", "Maternal gene", "Paternal gene", "Genes in mother's side", "Inherited from father"
]

# Mapping for categorical features (human-readable to numeric)
categorical_mapping = {
    "Genes in mother's side": {"No": 0, "Yes": 1},
    "Inherited from father": {"No": 0, "Yes": 1},
    "Maternal gene": {"No": 0, "Yes": 1},
    "Paternal gene": {"No": 0, "Yes": 1},
    "Status": {"Alive": 0, "Deceased": 1},
    "Respiratory Rate (breaths/min)": {"Normal": 0, "Tachypnea": 1},
    "Heart Rate (rates/min)": {"Normal": 0, "Tachycardia": 1},
    "Follow-up": {"High": 0, "Low": 1},
    "Gender": {"Female": 0, "Male": 1, "Prefer not to say": 2, "Other": 2},
    "Birth asphyxia": {"No": 0, "Yes": 1},
    "Autopsy shows birth defect (if applicable)": {"No": 0, "Yes": 1},
    "Place of birth": {"Home": 0, "Institute": 1},
    "Folic acid details (peri-conceptional)": {"No": 0, "Yes": 1},
    "History of serious maternal illness": {"No": 0, "Yes": 1},
    "H/O radiation exposure (x-ray)": {"No": 0, "Yes": 1},
    "H/O substance abuse": {"No": 0, "Yes": 1},
    "H/O serious maternal illness": {"No": 0, "Yes": 1},
    "Assisted conception IVF/ART": {"No": 0, "Yes": 1},
    "History of anomalies in previous pregnancies": {"No": 0, "Yes": 1},
    "No. of previous abortion": {"0": 0.0, "1": 0.25, "2": 0.5, "3": 0.75, "4": 1},
    "Birth defects": {"Multiple": 0, "Singular": 1},
    "Blood test result": {"Abnormal": 0, "Inconclusive": 1, "Normal": 2, "Slightly abnormal": 3}
}

# Mapping for categorical features (human-readable to numeric) for file upload
categorical_mapping_upload = {
    "Genes in mother's side": {"No": 0, "Yes": 1},
    "Inherited from father": {"No": 0, "Yes": 1},
    "Maternal gene": {"No": 0, "Yes": 1},
    "Paternal gene": {"No": 0, "Yes": 1},
    "Status": {"Alive": 0, "Deceased": 1},
    "Respiratory Rate (breaths/min)": {"Normal": 0, "Tachypnea": 1},
    "Heart Rate (rates/min)": {"Normal": 0, "Tachycardia": 1},
    "Follow-up": {"High": 0, "Low": 1},
    "Gender": {"Female": 0, "Male": 1, "Prefer not to say": 2, "Other": 2},
    "Birth asphyxia": {"No": 0, "Yes": 1},
    "Autopsy shows birth defect (if applicable)": {"No": 0, "Yes": 1},
    "Place of birth": {"Home": 0, "Institute": 1},
    "Folic acid details (peri-conceptional)": {"No": 0, "Yes": 1},
    "History of serious maternal illness": {"No": 0, "Yes": 1},
    "H/O radiation exposure (x-ray)": {"No": 0, "Yes": 1},
    "H/O substance abuse": {"No": 0, "Yes": 1},
    "H/O serious maternal illness": {"No": 0, "Yes": 1},
    "Assisted conception IVF/ART": {"No": 0, "Yes": 1},
    "History of anomalies in previous pregnancies": {"No": 0, "Yes": 1},
    "No. of previous abortion": {"0": 0.0, "1": 0.25, "2": 0.5, "3": 0.75, "4": 1},
    "Birth defects": {"Multiple": 0, "Singular": 1},
    "Blood test result": {"Abnormal": 0, "Inconclusive": 1, "Normal": 2, "Slightly abnormal": 3}
}

# Mapping for categorical features for chatbot
categorical_mapping_chatbot = {
    "Genes in mother's side": {"No": 0, "Yes": 1},
    "Inherited from father": {"No": 0, "Yes": 1},
    "Maternal gene": {"No": 0, "Yes": 1},
    "Paternal gene": {"No": 0, "Yes": 1},
    "Status": {"Alive": 0, "Deceased": 1},
    "Respiratory Rate (breaths/min)": {"Normal": 0, "Tachypnea": 1},
    "Heart Rate (rates/min)": {"Normal": 0, "Tachycardia": 1},
    "Follow-up": {"High": 0, "Low": 1},
    "Gender": {"Female": 0, "Male": 1, "Prefer not to say": 2, "Other": 2},
    "Birth asphyxia": {"No": 0, "Yes": 1},
    "Autopsy shows birth defect (if applicable)": {"No": 0, "Yes": 1},
    "Place of birth": {"Home": 0, "Institute": 1},
    "Folic acid details (peri-conceptional)": {"No": 0, "Yes": 1},
    "History of serious maternal illness": {"No": 0, "Yes": 1},
    "H/O radiation exposure (x-ray)": {"No": 0, "Yes": 1},
    "H/O substance abuse": {"No": 0, "Yes": 1},
    "H/O serious maternal illness": {"No": 0, "Yes": 1},
    "Assisted conception IVF/ART": {"No": 0, "Yes": 1},
    "History of anomalies in previous pregnancies": {"No": 0, "Yes": 1},
    "No. of previous abortion": {"0": 0.0, "1": 0.25, "2": 0.5, "3": 0.75, "4": 1},
    "Birth defects": {"Multiple": 0, "Singular": 1},
    "Blood test result": {"Abnormal": 0, "Inconclusive": 1, "Normal": 2, "Slightly abnormal": 3}
}

# Numerical feature ranges for inverse scaling
numerical_ranges = {
    "Blood cell count (mcL)": (4.2998825, 5.497859656),
    "White Blood cell count (thousand per microliter)": (3.0, 12.0),
    "No. of previous abortion": (0, 4) # For inverse transform, use original range
}

def transform_input(inputs, current_features):
    numeric_values = []
    for i, feature in enumerate(current_features):
        value = inputs[i]
        if feature in categorical_mapping:
            numeric = categorical_mapping[feature].get(value, None)
            if numeric is None:
                return f"Invalid value '{value}' for feature '{feature}'."
            numeric_values.append(numeric)
        elif feature in numerical_ranges:
            min_val, max_val = numerical_ranges[feature]
            try:
                numeric_value = (float(value) - min_val) / (max_val - min_val)
                numeric_values.append(numeric_value)
            except ValueError:
                return f"Invalid numerical input '{value}' for feature '{feature}'."
        else:
            try:
                numeric_values.append(float(value))
            except ValueError:
                return f"Invalid numerical input '{value}' for feature '{feature}'."
    return numeric_values

def transform_input_upload(row, current_features):
    numeric_values = []
    for feature in current_features:
        value = row[feature]
        if feature in categorical_mapping_upload:
            lookup_value = str(value).strip()  # Convert to string and remove leading/trailing whitespace
            numeric = categorical_mapping_upload[feature].get(lookup_value, None)
            if numeric is None:
                return f"Invalid value '{value}' for feature '{feature}'."
            numeric_values.append(numeric)
        elif feature in numerical_ranges:
            try:
                min_val, max_val = numerical_ranges[feature]
                numeric_value = (float(value) - min_val) / (max_val - min_val)
                numeric_values.append(numeric_value)
            except ValueError:
                return f"Invalid numerical input '{value}' for feature '{feature}'."
        else:
            try:
                numeric_values.append(float(value))
            except ValueError:
                return f"Invalid numerical input '{value}' for feature '{feature}'."
    return numeric_values

def transform_input_chatbot(inputs, current_features):
    numeric_values = []
    for feature in current_features:
        value = inputs[feature]
        if feature in categorical_mapping_chatbot:
            lookup_value = str(value).strip()  # Convert to string and remove leading/trailing whitespace
            numeric = categorical_mapping_chatbot[feature].get(lookup_value, None)
            if numeric is None:
                return f"Invalid value '{value}' for feature '{feature}'."
            numeric_values.append(numeric)
        elif feature in numerical_ranges:
            try:
                min_val, max_val = numerical_ranges[feature]
                numeric_value = (float(value) - min_val) / (max_val - min_val)
                numeric_values.append(numeric_value)
            except ValueError:
                return f"Invalid numerical input '{value}' for feature '{feature}'."
        else:
            try:
                numeric_values.append(float(value))
            except ValueError:
                return f"Invalid numerical input '{value}' for feature '{feature}'."
    return numeric_values

# Loading Page
@app.route('/')
def loading_page():
    return render_template('genome-loading.html')

# Dashboard Page
@app.route('/dashboard')
def main_home_dashboard():
    return render_template('genome-dashboard.html')

# Selection Model Page 
@app.route('/selection-model-form', methods=['GET', 'POST'])
def selection_model_form():
    if request.method == 'POST':
        choice = request.form['choice']
        return redirect(f'/detect/{choice}')

    return render_template('genome-selection-model.html')

# Form Input Detect Page
@app.route('/detect/<choice>', methods=['GET', 'POST'])
def detect(choice):
    if request.method == 'POST':
        if choice == 'genetic':
            current_features = features_genetic
            label_mapping = {0: "Mitochondrial genetic inheritance disorders", 1: "Multifactorial genetic inheritance disorders", 2: "No Genetic Disorder Detected", 3: "Single-gene inheritance diseases"}
        elif choice == 'subclass':
            current_features = features_subclass
            label_mapping = {0: "Alzheimer's", 1: "Cancer", 2: "Cystic fibrosis", 3: "Diabetes", 4: "Hemochromatosis", 5: "Leber's hereditary optic neuropathy", 6: "Leigh syndrome", 7: "Mitochondrial myopathy", 8: "No Disorder Subclass Detected", 9: "Tay-Sachs"}
        elif choice == 'combined':
            current_features = features_combined
            label_mapping = {0: "Mitochondrial genetic inheritance disorders - Leber's hereditary optic neuropathy", 1: "Mitochondrial genetic inheritance disorders - Leigh syndrome", 2: "Mitochondrial genetic inheritance disorders - Mitochondrial myopathy", 3: "Multifactorial genetic inheritance disorders - Alzheimer's", 4: "Multifactorial genetic inheritance disorders - Cancer", 5: "Multifactorial genetic inheritance disorders - Diabetes", 6: "No Genetic Disorder Detected - No Disorder Subclass Detected", 7: "Single-gene inheritance diseases - Cystic fibrosis", 8: "Single-gene inheritance diseases - Hemochromatosis", 9: "Single-gene inheritance diseases - Tay-Sachs"}
        else:
            return "Invalid Choice"

        data = [request.form[feature] for feature in current_features]
        numeric_data = transform_input(data, current_features)

        if isinstance(numeric_data, str):
            return render_template('genome-invalid-input.html', error_message=numeric_data)
        elif -1 in numeric_data:
            return render_template('genome-invalid-input.html')
        elif numeric_data is None:
            return render_template('genome-invalid-input.html')

        input_df = pd.DataFrame([numeric_data], columns=current_features)  # ✅ Use DataFrame to avoid warning

        if choice == 'genetic':
            detection = genetic_disorder_model.predict(input_df)[0]
        elif choice == 'subclass':
            detection = disorder_subclass_model.predict(input_df)[0]
        elif choice == 'combined':
            detection = combined_disorder_model.predict(input_df)[0]

        final_detection = label_mapping.get(detection, "Unknown Detection")
        return render_template('genome-clinical-remarks.html', final_detection=final_detection)

    # Render Form for inputs with a two-column layout (dynamic based on selected model)
    form_fields = '<div class="row">'
    if choice == 'genetic':
        current_features = features_genetic
    elif choice == 'subclass':
        current_features = features_subclass
    elif choice == 'combined':
        current_features = features_combined
    else:
        current_features = features_genetic # Default to genetic if no choice

    for i, feature in enumerate(current_features):
        col_class = 'col-md-6 mb-3'
        if i % 2 == 0:
            if i > 0:
                form_fields += '</div><div class="row">'

        if feature in categorical_mapping:
            options = '<option value="">-- Select --</option>'
            for human_val, _ in categorical_mapping[feature].items():
                options += f'<option value="{human_val}">{human_val}</option>'
            form_fields += f'<div class="{col_class}"><label for="{feature}">{feature}</label><select class="form-select" id="{feature}" name="{feature}" required>{options}</select></div>'
        else:
            placeholder_text = f"e.g., {numerical_ranges.get(feature, ('Enter', 'Value'))[0] if isinstance(numerical_ranges.get(feature, ('Enter', 'Value')), tuple) else numerical_ranges.get(feature, 'Enter Value')}" if feature in numerical_ranges else f"Enter {feature}"
            form_fields += f'<div class="{col_class}"><label for="{feature}">{feature}</label><input type="text" class="form-control" id="{feature}" name="{feature}" required placeholder="{placeholder_text}"></div>'

    form_fields += '</div>' # Close the last row

    return render_template('genome-patient-info-form.html', form_fields=form_fields)

# Selection Model Page for Upload Option
@app.route('/selection-model-upload', methods=['GET', 'POST'])
def selection_model_upload():
    if request.method == 'POST':
        choice = request.form['choice']
        return redirect(url_for('upload_page', selected_model=choice)) 
    return render_template('genome-selection-model.html')

@app.route('/upload/<selected_model>', methods=['GET'])
def upload_page(selected_model):
    expected_features = []
    if selected_model == 'genetic':
        expected_features = features_genetic
    elif selected_model == 'subclass':
        expected_features = features_subclass
    elif selected_model == 'combined':
        expected_features = features_combined
    return render_template('genome-upload.html', expected_features=expected_features, selected_model=selected_model)

# Upload Preview Route
@app.route('/upload-preview/<selected_model>', methods=['POST'])
def upload_preview(selected_model):
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        try:
            df = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)
            # Ensure the uploaded CSV/Excel has the required columns for the selected model
            if selected_model == 'genetic':
                current_features = features_genetic
            elif selected_model == 'subclass':
                current_features = features_subclass
            elif selected_model == 'combined':
                current_features = features_combined
            else:
                return jsonify({'error': 'Invalid model choice'})

            if not all(feature in df.columns for feature in current_features):
                missing_features = [feature for feature in current_features if feature not in df.columns]
                return jsonify({'error': f'Missing columns in file: {", ".join(missing_features)}'})

            # Return the first few rows as an HTML table
            html_table = df.head().to_html(index=False, classes='table table-striped')
            return jsonify({'html_table': html_table})
        except Exception as e:
            return jsonify({'error': f'Error reading file: {str(e)}'})
    return jsonify({'error': 'Invalid file type. Only CSV and Excel files are allowed.'})

# Handle File Upload and Detection
@app.route('/upload-and-detect/<selected_model>', methods=['POST'])
def upload_and_detect(selected_model):
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        try:
            df = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)

            if selected_model == 'genetic':
                current_features = features_genetic
                label_mapping = {0: "Mitochondrial genetic inheritance disorders", 1: "Multifactorial genetic inheritance disorders", 2: "No Genetic Disorder Detected", 3: "Single-gene inheritance diseases"}
                model = genetic_disorder_model
            elif selected_model == 'subclass':
                current_features = features_subclass
                label_mapping = {0: "Alzheimer's", 1: "Cancer", 2: "Cystic fibrosis", 3: "Diabetes", 4: "Hemochromatosis", 5: "Leber's hereditary optic neuropathy", 6: "Leigh syndrome", 7: "Mitochondrial myopathy", 8: "No Disorder Subclass Detected", 9: "Tay-Sachs"}
                model = disorder_subclass_model
            elif selected_model == 'combined':
                current_features = features_combined
                label_mapping = {0: "Mitochondrial genetic inheritance disorders - Leber's hereditary optic neuropathy", 1: "Mitochondrial genetic inheritance disorders - Leigh syndrome", 2: "Mitochondrial genetic inheritance disorders - Mitochondrial myopathy", 3: "Multifactorial genetic inheritance disorders - Alzheimer's", 4: "Multifactorial genetic inheritance disorders - Cancer", 5: "Multifactorial genetic inheritance disorders - Diabetes", 6: "No Genetic Disorder Detected - No Disorder Subclass Detected", 7: "Single-gene inheritance diseases - Cystic fibrosis", 8: "Single-gene inheritance diseases - Hemochromatosis", 9: "Single-gene inheritance diseases - Tay-Sachs"}
                model = combined_disorder_model
            else:
                return jsonify({'error': 'Invalid Choice'})

            if not all(feature in df.columns for feature in current_features):
                missing_features = [feature for feature in current_features if feature not in df.columns]
                return render_template('genome-invalid-upload.html', missing_features=missing_features)

            processed_data = []
            for index, row in df.iterrows():
                transformed_row = transform_input_upload(row, current_features)
                if isinstance(transformed_row, str):
                    return render_template('genome-invalid-input.html', error_message=f"Row {index + 1}: {transformed_row}")
                processed_data.append(transformed_row)

            predictions = model.predict(pd.DataFrame(processed_data, columns=current_features))
            final_detections = [label_mapping.get(pred, "Unknown Detection") for pred in predictions]

            print("Final Detections:", final_detections) # Add this line

            final_detections_str = ",".join(final_detections)
            original_data_json = json.dumps([row for row, _ in zip(df.to_dict('records'), final_detections)])

            return redirect(url_for('clinical_remarks_page', results=final_detections_str, original_data=original_data_json, selected_model=selected_model))

        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'})
    return jsonify({'error': 'Invalid file type. Only CSV and Excel files are allowed.'})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv', 'xls', 'xlsx'}

# Clinical Remarks Page
@app.route('/clinical-remarks')
def clinical_remarks_page():
    results_str = request.args.get('results')
    original_data_str = request.args.get('original_data')
    selected_model = request.args.get('selected_model')

    results = []
    if results_str:
        final_detections = results_str.split(',')
        if original_data_str:
            original_data_list = json.loads(original_data_str)
            results = list(zip(original_data_list, final_detections))

        print("Results in clinical_remarks:", results) # Add this line
        return render_template('genome-clinical-remarks.html', results=results, selected_model=selected_model)

@app.route('/selection-model-chatbot', methods=['GET', 'POST'])
def selection_model_chatbot():
    if request.method == 'POST':
        selected_model = request.form['choice']
        return redirect(url_for('chatbot_start', selected_model=selected_model))
    return render_template('genome-selection-model.html')

@app.route('/chatbot/start/<selected_model>', methods=['GET'])
def chatbot_start(selected_model):
    features = []
    questions_to_use = []  # Initialize a variable for the question list

    if selected_model == 'genetic':
        features = features_genetic
        questions_to_use = questions_genetic  # Use the genetic-specific questions
    elif selected_model == 'subclass':
        features = features_subclass
        questions_to_use = questions  # Use the existing 'questions' list
    elif selected_model == 'combined':
        features = features_combined
        questions_to_use = questions  # Use the existing 'questions' list

    if features:
        if len(features) == len(questions_to_use):
            welcome_message = """
            <strong>Hi there!</strong>
            <img src="https://media.giphy.com/media/hvRJCLFzcasrR4ia7z/giphy.gif" 
                 alt="wave" 
                 style="width: 25px; height: 25px; vertical-align: middle;" />
            Welcome to the Genome Detection Assistant. To detect the disorder, I’ll ask you a series of questions based on medical features. Let's begin.
            """

            first_question = questions_to_use[0]
            chat_history = [{'bot': welcome_message}, {'bot': first_question}]
            collected_answers = {}
            chat_history_json = json.dumps(chat_history)
            collected_answers_json = json.dumps(collected_answers)
            total_questions = len(questions_to_use)
            return render_template('genome-chatbot.html', chat_history=chat_history, selected_model=selected_model, current_question_index=0, collected_answers=collected_answers, chat_history_json=chat_history_json, collected_answers_json=collected_answers_json, current_features=json.dumps(features), total_questions=total_questions)
        else:
            return "Error: Number of features does not match the number of questions for the selected model."
    else:
        return "Invalid model selected."
    
questions_genetic = [
    "Please enter Blood cell count (mcL):",
    "Please enter White Blood cell count (thousand per microliter):",
    "Please enter Patient Age:",
    "Please enter Father's age:",
    "Please enter Mother's age:",
    "Please enter Number of previous abortions (0,1,2,3,4):",
    "Please enter Blood test result (Abnormal, Inconclusive, Normal, Slightly abnormal):",
    "Please enter Gender (Female, Male, Prefer not to say, Other):",
    "Please enter Place of birth (Home, Institute):",
    "Please enter History of anomalies in previous pregnancies (Yes, No):",
    "Please enter Folic acid details (peri-conceptional) [Yes, No]:",
    "Please enter Assisted conception IVF/ART (Yes, No):",
    "Please enter Follow-up (High, Low):",
    "Please enter Heart Rate (rates/min) [Normal, Tachycardia]:",
    "Please enter Respiratory Rate (breaths/min) [Normal, Tachypnea]:",
    "Please enter Birth asphyxia (Yes, No):",
    "Please enter Birth defects (Multiple, Singular):",
    "Please enter Status (Alive, Deceased):",
    "Please enter History of radiation exposure (x-ray) [Yes, No]:",
    "Please enter History of serious maternal illness (Yes, No):",
    "Please enter History of substance abuse (Yes, No):",
    "Please enter Paternal gene (Yes, No):",
    "Please enter Maternal gene (Yes, No):",
    "Please enter Autopsy shows birth defect (if applicable) [Yes, No]:",
    "Please enter Genes in mother's side (Yes, No):",
    "Please enter Inherited from father (Yes, No):"
]

# Questions and conversational prompts for chatbot
questions = [
    "Please enter Blood cell count (mcL):",
    "Please enter White Blood cell count (thousand per microliter):",
    "Please enter Patient Age:",
    "Please enter Father's age:",
    "Please enter Mother's age:",
    "Please enter Number of previous abortions (0,1,2,3,4):",
    "Please enter Blood test result (Abnormal, Inconclusive, Normal, Slightly abnormal):",
    "Please enter Gender (Female, Male, Prefer not to say, Other):",
    "Please enter History of anomalies in previous pregnancies (Yes, No):",
    "Please enter Folic acid details (peri-conceptional) [Yes, No]:",
    "Please enter Place of birth (Home, Institute):",
    "Please enter Assisted conception IVF/ART (Yes, No):",
    "Please enter Follow-up (High, Low):",
    "Please enter Heart Rate (rates/min) [Normal, Tachycardia]:",
    "Please enter Birth asphyxia (Yes, No):",
    "Please enter Respiratory Rate (breaths/min) [Normal, Tachypnea]:",
    "Please enter History of serious maternal illness (Yes, No):",
    "Please enter History of radiation exposure (x-ray) [Yes, No]:",
    "Please enter Birth defects (Multiple, Singular):",
    "Please enter Status (Alive, Deceased):",
    "Please enter History of substance abuse (Yes, No):",
    "Please enter Autopsy shows birth defect (if applicable) [Yes, No]:",
    "Please enter Maternal gene (Yes, No):",
    "Please enter Paternal gene (Yes, No):",
    "Please enter Genes in mother's side (Yes, No):",
    "Please enter Inherited from father (Yes, No):"
]

@app.route('/chatbot', methods=['POST'])
def chatbot():
    selected_model = request.form.get('model')
    user_response = request.form.get('user_response')
    current_question_index = int(request.form.get('current_question_index'))
    chat_history_str = request.form.get('chat_history', '[]')
    try:
        chat_history = json.loads(chat_history_str)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error (chat_history): {e}")
        print(f"Problematic JSON String (chat_history): {chat_history_str}")
        chat_history = []

    collected_answers_str = request.form.get('collected_answers', '{}')
    try:
        collected_answers = json.loads(collected_answers_str)
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error (collected_answers): {e}")
        print(f"Problematic JSON String (collected_answers): {collected_answers_str}")
        collected_answers = {}

    features = []
    label_mapping = {}
    model = None
    questions_to_use = []  # Initialize the question list here

    if selected_model == 'genetic':
        features = features_genetic
        label_mapping = {0: "Mitochondrial genetic inheritance disorders", 1: "Multifactorial genetic inheritance disorders", 2: "No Genetic Disorder Detected", 3: "Single-gene inheritance diseases"}
        model = genetic_disorder_model
        questions_to_use = questions_genetic  # Use genetic questions
    elif selected_model == 'subclass':
        features = features_subclass
        label_mapping = {0: "Alzheimer's", 1: "Cancer", 2: "Cystic fibrosis", 3: "Diabetes", 4: "Hemochromatosis", 5: "Leber's hereditary optic neuropathy", 6: "Leigh syndrome", 7: "Mitochondrial myopathy", 8: "No Disorder Subclass Detected", 9: "Tay-Sachs"}
        model = disorder_subclass_model
        questions_to_use = questions  # Use generic questions
    elif selected_model == 'combined':
        features = features_combined
        label_mapping = {0: "Mitochondrial genetic inheritance disorders - Leber's hereditary optic neuropathy", 1: "Mitochondrial genetic inheritance disorders - Leigh syndrome", 2: "Mitochondrial genetic inheritance disorders - Mitochondrial myopathy", 3: "Multifactorial genetic inheritance disorders - Alzheimer's", 4: "Multifactorial genetic inheritance disorders - Cancer", 5: "Multifactorial genetic inheritance disorders - Diabetes", 6: "No Genetic Disorder Detected - No Disorder Subclass Detected", 7: "Single-gene inheritance diseases - Cystic fibrosis", 8: "Single-gene inheritance diseases - Hemochromatosis", 9: "Single-gene inheritance diseases - Tay-Sachs"}
        model = combined_disorder_model
        questions_to_use = questions  # Use generic questions

    chat_history.append({'user': user_response})

    current_feature = features[current_question_index]
    if current_feature in categorical_mapping_chatbot:
        numeric_response = None
        user_response_lower = user_response.lower().strip()
        mapping = categorical_mapping_chatbot[current_feature]
        for human_val, numeric_val in mapping.items():
            if user_response_lower == human_val.lower().strip():
                numeric_response = numeric_val
                break
        if numeric_response is None:
            chat_history.append({'bot': f"Invalid input for '{current_feature}'. Please answer with a valid option (e.g., Yes, No, Male, Female, Hospital, Home, Normal, Tachypnea, Alive, Deceased, Abnormal, Inconclusive, Normal, Slightly abnormal)."})
            chat_history_json = json.dumps(chat_history)
            collected_answers_json = json.dumps(collected_answers)
            return render_template('genome-chatbot.html', chat_history=chat_history, selected_model=selected_model, current_question_index=current_question_index, collected_answers=collected_answers, chat_history_json=chat_history_json, collected_answers_json=collected_answers_json, total_questions=len(questions_to_use)) # Use questions_to_use length
        collected_answers[current_feature] = user_response.strip()
    else:
        collected_answers[current_feature] = user_response

    next_question_index = current_question_index + 1
    next_question = None
    disable_input = False
    show_dashboard_button = False

    if next_question_index < len(features):
        next_question = questions_to_use[next_question_index]  # Use the correct question list
        chat_history.append({'bot': next_question})
    else:
        numeric_data = transform_input_chatbot(collected_answers, features) # Use the new function

        if isinstance(numeric_data, str):
            chat_history.append({'bot': f"Error: {numeric_data}"})
        else:
            input_df = pd.DataFrame([numeric_data], columns=features)
            prediction = model.predict(input_df)[0]
            final_detection = label_mapping.get(prediction, "Unknown Detection")
            chat_history.append({'bot': f"Prediction: {final_detection}"})
        chat_history.append({'bot': "For a comprehensive evaluation, it is recommended to consult a qualified medical professional."})
        chat_history.append({'bot': "Thank you for using the chatbot."})
        disable_input = True
        show_dashboard_button = True

    chat_history_json = json.dumps(chat_history)
    collected_answers_json = json.dumps(collected_answers)
    return render_template('genome-chatbot.html', chat_history=chat_history, selected_model=selected_model, current_question_index=next_question_index, collected_answers=collected_answers, chat_history_json=chat_history_json, collected_answers_json=collected_answers_json, total_questions=len(questions_to_use), disable_input=disable_input, show_dashboard_button=show_dashboard_button) # Use questions_to_use length here too

# Run Server (remains the same)
if __name__ == "__main__":
    app.run(port=5000, host='0.0.0.0')