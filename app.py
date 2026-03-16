from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, Doctor, Patient, MRIScan
from forms import (
    DoctorRegistrationForm, DoctorLoginForm, PatientRegistrationForm, 
    PatientLoginForm, AddPatientForm, ClinicalNotesForm
)
import tensorflow as tf
import numpy as np
import cv2
import os
from datetime import datetime
from dotenv import load_dotenv
import secrets

load_dotenv()

# ============ APP CONFIGURATION ============
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(16))
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///brain_tumor.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file

# ============ INITIALIZE ============
db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'home'
login_manager.login_message = 'Please log in first'

@app.after_request
def after_request(response):
    """Disable caching for all responses"""
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Load model
MODEL_PATH = "model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✓ Model loaded successfully")
except:
    print("✗ Model not found")
    model = None

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

# ============ USER LOADER ============
@login_manager.user_loader
def load_user(user_id):
    """Load user from database"""
    from flask import session

    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return None

    role = session.get('role')

    if role == 'doctor':
        return Doctor.query.get(user_id)
    elif role == 'patient':
        return Patient.query.get(user_id)

    return None

# ============ UTILITY FUNCTIONS ============
def generate_patient_id():
    """Generate unique patient ID"""
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    random_code = secrets.token_hex(3).upper()
    return f"PAT-{timestamp}-{random_code}"

def generate_scan_id():
    """Generate unique scan ID"""
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    random_code = secrets.token_hex(3).upper()
    return f"SCAN-{timestamp}-{random_code}"

def preprocess_image(image_path):
    """Preprocess image for model"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image")
    
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    
    return img

# ============ HOME & AUTH ROUTES ============
@app.route('/')
def home():
    """Landing page"""
    return render_template('home.html')

# ============ DOCTOR ROUTES ============
@app.route('/doctor/register', methods=['GET', 'POST'])
def doctor_register():
    """Doctor registration"""
    if current_user.is_authenticated:
        return redirect(url_for('doctor_dashboard'))
    
    form = DoctorRegistrationForm()
    if form.validate_on_submit():
        doctor = Doctor(
            username=form.username.data,
            email=form.email.data,
            full_name=form.full_name.data,
            specialization=form.specialization.data,
            license_number=form.license_number.data,
            hospital=form.hospital.data,
            phone=form.phone.data
        )
        doctor.set_password(form.password.data)
        db.session.add(doctor)
        db.session.commit()
        
        flash(f'Welcome Dr. {doctor.full_name}! Account created successfully.', 'success')
        return redirect(url_for('doctor_login'))
    
    return render_template('doctor_register.html', form=form)

@app.route('/doctor/login', methods=['GET', 'POST'])
def doctor_login():
    """Doctor login"""
    from flask import session
    
    # Force logout if already logged in
    if current_user.is_authenticated:
        session.clear()
        logout_user()
    
    form = DoctorLoginForm()
    if form.validate_on_submit():
        doctor = Doctor.query.filter_by(username=form.username.data).first()
        
        if doctor and doctor.check_password(form.password.data):
            login_user(doctor)
            session['role'] = 'doctor'
            print(f"✓ Doctor logged in: {doctor.username}")
            flash(f'Welcome back, Dr. {doctor.full_name}!', 'success')
            return redirect(url_for('doctor_dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('doctor_login.html', form=form)

@app.route('/doctor/dashboard')
@login_required
def doctor_dashboard():
    """Doctor dashboard"""
    print(f"\n[DOCTOR DASHBOARD CHECK]")
    print(f"  Current User: {current_user}")
    print(f"  Is Doctor: {isinstance(current_user, Doctor)}")
    print(f"  Is Patient: {isinstance(current_user, Patient)}\n")
    
    if not isinstance(current_user, Doctor):
        print(f"✗ ACCESS DENIED - User is not a Doctor")
        flash('Unauthorized access. Please login as a doctor.', 'danger')
        logout_user()
        return redirect(url_for('doctor_login'))
    
    patients = Patient.query.filter_by(doctor_id=current_user.id).all()
    total_patients = len(patients)
    total_scans = MRIScan.query.filter(
        MRIScan.patient_id.in_([p.id for p in patients])
    ).count()
    
    recent_scans = MRIScan.query.filter(
        MRIScan.patient_id.in_([p.id for p in patients])
    ).order_by(MRIScan.created_at.desc()).limit(10).all()
    
    return render_template('doctor_dashboard.html',
                         patients=patients,
                         total_patients=total_patients,
                         total_scans=total_scans,
                         recent_scans=recent_scans,
                         now=datetime.utcnow())

@app.route('/doctor/patients/<int:patient_id>')
@login_required
def doctor_patient_detail(patient_id):
    """View patient details (doctor)"""
    if not isinstance(current_user, Doctor):
        return redirect(url_for('home'))
    
    patient = Patient.query.get_or_404(patient_id)
    
    # Verify doctor owns this patient
    if patient.doctor_id != current_user.id:
        flash('Unauthorized access', 'danger')
        return redirect(url_for('doctor_dashboard'))
    
    scans = MRIScan.query.filter_by(patient_id=patient_id).order_by(MRIScan.created_at.desc()).all()
    
    return render_template('doctor_patient_detail.html', patient=patient, scans=scans)

@app.route('/doctor/add-patient', methods=['GET', 'POST'])
@login_required
def doctor_add_patient():
    """Add patient (doctor)"""
    if not isinstance(current_user, Doctor):
        return redirect(url_for('home'))
    
    form = AddPatientForm()
    if form.validate_on_submit():
        patient = Patient(
            patient_id=generate_patient_id(),
            full_name=form.full_name.data,
            email=form.email.data,
            phone=form.phone.data,
            date_of_birth=form.date_of_birth.data,
            gender=form.gender.data,
            blood_group=form.blood_group.data,
            address=form.address.data,
            city=form.city.data,
            state=form.state.data,
            postal_code=form.postal_code.data,
            medical_history=form.medical_history.data,
            allergies=form.allergies.data,
            doctor_id=current_user.id
        )
        patient.set_password(form.temporary_password.data)
        db.session.add(patient)
        db.session.commit()
        
        flash(f'Patient {patient.full_name} added successfully!', 'success')
        return redirect(url_for('doctor_patient_detail', patient_id=patient.id))
    
    return render_template('doctor_add_patient.html', form=form)

# ============ DOCTOR SCAN REVIEW ROUTE ============
@app.route('/doctor/scan/<int:scan_id>/review', methods=['GET', 'POST'])
@login_required
def doctor_review_scan(scan_id):
    """Doctor reviews and confirms scan"""
    if not isinstance(current_user, Doctor):
        return jsonify({"error": "Only doctors can review"}), 403
    
    scan = MRIScan.query.get_or_404(scan_id)
    
    # Verify doctor owns this patient
    patient = Patient.query.get(scan.patient_id)
    if not patient or patient.doctor_id != current_user.id:
        flash('Unauthorized access', 'danger')
        return redirect(url_for('doctor_dashboard'))
    
    # GET request - show review form
    if request.method == 'GET':
        return render_template('doctor_review_scan.html', scan=scan)
    
    # POST request - submit review
    data = request.get_json()
    
    scan.status = data.get('status', 'Confirmed')
    scan.doctor_diagnosis = data.get('diagnosis', scan.tumor_type)
    scan.doctor_notes = data.get('notes', '')
    scan.reviewed_by_id = current_user.id
    scan.review_date = datetime.utcnow()
    
    db.session.commit()
    
    print(f"✓ Scan {scan.scan_id} reviewed by Dr. {current_user.full_name}")
    
    return jsonify({
        "status": "success",
        "message": f"Scan {scan.scan_id} has been {scan.status.lower()}!"
    })
# ============ PATIENT ROUTES ============
@app.route('/patient/register', methods=['GET', 'POST'])
def patient_register():
    """Patient registration"""
    if current_user.is_authenticated:
        if isinstance(current_user, Patient):
            return redirect(url_for('patient_dashboard'))
        else:
            return redirect(url_for('doctor_dashboard'))
    
    form = PatientRegistrationForm()
    if form.validate_on_submit():
        # Find a default doctor
        default_doctor = Doctor.query.first()
        if not default_doctor:
            flash('No doctors available. Please contact administrator.', 'danger')
            return redirect(url_for('patient_register'))
        
        patient = Patient(
            patient_id=generate_patient_id(),
            full_name=form.full_name.data,
            email=form.email.data,
            phone=form.phone.data,
            date_of_birth=form.date_of_birth.data,
            gender=form.gender.data,
            blood_group=form.blood_group.data,
            address=form.address.data,
            city=form.city.data,
            state=form.state.data,
            postal_code=form.postal_code.data,
            emergency_contact=form.emergency_contact.data,
            emergency_phone=form.emergency_phone.data,
            medical_history=form.medical_history.data,
            allergies=form.allergies.data,
            doctor_id=default_doctor.id
        )
        patient.set_password(form.password.data)
        db.session.add(patient)
        db.session.commit()
        
        flash('Account created successfully! You can now login.', 'success')
        return redirect(url_for('patient_login'))
    
    return render_template('patient_register.html', form=form)

@app.route('/patient/login', methods=['GET', 'POST'])
def patient_login():
    """Patient login"""
    from flask import session
    
    # Force logout if already logged in
    if current_user.is_authenticated:
        session.clear()
        logout_user()
    
    form = PatientLoginForm()
    if form.validate_on_submit():
        patient = Patient.query.filter_by(email=form.email.data).first()
        
        if patient and patient.check_password(form.password.data):
            login_user(patient)
            session['role'] = 'patient'
            print(f"✓ Patient logged in: {patient.email}")
            flash(f'Welcome {patient.full_name}!', 'success')
            return redirect(url_for('patient_dashboard'))
        else:
            flash('Invalid email or password', 'danger')
    
    return render_template('patient_login.html', form=form)

@app.route('/patient/dashboard')
@login_required
def patient_dashboard():
    """Patient dashboard"""
    print(f"\n[PATIENT DASHBOARD CHECK]")
    print(f"  Current User: {current_user}")
    print(f"  Is Patient: {isinstance(current_user, Patient)}")
    print(f"  Is Doctor: {isinstance(current_user, Doctor)}\n")
    
    if not isinstance(current_user, Patient):
        print(f"✗ ACCESS DENIED - User is not a Patient")
        flash('Unauthorized access. Please login as a patient.', 'danger')
        logout_user()
        return redirect(url_for('patient_login'))
    
    scans = MRIScan.query.filter_by(patient_id=current_user.id).order_by(MRIScan.created_at.desc()).all()
    
    return render_template('patient_dashboard.html', patient=current_user, scans=scans)

@app.route('/patient/profile')
@login_required
def patient_profile():
    """Patient profile view"""
    if not isinstance(current_user, Patient):
        return redirect(url_for('home'))
    
    return render_template('patient_profile.html', patient=current_user)

@app.route('/patient/scan/<int:scan_id>')
@login_required
def patient_view_scan(scan_id):
    """View scan details (patient or doctor)"""
    scan = MRIScan.query.get_or_404(scan_id)
    
    # If patient is logged in
    if isinstance(current_user, Patient):
        # Verify patient owns this scan
        if scan.patient_id != current_user.id:
            flash('Unauthorized access', 'danger')
            return redirect(url_for('patient_dashboard'))
    
    # If doctor is logged in
    elif isinstance(current_user, Doctor):
        patient = Patient.query.get(scan.patient_id)
        if not patient or patient.doctor_id != current_user.id:
            flash('Unauthorized access', 'danger')
            return redirect(url_for('doctor_dashboard'))
    
    else:
        return redirect(url_for('home'))
    
    print(f"Viewing scan: {scan.scan_id}")
    print(f"Tumor Status: {scan.tumor_status}")
    print(f"Confidence: {scan.confidence}")
    
    return render_template('patient_view_scan.html', scan=scan)

# ============ UPLOAD & ANALYSIS ROUTES ============
@app.route('/upload-scan', methods=['POST'])
@login_required
def upload_scan():
    """Upload and analyze MRI scan"""
    if not isinstance(current_user, Patient):
        return jsonify({"error": "Only patients can upload scans"}), 403
    
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    temp_path = f"temp_{current_user.id}_{datetime.utcnow().timestamp()}.jpg"
    file.save(temp_path)
    
    try:
        img = preprocess_image(temp_path)
        preds = model.predict(img, verbose=0)[0]
        
        idx = np.argmax(preds)
        confidence = float(preds[idx]) * 100
        label = CLASS_NAMES[idx]
        
        all_predictions = [
            {"class": CLASS_NAMES[i], "probability": float(preds[i]), "confidence": float(preds[i]) * 100}
            for i in range(len(CLASS_NAMES))
        ]
        all_predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        tumor_status = "No Tumor" if label == "no_tumor" else "Tumor Detected"
        tumor_type = None if label == "no_tumor" else label
        
        # Save scan to database
        scan = MRIScan(
            scan_id=generate_scan_id(),
            patient_id=current_user.id,
            tumor_status=tumor_status,
            tumor_type=tumor_type,
            confidence=confidence,
            image_path=temp_path,
            prediction_data={
                "all_predictions": all_predictions,
                "class_names": CLASS_NAMES
            },
            status='Pending'
        )
        db.session.add(scan)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "scan_id": scan.id,
            "tumor_status": tumor_status,
            "tumor_type": tumor_type,
            "confidence": round(confidence, 2),
            "all_predictions": all_predictions
        })
    
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500


# ============ DOWNLOAD REPORT ROUTE ============
@app.route('/patient/scan/<int:scan_id>/download-report')
@login_required
def download_report(scan_id):
    """Download scan report as PDF"""
    scan = MRIScan.query.get_or_404(scan_id)
    
    # Verify access
    if isinstance(current_user, Patient):
        if scan.patient_id != current_user.id:
            return jsonify({"error": "Unauthorized"}), 403
    elif isinstance(current_user, Doctor):
        patient = Patient.query.get(scan.patient_id)
        if not patient or patient.doctor_id != current_user.id:
            return jsonify({"error": "Unauthorized"}), 403
    else:
        return redirect(url_for('home'))
    
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    import io
    
    # Tumor information database
    tumor_info = {
        'glioma': {
            'name': 'Glioma (Glioblastoma Multiforme)',
            'description': 'Gliomas are tumors that originate in glial cells, which support brain neurons. Glioblastomas are the most aggressive grade IV brain tumors.',
            'symptoms': [
                'Severe and persistent headaches',
                'Vision or hearing problems',
                'Balance and coordination difficulties',
                'Progressive memory loss',
                'Seizures (40% of cases)',
                'Weakness on one side of body',
                'Difficulty speaking or understanding speech',
                'Personality changes'
            ],
            'causes': [
                'Unknown cause in most cases',
                'Previous brain radiation',
                'Family history of brain tumors',
                'Certain genetic syndromes'
            ],
            'prevention': [
                'Limit exposure to radiation',
                'Maintain healthy diet rich in antioxidants',
                'Regular exercise and physical activity',
                'Avoid smoking and excessive alcohol',
                'Manage stress effectively',
                'Regular health checkups'
            ],
            'treatment': [
                'Surgical debulking (remove as much tumor as possible)',
                'Radiation therapy (external beam)',
                'Chemotherapy (usually Temozolomide)',
                'Combination therapy (most effective)',
                'Experimental immunotherapy'
            ],
            'prognosis': 'Median survival: 12-15 months with treatment',
            'survival_rate': '2-year: 25-30%, 5-year: <10%'
        },
        'meningioma': {
            'name': 'Meningioma',
            'description': 'Meningiomas develop in the meninges (membranes surrounding the brain and spinal cord). 80% are benign (grade I).',
            'symptoms': [
                'Gradual onset headaches',
                'Vision changes',
                'Hearing loss or tinnitus',
                'Balance problems',
                'Memory issues',
                'Seizures (in some cases)',
                'Weakness or numbness',
                'May be asymptomatic for years'
            ],
            'causes': [
                'Unknown in most cases',
                'Female hormones (more common in women)',
                'Previous head radiation',
                'Neurofibromatosis type 2',
                'Increasing age'
            ],
            'prevention': [
                'Avoid unnecessary head radiation',
                'Protect head from trauma',
                'Maintain healthy weight',
                'Regular exercise',
                'Manage hormone levels',
                'Annual health checkups if at risk'
            ],
            'treatment': [
                'Observation (if benign and asymptomatic)',
                'Surgery (typically curative for benign)',
                'Radiation therapy (if surgical risks high)',
                'Stereotactic radiosurgery',
                'Chemotherapy (rarely used)'
            ],
            'prognosis': 'Generally favorable for benign tumors',
            'survival_rate': '5-year: 84% (benign), 60% (atypical), 40% (malignant)'
        },
        'pituitary': {
            'name': 'Pituitary Adenoma',
            'description': 'Pituitary adenomas are tumors in the pituitary gland (at brain base). Usually benign and slow-growing.',
            'symptoms': [
                'Severe headaches',
                'Vision problems (bitemporal hemianopsia)',
                'Hormonal imbalances',
                'Sexual dysfunction',
                'Mood changes and depression',
                'Fatigue and weakness',
                'Weight gain or loss',
                'Galactorrhea (inappropriate lactation)'
            ],
            'causes': [
                'Unknown cause in most cases',
                'Family history',
                'Multiple Endocrine Neoplasia (MEN)',
                'Pituitary radiation',
                'Estrogen hormone therapy'
            ],
            'prevention': [
                'Regular hormone level monitoring',
                'Limit estrogen exposure if possible',
                'Healthy lifestyle and diet',
                'Manage stress effectively',
                'Regular endocrinology checkups',
                'Report hormonal symptoms promptly'
            ],
            'treatment': [
                'Observation (if non-growing)',
                'Medication (hormone-blocking drugs)',
                'Transsphenoidal surgery (minimally invasive)',
                'Radiation therapy (if surgery fails)',
                'Hormone replacement therapy'
            ],
            'prognosis': 'Excellent with proper treatment',
            'survival_rate': '5-year: 95% overall, Quality of life usually good'
        },
        'no_tumor': {
            'name': 'Normal Brain MRI',
            'description': 'No brain tumor detected. The MRI scan shows normal brain tissue with no abnormal growths or lesions.',
            'symptoms': [],
            'causes': ['No tumor present'],
            'prevention': [
                'Maintain healthy lifestyle',
                'Regular exercise',
                'Balanced diet',
                'Manage stress',
                'Avoid smoking and excessive alcohol',
                'Regular health checkups'
            ],
            'treatment': ['No treatment required'],
            'prognosis': 'No tumor detected',
            'survival_rate': 'N/A'
        }
    }
    
    # Get tumor info
    tumor_type = scan.tumor_type or 'no_tumor'
    info = tumor_info.get(tumor_type, tumor_info['no_tumor'])
    
    # Create PDF in memory
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#3b5cff'),
        spaceAfter=12,
        alignment=1
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#3b5cff'),
        spaceAfter=8,
        spaceBefore=8
    )
    
    # Title
    elements.append(Paragraph("Brain MRI Analysis Report", title_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Patient Info
    patient = Patient.query.get(scan.patient_id)
    elements.append(Paragraph("<b>Patient Information</b>", heading_style))
    patient_data = [
        ['Patient Name', patient.full_name],
        ['Patient ID', patient.patient_id],
        ['Date of Birth', patient.date_of_birth.strftime('%Y-%m-%d')],
        ['Gender', patient.gender],
        ['Blood Group', patient.blood_group or 'N/A']
    ]
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f7ff')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(patient_table)
    elements.append(Spacer(1, 0.15*inch))
    
    # Scan Info
    elements.append(Paragraph("<b>Scan Information</b>", heading_style))
    scan_data = [
        ['Scan ID', scan.scan_id],
        ['Scan Date', scan.created_at.strftime('%Y-%m-%d %H:%M')],
        ['Status', scan.status],
        ['Doctor Review Date', scan.review_date.strftime('%Y-%m-%d') if scan.review_date else 'Pending']
    ]
    scan_table = Table(scan_data, colWidths=[2*inch, 4*inch])
    scan_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f7ff')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(scan_table)
    elements.append(Spacer(1, 0.15*inch))
    
    # Analysis Results
    elements.append(Paragraph("<b>Analysis Results</b>", heading_style))
    result_data = [
        ['Tumor Status', scan.tumor_status],
        ['Detected Type', scan.tumor_type or 'N/A'],
        ['AI Confidence', f"{scan.confidence}%"],
        ['Doctor Diagnosis', scan.doctor_diagnosis or 'Pending'],
        ['Status', scan.status]
    ]
    result_table = Table(result_data, colWidths=[2*inch, 4*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3b5cff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(result_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Page break
    elements.append(PageBreak())
    
    # Tumor Information
    elements.append(Paragraph(f"<b>{info['name']}</b>", title_style))
    elements.append(Spacer(1, 0.1*inch))
    
    # Description
    elements.append(Paragraph("<b>Description</b>", heading_style))
    elements.append(Paragraph(info['description'], styles['Normal']))
    elements.append(Spacer(1, 0.1*inch))
    
    # Symptoms
    elements.append(Paragraph("<b>Common Symptoms</b>", heading_style))
    if info['symptoms']:
        for symptom in info['symptoms']:
            elements.append(Paragraph(f"• {symptom}", styles['Normal']))
    else:
        elements.append(Paragraph("No specific symptoms", styles['Normal']))
    elements.append(Spacer(1, 0.1*inch))
    
    # Causes
    elements.append(Paragraph("<b>Causes & Risk Factors</b>", heading_style))
    for cause in info['causes']:
        elements.append(Paragraph(f"• {cause}", styles['Normal']))
    elements.append(Spacer(1, 0.1*inch))
    
    # Prevention
    elements.append(Paragraph("<b>Prevention Measures</b>", heading_style))
    for prevention in info['prevention']:
        elements.append(Paragraph(f"• {prevention}", styles['Normal']))
    elements.append(Spacer(1, 0.1*inch))
    
    # Treatment
    elements.append(Paragraph("<b>Treatment Options</b>", heading_style))
    for treatment in info['treatment']:
        elements.append(Paragraph(f"• {treatment}", styles['Normal']))
    elements.append(Spacer(1, 0.1*inch))
    
    # Prognosis
    elements.append(Paragraph("<b>Prognosis & Survival</b>", heading_style))
    elements.append(Paragraph(f"• {info['prognosis']}", styles['Normal']))
    elements.append(Paragraph(f"• {info['survival_rate']}", styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Doctor Notes
    if scan.doctor_notes:
        elements.append(PageBreak())
        elements.append(Paragraph("<b>Doctor's Notes</b>", heading_style))
        elements.append(Paragraph(scan.doctor_notes, styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=1,
        spaceAfter=12
    )
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph(
        "<i>This report is generated by an AI system for demonstration purposes. "
        "It should not be used for actual medical diagnosis without professional consultation. "
        "Always consult with qualified medical professionals for diagnosis and treatment decisions.</i>",
        disclaimer_style
    ))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    
    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"MRI_Report_{scan.scan_id}.pdf"
    )

# ============ LOGOUT ============

@app.route('/logout')
@login_required
def logout():
    """Logout user"""
    from flask import session
    
    user_name = current_user.full_name if hasattr(current_user, 'full_name') else 'User'
    
    # Clear everything
    session.clear()
    logout_user()
    
    print(f"✓ User logged out: {user_name}")
    
    flash(f'{user_name}, you have been logged out.', 'info')
    response = redirect(url_for('home'))
    
    # Clear browser cache
    response.delete_cookie('session')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    
    return response

# ============ ERROR HANDLERS ============
@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    PORT = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=PORT, debug=os.environ.get('FLASK_ENV') != 'production')