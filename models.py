from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

# ============ DOCTOR MODEL ============
class Doctor(UserMixin, db.Model):
    __tablename__ = 'doctors'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(120), nullable=False)
    specialization = db.Column(db.String(100), nullable=True)
    license_number = db.Column(db.String(50), unique=True, nullable=False)
    hospital = db.Column(db.String(150), nullable=True)
    phone = db.Column(db.String(15), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships - Fixed to avoid conflicts
    patients = db.relationship('Patient', backref='doctor', lazy=True, cascade='all, delete-orphan')
    reviewed_scans = db.relationship(
        'MRIScan',
        foreign_keys='MRIScan.reviewed_by_id',
        backref='doctor_reviewer',
        lazy=True,
        overlaps="doctor_reviewer"
    )
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if password is correct"""
        return check_password_hash(self.password_hash, password)
    
    @property
    def user_type(self):
        """User type for session management"""
        return 'doctor'
    
    def __repr__(self):
        return f'<Doctor {self.username}>'


# ============ PATIENT MODEL ============
class Patient(UserMixin, db.Model):
    __tablename__ = 'patients'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(50), unique=True, nullable=False)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(15), unique=True, nullable=False)
    date_of_birth = db.Column(db.Date, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    blood_group = db.Column(db.String(5), nullable=True)
    address = db.Column(db.String(255), nullable=True)
    city = db.Column(db.String(50), nullable=True)
    state = db.Column(db.String(50), nullable=True)
    postal_code = db.Column(db.String(10), nullable=True)
    emergency_contact = db.Column(db.String(120), nullable=True)
    emergency_phone = db.Column(db.String(15), nullable=True)
    medical_history = db.Column(db.Text, nullable=True)
    allergies = db.Column(db.Text, nullable=True)
    current_medications = db.Column(db.Text, nullable=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    scans = db.relationship('MRIScan', backref='patient', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if password is correct"""
        return check_password_hash(self.password_hash, password)
    
    @property
    def user_type(self):
        """User type for session management"""
        return 'patient'
    
    def __repr__(self):
        return f'<Patient {self.patient_id}>'


# ============ MRI SCAN MODEL ============
class MRIScan(db.Model):
    __tablename__ = 'mri_scans'
    
    id = db.Column(db.Integer, primary_key=True)
    scan_id = db.Column(db.String(50), unique=True, nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    
    # Scan Information
    scan_date = db.Column(db.DateTime, default=datetime.utcnow)
    tumor_status = db.Column(db.String(50), nullable=False)
    tumor_type = db.Column(db.String(50), nullable=True)
    confidence = db.Column(db.Float, nullable=False)
    
    # Image Paths
    image_path = db.Column(db.String(255), nullable=True)
    annotated_image_path = db.Column(db.String(255), nullable=True)
    
    # AI Analysis Data
    clinical_notes = db.Column(db.Text, nullable=True)
    prediction_data = db.Column(db.JSON, nullable=True)
    
    # Doctor Review
    status = db.Column(db.String(20), default='Pending')
    doctor_notes = db.Column(db.Text, nullable=True)
    doctor_diagnosis = db.Column(db.String(100), nullable=True)
    reviewed_by_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=True)
    review_date = db.Column(db.DateTime, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships - Fixed
    reviewed_by = db.relationship(
        'Doctor',
        foreign_keys=[reviewed_by_id],
        backref='scans_reviewed',
        overlaps="doctor_reviewer,reviewed_scans"
    )
    
    def __repr__(self):
        return f'<MRIScan {self.scan_id}>'