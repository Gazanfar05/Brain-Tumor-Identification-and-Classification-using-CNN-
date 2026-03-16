from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, TextAreaField, SelectField, DateField
from wtforms.validators import DataRequired, Email, EqualTo, Length, ValidationError, Regexp
from models import Doctor, Patient
import re

# ============ DOCTOR REGISTRATION ============
class DoctorRegistrationForm(FlaskForm):
    username = StringField('Username', validators=[
        DataRequired(),
        Length(min=4, max=20)
    ])
    email = StringField('Email', validators=[
        DataRequired(),
        Email()
    ])
    full_name = StringField('Full Name', validators=[DataRequired()])
    specialization = StringField('Specialization', validators=[DataRequired()])
    license_number = StringField('Medical License Number', validators=[DataRequired()])
    hospital = StringField('Hospital/Clinic Name')
    phone = StringField('Phone Number', validators=[
        DataRequired(),
        Regexp(r'^\d{10}$', message='Phone must be 10 digits')
    ])
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=6)
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(),
        EqualTo('password', message='Passwords must match')
    ])
    submit = SubmitField('Register as Doctor')
    
    def validate_username(self, field):
        if Doctor.query.filter_by(username=field.data).first():
            raise ValidationError('Username already exists')
    
    def validate_email(self, field):
        if Doctor.query.filter_by(email=field.data).first():
            raise ValidationError('Email already registered')
    
    def validate_license_number(self, field):
        if Doctor.query.filter_by(license_number=field.data).first():
            raise ValidationError('License number already registered')


# ============ DOCTOR LOGIN ============
class DoctorLoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login as Doctor')


# ============ PATIENT REGISTRATION ============
class PatientRegistrationForm(FlaskForm):
    full_name = StringField('Full Name', validators=[DataRequired()])
    email = StringField('Email', validators=[
        DataRequired(),
        Email()
    ])
    phone = StringField('Phone Number', validators=[
        DataRequired(),
        Regexp(r'^\d{10}$', message='Phone must be 10 digits')
    ])
    date_of_birth = DateField('Date of Birth', validators=[DataRequired()])
    gender = SelectField('Gender', choices=[('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')], validators=[DataRequired()])
    blood_group = SelectField('Blood Group', choices=[
        ('', 'Select Blood Group'),
        ('O+', 'O+'),
        ('O-', 'O-'),
        ('A+', 'A+'),
        ('A-', 'A-'),
        ('B+', 'B+'),
        ('B-', 'B-'),
        ('AB+', 'AB+'),
        ('AB-', 'AB-')
    ])
    address = StringField('Address')
    city = StringField('City')
    state = StringField('State')
    postal_code = StringField('Postal Code')
    emergency_contact = StringField('Emergency Contact Name')
    emergency_phone = StringField('Emergency Phone', validators=[Regexp(r'^\d{10}$', message='Phone must be 10 digits')])
    medical_history = TextAreaField('Medical History')
    allergies = TextAreaField('Allergies')
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=6)
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(),
        EqualTo('password', message='Passwords must match')
    ])
    submit = SubmitField('Create Patient Account')
    
    def validate_email(self, field):
        if Patient.query.filter_by(email=field.data).first():
            raise ValidationError('Email already registered')
    
    def validate_phone(self, field):
        if Patient.query.filter_by(phone=field.data).first():
            raise ValidationError('Phone number already registered')


# ============ PATIENT LOGIN ============
class PatientLoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login as Patient')


# ============ ADD PATIENT (Doctor Side) ============
class AddPatientForm(FlaskForm):
    full_name = StringField('Full Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    phone = StringField('Phone Number', validators=[
        DataRequired(),
        Regexp(r'^\d{10}$', message='Phone must be 10 digits')
    ])
    date_of_birth = DateField('Date of Birth', validators=[DataRequired()])
    gender = SelectField('Gender', choices=[('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other')], validators=[DataRequired()])
    blood_group = SelectField('Blood Group', choices=[
        ('', 'Select Blood Group'),
        ('O+', 'O+'),
        ('O-', 'O-'),
        ('A+', 'A+'),
        ('A-', 'A-'),
        ('B+', 'B+'),
        ('B-', 'B-'),
        ('AB+', 'AB+'),
        ('AB-', 'AB-')
    ])
    address = StringField('Address')
    city = StringField('City')
    state = StringField('State')
    postal_code = StringField('Postal Code')
    medical_history = TextAreaField('Medical History')
    allergies = TextAreaField('Allergies')
    temporary_password = PasswordField('Temporary Password', validators=[DataRequired(), Length(min=6)])
    submit = SubmitField('Add Patient')


# ============ CLINICAL NOTES ============
class ClinicalNotesForm(FlaskForm):
    notes = TextAreaField('Clinical Notes', validators=[DataRequired()])
    submit = SubmitField('Save Notes')