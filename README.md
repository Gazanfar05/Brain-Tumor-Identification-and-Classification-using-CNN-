Brain Tumor MRI Analyzer
A deep learning-powered web application for classifying brain tumors from MRI scans using a Convolutional Neural Network (CNN). This tool uses artificial intelligence to analyze MRI images and provide instant predictions with detailed medical information.

Features
AI-Powered Classification: Uses a trained CNN model to classify brain tumors into 4 categories:

Glioma (Glioblastoma Multiforme)
Meningioma
Pituitary Adenoma
No Tumor
Modern Web Interface:

Drag-and-drop file upload
Real-time image preview
Interactive annotation tools
Responsive design for all devices
Detailed Analysis:

Confidence scores for each prediction
Comprehensive tumor information database
Detailed characteristics, symptoms, and treatment options
PDF report generation with annotations
Medical Information Database:

Full descriptions of each tumor type
Key characteristics and symptoms
Treatment options
Survival rates and prevalence data

_**Advanced Features:**_

Image annotation tools (rectangle, circle, freehand)
PDF report generation with images and analysis
Prediction history and metrics tracking
Server-side metrics calculation

 **Usage**
Upload MRI Image

Drag and drop an MRI scan image or click "Browse Files"
Supported formats: JPG, PNG, GIF
Maximum file size: 10MB
Analyze

Click the "Analyze MRI" button
Wait for the model to process the image
View Results

See tumor classification and confidence score
Browse detailed medical information
View all classification probabilities
Generate Report

Annotate the MRI image if desired
Generate a PDF report with analysis and annotations
Download for medical records



🔧 **Model Details**
**Architecture**
Type: Convolutional Neural Network (CNN)
Input: 224×224 grayscale MRI images
Output: 4-class classification
Framework: TensorFlow/Keras
**Input Processing**
Images converted to grayscale
Resized to 224×224 pixels
Normalized to 0-1 range
Channel dimension added: (224, 224, 1)
Batch dimension added for prediction
**Training Data**
Located in train directory
Organized by tumor type subdirectories
Evaluated on test dataset
**Model Files**
model.h5 - Current production model
brain_tumor_cnn.h5 - Backup model

 **Frontend Technologies**
HTML5 - Semantic markup
CSS3 - Modern styling with CSS variables and gradients
JavaScript (ES6) - Interactive features
Font Awesome - Icons
jsPDF - PDF generation
File Drag & Drop API - Upload handling
 **UI Features**
Responsive Design: Works on desktop, tablet, and mobile
Dark Mode Ready: CSS variables for easy theme switching
Accessibility: WCAG 2.1 compliant
Smooth Animations: CSS transitions and keyframe animations
Interactive Components:
Drag-and-drop upload
Real-time preview
Progress indicators
Loading spinners
