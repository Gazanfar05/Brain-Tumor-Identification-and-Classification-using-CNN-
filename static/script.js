let currentResult = null;  // 
// ============ DOM ELEMENTS ============
const uploadArea = document.getElementById("uploadArea");
const imageInput = document.getElementById("imageInput");
const loadingSpinner = document.getElementById("loadingSpinner");
const resultsSection = document.getElementById("resultsSection");
const errorMessage = document.getElementById("errorMessage");

console.log("✓ Script initialized", {
    uploadArea: !!uploadArea,
    imageInput: !!imageInput,
    loadingSpinner: !!loadingSpinner,
    resultsSection: !!resultsSection,
    errorMessage: !!errorMessage
});

// ============ DRAG & DROP ============
if (uploadArea) {
    uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadArea.classList.add("drag-over");
    });

    uploadArea.addEventListener("dragleave", () => {
        uploadArea.classList.remove("drag-over");
    });

    uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadArea.classList.remove("drag-over");
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            imageInput.files = files;
            previewImage({ target: { files: files } });
        }
    });
}

// ============ PREVIEW IMAGE ============
function previewImage(event) {
    const files = event.target ? event.target.files : event.files;
    const file = files[0];
    
    if (!file) return;

    if (file.size > 10 * 1024 * 1024) {
        showError("File too large. Maximum 10MB allowed.");
        return;
    }

    const reader = new FileReader();
    reader.onload = function() {
        document.getElementById("preview").src = reader.result;
        document.getElementById("filename").textContent = "File: " + file.name;
        document.getElementById("previewSection").style.display = "block";
        document.getElementById("uploadArea").style.display = "none";
        hideResults();
        hideError();
    };
    reader.readAsDataURL(file);
}

// ============ CLEAR IMAGE ============
function clearImage() {
    document.getElementById("imageInput").value = "";
    document.getElementById("previewSection").style.display = "none";
    document.getElementById("uploadArea").style.display = "block";
    hideResults();
    hideError();
}

// ============ HIDE/SHOW RESULTS ============
function hideResults() {
    resultsSection.style.display = "none";
}

function showResults() {
    resultsSection.style.display = "block";
    errorMessage.style.display = "none";
}

// ============ HIDE/SHOW ERROR ============
function hideError() {
    errorMessage.style.display = "none";
}

function showError(message) {
    document.getElementById("errorText").textContent = message;
    errorMessage.style.display = "block";
    resultsSection.style.display = "none";
    if (loadingSpinner) loadingSpinner.style.display = "none";
}

// ============ ANALYZE ============
function analyze() {
    const input = document.getElementById("imageInput");
    
    if (!input || input.files.length === 0) {
        showError("Please upload an MRI image first.");
        return;
    }

    const formData = new FormData();
    formData.append("image", input.files[0]);

    if (loadingSpinner) loadingSpinner.style.display = "flex";
    hideResults();
    hideError();

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => {
        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        return res.json();
    })
    .then(data => {
        if (loadingSpinner) loadingSpinner.style.display = "none";

        if (data.error) {
            showError(data.error);
            return;
        }

        displayResults(data);
    })
    .catch(err => {
        if (loadingSpinner) loadingSpinner.style.display = "none";
        showError("Prediction failed: " + err.message);
    });
}

// ============ DISPLAY RESULTS ============
function displayResults(data) {
    console.log("Results:", data);

    // SAVE CURRENT RESULT
    currentResult = data;

    const tumorStatus = document.getElementById("tumorStatus");
    const typeCard = document.getElementById("typeCard");
    
    if (data.tumor_status === "No Tumor") {
        tumorStatus.textContent = "✓ No Tumor Detected";
        tumorStatus.className = "status-value no-tumor";
        typeCard.style.display = "none";
    } else {
        tumorStatus.textContent = "⚠ Tumor Detected";
        tumorStatus.className = "status-value tumor-detected";
        document.getElementById("tumorType").textContent = (data.tumor_type || "Unknown").toUpperCase();
        typeCard.style.display = "flex";
    }

    const confidence = data.confidence || 0;
    document.getElementById("confidencePercent").textContent = confidence.toFixed(1) + "%";
    
    const progressFill = document.getElementById("progressFill");
    progressFill.style.width = "0%";
    setTimeout(() => {
        progressFill.style.width = confidence + "%";
    }, 100);

    if (data.all_predictions) {
        const tbody = document.getElementById("classBody");
        tbody.innerHTML = "";
        data.all_predictions.forEach(pred => {
            tbody.insertRow().innerHTML = `
                <td><strong>${pred.class.replace(/_/g, ' ').toUpperCase()}</strong></td>
                <td>${(pred.probability * 100).toFixed(2)}%</td>
                <td><strong>${(pred.confidence || pred.probability * 100).toFixed(2)}%</strong></td>
            `;
        });
    }

    showResults();
    fetchMetrics();
    
    // Initialize annotation and show report button
    setTimeout(() => {
        initializeAnnotation();
        document.getElementById("reportSection").style.display = "block";
    }, 300);
    
    setTimeout(() => resultsSection.scrollIntoView({ behavior: "smooth" }), 300);
}

// ============ FETCH METRICS ============
function fetchMetrics() {
    fetch("/metrics")
        .then(res => res.json())
        .then(data => {
            if (data.accuracy !== undefined) {
                document.getElementById("metricAccuracy").textContent = data.accuracy + "%";
                document.getElementById("metricPrecision").textContent = data.precision + "%";
                document.getElementById("metricRecall").textContent = data.recall + "%";
                document.getElementById("totalPredictions").textContent = data.total_predictions;
                document.getElementById("metricsSection").style.display = "block";
            } else {
                document.getElementById("metricsSection").style.display = "none";
            }
        })
        .catch(err => console.log("Metrics unavailable"));
}