// ============ ANNOTATION STATE ============
let currentTool = "rect";
let isDrawing = false;
let startX, startY;
let canvas, ctx;
let originalImage = new Image();
let annotationHistory = [];
let canvasScale = 1;

// Tumor information database
const tumorDatabase = {
    glioma: {
        name: "Glioma",
        fullName: "Glioblastoma Multiforme (GBM)",
        description: "Gliomas are tumors that originate in glial cells, which support brain neurons. Glioblastomas are the most aggressive type of brain tumor.",
        characteristics: [
            "Highly aggressive grade IV tumor",
            "Rapid growth pattern",
            "Poor prognosis without treatment",
            "Often infiltrative into surrounding tissue"
        ],
        symptoms: [
            "Severe headaches",
            "Vision/hearing problems",
            "Balance and coordination issues",
            "Memory loss",
            "Seizures"
        ],
        treatment: [
            "Surgery (debulking)",
            "Radiation therapy",
            "Chemotherapy",
            "Targeted therapy"
        ],
        survivalRate: "12-15 months (average)",
        prevalence: "45% of all brain tumors"
    },
    meningioma: {
        name: "Meningioma",
        fullName: "Meningioma",
        description: "Meningiomas are tumors that develop in the membranes surrounding the brain and spinal cord (meninges). Most are benign.",
        characteristics: [
            "Usually slow-growing",
            "Often benign (non-cancerous)",
            "Can be asymptomatic for years",
            "May become symptomatic over time"
        ],
        symptoms: [
            "Headaches",
            "Vision changes",
            "Hearing loss",
            "Balance problems",
            "Memory issues",
            "Seizures (in some cases)"
        ],
        treatment: [
            "Observation (if benign/asymptomatic)",
            "Surgery",
            "Radiation therapy",
            "Chemotherapy (rare)"
        ],
        survivalRate: "5-year: 84% (benign), 60% (atypical)",
        prevalence: "30% of all brain tumors"
    },
    pituitary: {
        name: "Pituitary Adenoma",
        fullName: "Pituitary Adenoma",
        description: "Pituitary adenomas are tumors in the pituitary gland, which controls many hormones. Most are benign and slow-growing.",
        characteristics: [
            "Usually benign",
            "Slow growth",
            "Hormone-secreting or non-secreting",
            "Often manageable with treatment"
        ],
        symptoms: [
            "Headaches",
            "Vision problems (bitemporal hemianopsia)",
            "Hormonal imbalances",
            "Sexual dysfunction",
            "Mood changes",
            "Fatigue"
        ],
        treatment: [
            "Observation",
            "Medication (hormone management)",
            "Surgery",
            "Radiation (if needed)"
        ],
        survivalRate: "5-year: 95% (overall good)",
        prevalence: "10-15% of brain tumors"
    },
    no_tumor: {
        name: "No Tumor",
        fullName: "Normal Brain MRI",
        description: "The MRI scan shows no evidence of brain tumor. The imaging appears normal with no suspicious lesions detected.",
        characteristics: [
            "Normal brain tissue",
            "No abnormal growths detected",
            "All structures appear normal",
            "No concerning lesions"
        ],
        symptoms: [],
        treatment: ["No treatment required. Regular follow-up if symptoms persist."],
        survivalRate: "N/A",
        prevalence: "N/A"
    }
};

// ============ INITIALIZE ANNOTATION ============
function initializeAnnotation() {
    canvas = document.getElementById("annotationCanvas");
    if (!canvas) return;
    
    ctx = canvas.getContext("2d");
    
    // Get image from preview
    const previewImg = document.getElementById("preview");
    if (!previewImg || !previewImg.src) {
        console.log("No preview image found");
        return;
    }
    
    originalImage = new Image();
    originalImage.crossOrigin = "anonymous";
    
    originalImage.onload = () => {
        console.log("Image loaded:", originalImage.width, "x", originalImage.height);
        
        // Set canvas size to match image
        canvas.width = originalImage.width;
        canvas.height = originalImage.height;
        
        // Draw original image
        ctx.drawImage(originalImage, 0, 0);
        
        // Show annotation section
        document.getElementById("annotationSection").style.display = "block";
        
        console.log("Canvas ready:", canvas.width, "x", canvas.height);
    };
    
    originalImage.onerror = () => {
        console.error("Failed to load image for annotation");
    };
    
    originalImage.src = previewImg.src;
    
    // Mouse events
    canvas.addEventListener("mousedown", startDrawing);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", stopDrawing);
    canvas.addEventListener("mouseout", stopDrawing);
}

// ============ DRAWING FUNCTIONS ============
function setAnnotationTool(tool) {
    currentTool = tool;
    document.querySelectorAll(".tool-btn").forEach(btn => btn.classList.remove("active"));
    event.target.closest(".tool-btn").classList.add("active");
}

function startDrawing(e) {
    if (!canvas) return;
    
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    
    // Account for canvas scaling
    startX = (e.clientX - rect.left) * (canvas.width / rect.width);
    startY = (e.clientY - rect.top) * (canvas.height / rect.height);
    
    if (currentTool === "text") {
        const text = prompt("Enter text:");
        if (text) {
            ctx.font = "20px Arial";
            ctx.fillStyle = "#FF0000";
            ctx.strokeStyle = "#FFFFFF";
            ctx.lineWidth = 3;
            ctx.strokeText(text, startX, startY);
            ctx.fillText(text, startX, startY);
            annotationHistory.push({type: "text", x: startX, y: startY, text: text});
        }
        isDrawing = false;
    }
}

function draw(e) {
    if (!isDrawing || !canvas || currentTool === "text") return;
    
    const rect = canvas.getBoundingClientRect();
    const currentX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const currentY = (e.clientY - rect.top) * (canvas.height / rect.height);
    
    // Redraw original image
    ctx.drawImage(originalImage, 0, 0);
    
    // Redraw annotations
    redrawAnnotations();
    
    // Draw current shape
    ctx.strokeStyle = "#FF0000";
    ctx.lineWidth = 3;
    ctx.fillStyle = "rgba(255, 0, 0, 0.1)";
    
    if (currentTool === "rect") {
        const width = currentX - startX;
        const height = currentY - startY;
        ctx.fillRect(startX, startY, width, height);
        ctx.strokeRect(startX, startY, width, height);
    } else if (currentTool === "circle") {
        const radius = Math.sqrt((currentX - startX) ** 2 + (currentY - startY) ** 2);
        ctx.beginPath();
        ctx.arc(startX, startY, radius, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
    } else if (currentTool === "line") {
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(currentX, currentY);
        ctx.stroke();
    }
}

function stopDrawing(e) {
    if (!isDrawing || !canvas || currentTool === "text") return;
    
    const rect = canvas.getBoundingClientRect();
    const endX = (e.clientX - rect.left) * (canvas.width / rect.width);
    const endY = (e.clientY - rect.top) * (canvas.height / rect.height);
    
    annotationHistory.push({
        type: currentTool,
        startX, startY, endX, endY
    });
    
    isDrawing = false;
}

function redrawAnnotations() {
    annotationHistory.forEach(annotation => {
        ctx.strokeStyle = "#FF0000";
        ctx.lineWidth = 3;
        ctx.fillStyle = "rgba(255, 0, 0, 0.1)";
        
        if (annotation.type === "rect") {
            ctx.fillRect(annotation.startX, annotation.startY, 
                        annotation.endX - annotation.startX, 
                        annotation.endY - annotation.startY);
            ctx.strokeRect(annotation.startX, annotation.startY, 
                          annotation.endX - annotation.startX, 
                          annotation.endY - annotation.startY);
        } else if (annotation.type === "circle") {
            const radius = Math.sqrt(
                (annotation.endX - annotation.startX) ** 2 + 
                (annotation.endY - annotation.startY) ** 2
            );
            ctx.beginPath();
            ctx.arc(annotation.startX, annotation.startY, radius, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
        } else if (annotation.type === "line") {
            ctx.beginPath();
            ctx.moveTo(annotation.startX, annotation.startY);
            ctx.lineTo(annotation.endX, annotation.endY);
            ctx.stroke();
        } else if (annotation.type === "text") {
            ctx.font = "20px Arial";
            ctx.fillStyle = "#FF0000";
            ctx.strokeStyle = "#FFFFFF";
            ctx.lineWidth = 3;
            ctx.strokeText(annotation.text, annotation.x, annotation.y);
            ctx.fillText(annotation.text, annotation.x, annotation.y);
        }
    });
}

function clearAnnotations() {
    if (!canvas || !ctx) return;
    annotationHistory = [];
    ctx.drawImage(originalImage, 0, 0);
}