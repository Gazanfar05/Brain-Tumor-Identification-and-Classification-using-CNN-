// ============ PDF REPORT GENERATION ============
async function generatePDFReport() {
    try {
        // Check if result exists
        if (!currentResult) {
            alert("Error: No diagnosis results found. Please analyze an image first.");
            console.error("currentResult is:", currentResult);
            return;
        }

        const { jsPDF } = window.jspdf;
        if (!jsPDF) {
            alert("PDF library not loaded. Please refresh the page.");
            return;
        }
        
        console.log("Generating PDF with result:", currentResult);
        
        const doc = new jsPDF();
        const pageWidth = doc.internal.pageSize.getWidth();
        const pageHeight = doc.internal.pageSize.getHeight();
        let yPosition = 20;
        
        // Helper functions
        function addText(text, size = 12, color = [0, 0, 0], bold = false) {
            doc.setFontSize(size);
            doc.setTextColor(...color);
            doc.setFont("helvetica", bold ? "bold" : "normal");
            doc.text(text, 20, yPosition);
            yPosition += size / 2 + 5;
        }
        
        function addSection(title) {
            doc.setDrawColor(59, 92, 255);
            doc.setLineWidth(0.5);
            doc.line(20, yPosition, pageWidth - 20, yPosition);
            yPosition += 5;
            addText(title, 14, [59, 92, 255], true);
            yPosition += 3;
        }
        
        function checkPageBreak() {
            if (yPosition > pageHeight - 30) {
                doc.addPage();
                yPosition = 20;
            }
        }
        
        // Header
        doc.setFillColor(59, 92, 255);
        doc.rect(0, 0, pageWidth, 35, "F");
        doc.setTextColor(255, 255, 255);
        doc.setFontSize(24);
        doc.setFont("helvetica", "bold");
        doc.text("Brain MRI Analysis Report", 20, 20);
        
        yPosition = 50;
        doc.setTextColor(0, 0, 0);
        
        // Report Info
        addSection("Report Information");
        addText(`Date: ${new Date().toLocaleString()}`);
        addText(`Analysis Type: CNN-based Brain Tumor Classification`);
        
        checkPageBreak();
        
        // Add MRI Image
        try {
            const canvasElement = document.getElementById("annotationCanvas");
            if (canvasElement && canvasElement.width > 0) {
                addSection("Annotated MRI Image");
                const imgData = canvasElement.toDataURL("image/png");
                const imgWidth = pageWidth - 40;
                const imgHeight = (canvasElement.height / canvasElement.width) * imgWidth;
                doc.addImage(imgData, "PNG", 20, yPosition, imgWidth, imgHeight);
                yPosition += imgHeight + 10;
                checkPageBreak();
            }
        } catch (err) {
            console.log("Could not add image to PDF:", err);
        }
        
        // Diagnosis Results
        addSection("Diagnosis Results");
        const result = currentResult;
        
        doc.setFontSize(12);
        doc.setFont("helvetica", "bold");
        const statusColor = result.tumor_status === "No Tumor" ? [16, 185, 129] : [239, 68, 68];
        doc.setTextColor(...statusColor);
        doc.text(`Status: ${result.tumor_status}`, 20, yPosition);
        yPosition += 10;
        
        doc.setTextColor(0, 0, 0);
        if (result.tumor_type) {
            addText(`Tumor Type: ${result.tumor_type.toUpperCase()}`);
        }
        addText(`Confidence Score: ${result.confidence}%`);
        
        checkPageBreak();
        
        // Tumor Details
        let tumorType = result.tumor_type?.toLowerCase().replace(/ /g, "_") || "no_tumor";
        const tumorInfo = tumorDatabase[tumorType];
        
        if (tumorInfo) {
            addSection("Detailed Tumor Information");
            
            addText(`Name: ${tumorInfo.fullName}`, 11, [0, 0, 0], true);
            yPosition += 2;
            
            addText("Description:", 10, [0, 0, 0], true);
            doc.setFontSize(10);
            doc.setTextColor(0, 0, 0);
            const descLines = doc.splitTextToSize(tumorInfo.description, pageWidth - 40);
            doc.text(descLines, 20, yPosition);
            yPosition += descLines.length * 4 + 5;
            
            checkPageBreak();
            
            // Characteristics
            addText("Key Characteristics:", 10, [0, 0, 0], true);
            tumorInfo.characteristics.forEach(char => {
                addText(`• ${char}`, 9);
            });
            
            checkPageBreak();
            
            // Symptoms
            if (tumorInfo.symptoms.length > 0) {
                addText("Common Symptoms:", 10, [0, 0, 0], true);
                tumorInfo.symptoms.forEach(symptom => {
                    addText(`• ${symptom}`, 9);
                });
            }
            
            checkPageBreak();
            
            // Treatment Options
            addText("Treatment Options:", 10, [0, 0, 0], true);
            tumorInfo.treatment.forEach(treat => {
                addText(`• ${treat}`, 9);
            });
            
            // Additional Info
            checkPageBreak();
            addSection("Additional Information");
            addText(`Survival Rate: ${tumorInfo.survivalRate}`, 10);
            addText(`Prevalence: ${tumorInfo.prevalence}`, 10);
        }
        
        checkPageBreak();
        
        // Classification Details
        addSection("Classification Probabilities");
        
        if (result.all_predictions && Array.isArray(result.all_predictions)) {
            result.all_predictions.forEach(pred => {
                checkPageBreak();
                const barWidth = (pred.confidence / 100) * 50;
                doc.setFillColor(59, 92, 255);
                doc.rect(20, yPosition, barWidth, 5, "F");
                doc.setTextColor(0, 0, 0);
                doc.setFontSize(9);
                doc.text(`${pred.class.replace(/_/g, " ").toUpperCase()}: ${pred.confidence.toFixed(1)}%`, 75, yPosition + 4);
                yPosition += 10;
            });
        }
        
        // Clinical Notes
        const notesField = document.getElementById("annotationNotes");
        const notes = notesField ? notesField.value : "";
        if (notes && notes.trim()) {
            checkPageBreak();
            addSection("Clinical Notes");
            doc.setFontSize(10);
            const noteLines = doc.splitTextToSize(notes, pageWidth - 40);
            doc.text(noteLines, 20, yPosition);
            yPosition += noteLines.length * 4;
        }
        
        // Footer
        checkPageBreak();
        doc.setFontSize(8);
        doc.setTextColor(128, 128, 128);
        doc.text("DISCLAIMER: This report is generated by an AI system for demonstration purposes.", 20, pageHeight - 15);
        doc.text("It should not be used for medical diagnosis without professional medical consultation.", 20, pageHeight - 10);
        
        // Save PDF
        const filename = `MRI-Report-${new Date().getTime()}.pdf`;
        doc.save(filename);
        
        console.log("✓ PDF saved successfully:", filename);
        alert("✓ PDF Report generated and downloaded successfully!");
        
    } catch (error) {
        console.error("✗ PDF generation error:", error);
        alert("Error generating PDF: " + error.message);
    }
}