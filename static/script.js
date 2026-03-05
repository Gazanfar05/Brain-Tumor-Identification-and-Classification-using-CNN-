function analyze() {
    const input = document.getElementById("imageInput");
    if (input.files.length === 0) {
        alert("Please upload an MRI image");
        return;
    }

    const formData = new FormData();
    formData.append("image", input.files[0]);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("result").classList.remove("hidden");

        if (data.error) {
            document.getElementById("prediction").innerText =
                "Error: " + data.error;
            document.getElementById("confidence").innerText = "";
            return;
        }

        if (data.tumor_status === "No Tumor") {
            document.getElementById("prediction").innerText =
                "Tumor Status: No Tumor";
            document.getElementById("confidence").innerText =
                "Confidence: " + data.confidence + "%";
        } else {
            document.getElementById("prediction").innerText =
                "Tumor Status: Tumor Detected";
            document.getElementById("confidence").innerText =
                "Tumor Type: " + data.tumor_type + " (" + data.confidence + "%)";
        }
    })
    .catch(() => alert("Prediction failed"));
}