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
        document.getElementById("prediction").innerText =
            "Prediction: " + data.prediction;
        document.getElementById("confidence").innerText =
            "Confidence: " + data.confidence + "%";
    })
    .catch(() => alert("Prediction failed"));
}