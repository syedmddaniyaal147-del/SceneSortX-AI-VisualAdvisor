const moduleSelect = document.getElementById("moduleSelect");
const uploadBox = document.getElementById("uploadBox");
const imageInput = document.getElementById("imageInput");
const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");
const analyzeBtn = document.getElementById("analyzeBtn");
const results = document.getElementById("results");
const resultJSON = document.getElementById("resultJSON");
const loading = document.getElementById("loading");
const clearBtn = document.getElementById("clearBtn");

let selectedFile = null;

uploadBox.addEventListener("click", () => imageInput.click());

imageInput.addEventListener("change", (e) => {
    selectedFile = e.target.files[0];
    showPreview();
});

function showPreview() {
    if (!selectedFile) return;

    previewImage.src = URL.createObjectURL(selectedFile);
    previewContainer.classList.remove("hidden");
    analyzeBtn.classList.remove("hidden");
}

analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return alert("Upload an image first!");

    const module = moduleSelect.value;

    const endpoint = `http://127.0.0.1:5000/${module}`;

    const formData = new FormData();
    formData.append("image", selectedFile);

    results.classList.add("hidden");
    loading.classList.remove("hidden");

    try {
        const response = await fetch(endpoint, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        loading.classList.add("hidden");
        results.classList.remove("hidden");

        resultJSON.textContent = JSON.stringify(data, null, 4);

    } catch (err) {
        loading.classList.add("hidden");
        alert("Error: " + err);
    }
});

clearBtn.addEventListener("click", () => {
    selectedFile = null;
    previewContainer.classList.add("hidden");
    analyzeBtn.classList.add("hidden");
    results.classList.add("hidden");
    imageInput.value = "";
});
