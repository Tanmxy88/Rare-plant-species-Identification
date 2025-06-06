document.addEventListener('DOMContentLoaded', () => {
  const uploadArea = document.getElementById('upload-area');
  const fileInput = document.getElementById('plant-image');
  const previewGrid = document.getElementById('image-preview');
  const resultContainer = document.getElementById('result');
  const speciesName = document.getElementById('species-name');
  const confidence = document.getElementById('confidence');
  const conservationStatus = document.getElementById('rarity-status');
  const description = document.getElementById('species-description');
  const resultImg = document.getElementById('result-image');
  const loading = document.getElementById('loading');
  const resetBtn = document.getElementById('reset-btn');
  const form = document.getElementById('identify-form');
  const detailsBtn = document.getElementById('details-btn');
  const newIdentificationBtn = document.getElementById('new-identification');
  let detailsUrl = "";

  // Handle browse button
  const browseBtn = document.getElementById('browse-btn');
  if (browseBtn) {
    browseBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      fileInput.click();
    });
  }
  uploadArea.addEventListener('click', (e) => {
    if (e.target !== browseBtn && !(browseBtn && browseBtn.contains(e.target))) {
      fileInput.click();
    }
  });

  uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
  });
  uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
  });
  uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      handleImageUpload(e.dataTransfer.files);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
      handleImageUpload(e.target.files);
    }
  });

  resetBtn.addEventListener('click', resetForm);

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const files = fileInput.files;
    if (!files.length) {
      alert('Please upload at least one image');
      return;
    }
    if (files.length > 5) {
      alert('Maximum 5 images allowed');
      return;
    }
    loading.classList.remove('hidden');
    resultContainer.classList.add('hidden');
    try {
      const formData = new FormData();
      for (let i = 0; i < files.length; i++) {
        formData.append('images', files[i]);
      }
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) throw new Error('Identification failed');

      const data = await response.json();

      // Show the first image as the result image
      resultImg.src = URL.createObjectURL(files[0]);
      speciesName.textContent = data.species;
      confidence.textContent = `${(data.confidence * 100).toFixed(1)}%`;
      conservationStatus.textContent = data.conservation_status;
      description.textContent = data.description;
      detailsUrl = data.details_url || "";

      // Set badge class based on status
      const status = data.conservation_status.toLowerCase();
      conservationStatus.className = `badge ${
        status.includes('endangered') ? 'badge-endangered' :
        status.includes('vulnerable') ? 'badge-vulnerable' : 'badge-common'
      }`;

      resultContainer.classList.remove('hidden');
      loading.classList.add('hidden');
      resultContainer.scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
      alert(`Error: ${error.message}`);
      loading.classList.add('hidden');
    }
  });

  if (detailsBtn) {
    detailsBtn.addEventListener('click', () => {
      if (detailsUrl) {
        window.open(detailsUrl, '_blank');
      } else {
        alert('No further details available for this species.');
      }
    });
  }
  if (newIdentificationBtn) {
    newIdentificationBtn.addEventListener('click', resetForm);
  }

  function handleImageUpload(files) {
    previewGrid.innerHTML = '';
    Array.from(files).forEach(file => {
      if (!file.type.match('image.*')) {
        alert('Please upload an image file');
        return;
      }
      if (file.size > 5 * 1024 * 1024) {
        alert('File size exceeds 5MB');
        return;
      }
      const reader = new FileReader();
      reader.onload = (e) => {
        const div = document.createElement('div');
        div.className = 'preview-item';
        const img = document.createElement('img');
        img.src = e.target.result;
        img.className = 'preview-thumbnail';
        div.appendChild(img);
        previewGrid.appendChild(div);
      };
      reader.readAsDataURL(file);
    });
    previewGrid.classList.remove('hidden');
    uploadArea.classList.add('hidden');
  }

  function resetForm() {
    previewGrid.classList.add('hidden');
    uploadArea.classList.remove('hidden');
    previewGrid.innerHTML = '';
    fileInput.value = '';
    resultContainer.classList.add('hidden');
    loading.classList.add('hidden');
    detailsUrl = "";
  }
});
