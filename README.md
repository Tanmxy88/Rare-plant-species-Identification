# Plant Conservation System 🌿
AI-powered model to identify rare and endangered plant species using image-based analysis.

## 🚀 Overview
This project uses a hybrid deep learning approach combining Convolutional Neural Networks (CNNs) and K-Nearest Neighbors (KNN) to identify plant species from images. It is optimized for mobile and field deployment and aims to assist conservationists and ecologists in biodiversity preservation.

## 📁 Project Structure
plant-conservation-system/
├── ai_model/ # AI/ML core (preprocessing, model, training)
├── backend/ # API backend (Flask/FastAPI)
├── datasets/ # Image datasets (train/val/test)
├── documentation/ # Report, user guide, and PPT
├── frontend/ # Web frontend using Bootstrap
├── logs/ # Training logs
├── scripts/ # Utilities for labeling and splitting
├── tests/ # Unit tests for inference and API
├── .gitignore
├── README.md
└── requirements.txt

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/plant-conservation-system.git
cd plant-conservation-system

2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt

3. Train or Load Model
Run training (optional):

bash
Copy
Edit
python ai_model/train.py


4. Start Backend
bash
Copy
Edit
cd backend
uvicorn app:app --reload


5. Open Frontend
Open frontend/index.html in your browser to upload an image and get predictions