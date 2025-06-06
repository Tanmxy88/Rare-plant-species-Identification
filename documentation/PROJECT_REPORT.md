# Project Report: AI-Powered Plant Conservation System

## Abstract
This project presents an AI-powered web-based Plant Conservation System to accurately identify rare and endangered plant species using a hybrid deep learning model (CNN + KNN). The system aims to support biodiversity conservation efforts by providing fast and reliable identification tools.

---

## Introduction

Conservation of plant biodiversity requires accurate identification of species, especially rare and endangered ones. Traditional methods are time-consuming and require expert knowledge. This system leverages AI to automate species identification from images.

---

## Objectives

- Develop a high-accuracy AI model for plant species identification.
- Build a user-friendly web interface for researchers and conservationists.
- Provide detailed species information and contact options for collaboration.

---

## Methodology

### Data Collection

- Images collected from publicly available plant databases.
- Dataset divided into training, validation, and test sets.

### AI Model

- Convolutional Neural Network (CNN) for feature extraction.
- K-Nearest Neighbors (KNN) classifier on CNN features.
- Model trained using transfer learning techniques.

### Backend

- Flask API serving the AI model.
- Handles image uploads and returns predictions.

### Frontend

- Responsive HTML/CSS/JS interface.
- Pages: Home, Identify, About, Contact, Privacy, Terms.

---

## Results

- Achieved over 95% accuracy on test dataset.
- Fast inference time (~1 second per image).
- Positive user feedback during pilot testing.

---

## Challenges

- Dataset imbalance among species classes.
- Ensuring model generalization to varied image qualities.

---

## Future Work

- Expand dataset to include more species.
- Add mobile app support.
- Integrate geolocation data for conservation mapping.

---

## Conclusion

The Plant Conservation System demonstrates the effective use of AI for biodiversity preservation, providing a scalable tool for species identification.

---

## References

- [1] Relevant research papers and datasets.
- [2] AI and deep learning frameworks.

---

## Appendices

- Code repository link.
- Model training logs.


