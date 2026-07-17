# 🦟 MalariaNet (MalNet)

MalariaNet (MalNet) is an AI-powered web application for malaria diagnosis from Giemsa-stained peripheral blood smear images. The system detects malaria parasites, identifies the *Plasmodium* species, predicts the parasite developmental stage, localizes infected cells with bounding boxes, and generates a simple diagnostic report.

## ✨ Features

- AI-powered malaria parasite detection
- *Plasmodium* species classification
- Parasite stage identification
- Bounding box localization of infected cells
- Confidence score prediction
- Parasitemia estimation
- Web-based interface using Flask
- YOLOv8 object detection model

## 🛠️ Technologies Used

- Python
- Flask
- Ultralytics YOLOv8
- OpenCV
- PyTorch
- HTML & CSS

## 📁 Project Structure

```text
MalariaNet/
│
├── app.py
├── prediction.py
├── train.py
├── dataset.py
├── requirements.txt
├── malaria.yaml
├── models/
│   └── best.pt
├── static/
├── uploads/
└── templates/
```

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/MalariaNet.git
```

Move into the project folder:

```bash
cd MalariaNet
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## 📊 Model Output

For each uploaded blood smear image, MalariaNet provides:

- Diagnosis (Positive / Negative)
- Predicted *Plasmodium* species
- Predicted parasite stage
- Detection confidence
- Parasitemia estimation
- Annotated detection image

## ⚠️ Disclaimer

This project is intended **for research and educational purposes only**. It is **not a certified medical diagnostic system** and should not be used as a substitute for professional clinical diagnosis or laboratory confirmation.

The dataset exhibits a naturally imbalanced distribution of *Plasmodium* species and developmental stages. This reflects the epidemiological prevalence of malaria parasites, where *Plasmodium falciparum* ring forms are considerably more common than *Plasmodium malariae* or *Plasmodium ovale* stages. Consequently, classification performance is expected to be higher for frequently represented classes.

## 👨‍💻 Developer

**Bisrat Weldegiyorgis**

## 📄 License

This project is released for academic, educational, and research purposes.
