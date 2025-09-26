---

# ✋🤟 Sign Language Detection System

A **Deep Learning–based Sign Language Detection System** built with **TensorFlow/Keras** and deployed using **Streamlit**.
This project can recognize **sign language gestures** from images or webcam input using a trained **CNN model**.

---

## 📂 **Project Structure**

```
old-sign-language-detection/
│── dataset/                  # Training & testing dataset
│── debug_crops/              # Debugging cropped images
│── Documentation/            # Notes, reports, docs
│── venv/                     # Virtual environment (ignored in GitHub)
│── .gitignore                # Git ignore rules
│── best_asl_model.h5         # Trained CNN model
│── label_encoder.pkl         # Label encoder for gesture classes
│── main.py                   # Streamlit app entry point
│── old.py                    # Previous version of app
│── pre-run.py                # Pre-run testing script
│── debug.py                  # Debugging utility
│── Model-training.ipynb      # **Main notebook for training**
│── Sign Model Training.ipynb # Alternate training notebook
```

---

## 🛠️ **Installation Guide**

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/sign-language-detection.git
cd sign-language-detection
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Mac/Linux
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

💡 If you don’t have `requirements.txt`, create one with:

```txt
streamlit
tensorflow
opencv-python
numpy
pandas
scikit-learn
matplotlib
```

---

## ▶️ **Run the Application**

Start the Streamlit app with:

```bash
streamlit run main.py
```

🌐 By default, the app runs at: **[http://localhost:8501](http://localhost:8501)**

---

## 📊 **Model Details**

* 🧠 **Trained Model:** `best_asl_model.h5`
* 🏷 **Labels:** `label_encoder.pkl`
* 📒 **Training Notebook:** `Model-training.ipynb` (**main notebook**)

---

## 🔄 **Workflow**

1. 📷 Upload an image or enable webcam.
2. 🧹 Input is **preprocessed** using OpenCV.
3. 🤖 The trained **CNN model** predicts the gesture.
4. 🎯 Streamlit displays the **prediction result in real-time**.

---

## 🌍 **Future Enhancements**

* ✨ Extend to **continuous word/sentence recognition**.
* 📱 Optimize with **TensorFlow Lite** for mobile deployment.
* ☁️ Deploy to **Streamlit Cloud** or **Hugging Face Spaces**.

---

## 🤝 **Contributing**

Contributions, issues, and feature requests are welcome!
Feel free to **fork this repo** and submit a pull request 🚀

---
