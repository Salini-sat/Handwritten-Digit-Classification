# ✍️ Handwritten Digits Classification using SVM

🚀 A Machine Learning project that classifies handwritten digits (0–9) using **Support Vector Machines (SVM)** on the sklearn digits dataset.

---

## 📌 Project Overview

This project demonstrates how SVM can be applied to image classification tasks.
It includes:

* 🧠 Non-linear SVM (RBF Kernel)
* ⚡ Linear SVM using SGD (epoch-wise training)
* 📊 Performance evaluation with accuracy, confusion matrix, and classification report
* 📈 Visualization of training vs validation performance

---

## 📂 Dataset

* Source: `sklearn.datasets.load_digits()`
* 🔢 Total Samples: 1797
* 🧾 Features: 64 (8x8 pixel images)
* 🎯 Classes: Digits from 0 to 9

---

## 🛠️ Tech Stack

* 🐍 Python
* 📦 NumPy
* 📊 Matplotlib
* 🤖 Scikit-learn

---

## ⚙️ Models Used

### 🔹 1. SVM with RBF Kernel

* Captures non-linear patterns
* High accuracy on test data
* Uses: `SVC(kernel='rbf')`

### 🔹 2. Linear SVM (SGDClassifier)

* Faster, scalable alternative
* Supports incremental (epoch-wise) learning
* Uses: `SGDClassifier(loss='hinge')`

---

## 📊 Results

### ✅ Accuracy

* RBF SVM achieves **high accuracy (~98%)**
* Linear SVM performs slightly lower but trains faster

### 📉 Evaluation Metrics

* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)

---

## 📈 Visualizations

* 🖼️ Sample digit predictions
* 📊 Accuracy vs Epochs (Linear SVM)
* 📉 Loss vs Epochs
* 📈 RBF SVM performance comparison

---

## ⚡ Key Learnings

* SVM performs very well for small-to-medium image datasets
* RBF kernel handles complex boundaries better
* SGD-based SVM is useful for large-scale or streaming data
* Visualization helps understand model performance trends

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/svm-digit-classifier.git

# Navigate to project folder
cd svm-digit-classifier

# Install dependencies
pip install -r requirements.txt

# Run the script
python main.py
```

---

## 📌 Future Improvements

* 🔍 Hyperparameter tuning (GridSearchCV)
* 🧠 Try CNN for better performance
* 🌐 Deploy as a web app
* 📱 Convert into a mobile ML app

---

## 🙋‍♀️ Author

**Salini Satpathy**

* 🎓 BTech CSE Student
* 💡 Interested in Data Science & ML

---

✨ *"Turning data into insights, one model at a time."*

