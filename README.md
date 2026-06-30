# 💼 Salary Prediction with LSTM — Keras

A deep learning project that applies a **Long Short-Term Memory (LSTM)** network to salary prediction, built with Keras. This project demonstrates how sequence-capable neural network architectures can be applied to structured regression problems.

---

## 📌 Project Overview

This notebook walks through an end-to-end pipeline for predicting salary using an LSTM model:

- Load and explore the salary dataset
- Preprocess and normalize features
- Reshape data into the 3D tensor format required by LSTM
- Build and compile an LSTM model with Keras
- Train and evaluate the model
- Visualize training loss and prediction results

---

## 🧠 Model Architecture

```
Input → LSTM Layer → Dense Layer(s) → Output (Salary)
```

| Layer | Details |
|-------|---------|
| LSTM | Learns temporal/sequential patterns in feature space |
| Dense | Fully connected output for regression |
| Loss Function | Mean Squared Error (MSE) |
| Optimizer | Adam |

---

## 📂 Dataset

The project uses a salary dataset with features such as:

- **Years of Experience** — primary input feature
- **Salary** — target variable (continuous)

> The dataset is reshaped to `(samples, timesteps, features)` as required by Keras LSTM layers.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Keras / TensorFlow | LSTM model building & training |
| NumPy | Array operations & reshaping |
| Pandas | Data loading & preprocessing |
| Matplotlib | Visualizing loss curves & predictions |
| Scikit-learn | Train/test split, normalization |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/salary-lstm-keras.git
cd salary-lstm-keras
```

### 2. Install dependencies

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

### 3. Run the notebook

```bash
jupyter notebook Simple_Salary_LSTM_Keras22.ipynb
```

---

## 📊 Results

The model was trained to minimize MSE on the salary regression task. Predictions closely follow the actual salary curve, demonstrating that LSTM can capture patterns in structured tabular data when properly reshaped.

| Metric | Value |
|--------|-------|
| Loss Function | MSE |
| Framework | Keras (TensorFlow backend) |
| Task | Regression |

---

## 📁 Project Structure

```
salary-lstm-keras/
│
├── Simple_Salary_LSTM_Keras22.ipynb   # Main notebook
├── Salary_Data.csv                    # Dataset (if included)
└── README.md                          # Project documentation
```

---

## 🎯 Key Concepts Demonstrated

- Reshaping 2D tabular data into 3D LSTM-compatible tensors
- Building a regression model using LSTM in Keras
- Normalizing target and input features with `MinMaxScaler`
- Monitoring training via loss curves
- Comparing predicted vs. actual salary values

---

## 👩‍💻 Author

**Toqa Mohamed Fathy**  
AI Engineering Student  
[GitHub](https://github.com/your-username) · [LinkedIn](https://linkedin.com/in/your-profile)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
