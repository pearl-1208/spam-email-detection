# 📧 Email Spam Detection using Machine Learning

Email spam is a very common problem where unwanted or harmful emails reach users daily.  
In this project, I have built a **Machine Learning based Email Spam Detection system** that can automatically classify an email as **Spam** or **Ham (Not Spam)**.

The main focus of this project is to understand how **text data (emails)** can be processed and used in Machine Learning models.

---

## 🔍 Problem Description
Many users receive promotional or fraudulent emails that may contain fake offers, links, or misleading information.  
Manually identifying such emails is not efficient.  
This project solves the problem by using **Natural Language Processing (NLP)** and **Machine Learning** techniques.

---

## 🎯 Objectives of the Project
- To learn how text data is handled in Machine Learning
- To use NLP techniques for feature extraction
- To build a classification model to identify spam emails
- To evaluate the model using accuracy and other metrics
- To deploy the complete project on GitHub

---

## 🧠 Machine Learning Approach
This project uses a **Supervised Learning Classification** approach.

### Algorithm Used
- **Logistic Regression**
  - Logistic Regression is used to classify emails into two categories: Spam and Ham
  - It works well for binary classification problems

### Text Processing Technique
- **TF-IDF (Term Frequency–Inverse Document Frequency)**
  - Converts email text into numerical features
  - Helps the model understand important words in an email

---

## 📊 Dataset Information
The dataset is stored in a CSV file named `spam_data.csv`.

It contains:
- **Text**: Email message content
- **Label**: 
  - `spam` → unwanted or promotional email  
  - `ham` → normal or genuine email

---

## ⚙️ Project Workflow
1. Load the dataset using Pandas
2. Separate features (email text) and labels
3. Split data into training and testing sets
4. Convert text data into numerical form using TF-IDF
5. Train Logistic Regression model
6. Evaluate model performance
7. Predict spam or ham for user input emails

---

## 🛠️ Technologies Used
- Python
- Pandas
- Scikit-learn
- Natural Language Processing (NLP)
- VS Code
- Git & GitHub

---

## ▶️ How to Run the Project

### Step 1: Install required libraries
```bash
pip install pandas scikit-learn

Step 2: Run the Python file
python spam_classifier_ml.py

Step 3: Test with your own email

Enter any email text when prompted, and the model will predict whether it is spam or not.
