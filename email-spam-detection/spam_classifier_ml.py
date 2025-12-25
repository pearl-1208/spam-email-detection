import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def main():
    # 1) CSV file load karna
    data = pd.read_csv("spam_data.csv")
    print("Dataset sample:")
    print(data.head(), "\n")

    # 2) Text (X) aur label (y) alag karna
    X = data["text"]          # emails
    y = data["label"]         # spam / ham

    # 3) Train–Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) Text ko numbers me convert karna (TF-IDF)
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 5) ML model banana (Logistic Regression)
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # 6) Testing & evaluation
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.2f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 7) User se input leke predict karna
    print("\n=== Test your own email text ===")
    user_email = input("Enter email text: ")

    user_tfidf = vectorizer.transform([user_email])
    prediction = model.predict(user_tfidf)[0]

    print(f"\nPrediction: This email is **{prediction.upper()}**")


if __name__ == "__main__":
    main()
