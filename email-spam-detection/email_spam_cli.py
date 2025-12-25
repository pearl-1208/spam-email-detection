from spam_classifier_ml import model, vectorizer


def check_email():
    print("📧 Email Spam Checker")
    print("Type 'exit' to quit\n")

    while True:
        text = input("Enter email text: ")

        if text.lower() == "exit":
            break

        transformed = vectorizer.transform([text])
        result = model.predict(transformed)

        if result[0] == 1:
            print("⚠️ This email is SPAM\n")
        else:
            print("✅ This email is NOT spam\n")


if __name__ == "__main__":
    check_email()
