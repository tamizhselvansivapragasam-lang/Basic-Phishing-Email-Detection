# ==========================================
# PHISHING EMAIL DETECTION
# Predict New Emails
# ==========================================

import joblib

# Load the saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("=" * 50)
print("     PHISHING EMAIL DETECTION SYSTEM")
print("=" * 50)

while True:
    print("\nEnter an email message")
    email = input("> ")

    # Convert email into TF-IDF features
    email_features = vectorizer.transform([email])

    # Predict
    prediction = model.predict(email_features)[0]

    # Display result
    if prediction == 1:
        print("\nPrediction : PHISHING")
    else:
        print("\nPrediction : SAFE")

    # Ask user if they want to continue
    choice = input("\nCheck another email? (y/n): ").lower()

    if choice != "y":
        print("\nThank you for using the Phishing Email Detection System.")
        break