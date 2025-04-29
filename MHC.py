import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import gradio as gr

# Load dataset
data = pd.read_csv(r"C:\Users\cnity\Downloads\Mental_Health_FAQ.csv")

# Print column names to verify
print("Columns in the dataset:", data.columns)

# Update column names if necessary
data.columns = data.columns.str.strip()  # Remove extra spaces
questions = data["question"].tolist()  # Use the correct column name
answers = data["answer"].tolist()      # Use the correct column name

# Train TF-IDF + MLP Classifier
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)
model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000)
model.fit(X, answers)

# Add crisis keyword detection
CRISIS_KEYWORDS = ["suicide", "self harm", "kill myself"]
CRISIS_RESPONSE = "Please contact a professional immediately. Here’s a hotline: 1-800-273-TALK."

def chatbot_response(user_input):
    # Check for crisis keywords first
    if any(keyword in user_input.lower() for keyword in CRISIS_KEYWORDS):
        return CRISIS_RESPONSE
    # Predict answer
    input_vec = vectorizer.transform([user_input])
    predicted_answer = model.predict(input_vec)[0]
    return f"Bot: {predicted_answer}"

# Create Gradio interface
gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(placeholder="How are you feeling today?"),
    outputs="text",
    title="Mental Health Chatbot (Demo)",
    description="A simple FAQ-based mental health assistant. NOTE: This is a demo, not a substitute for professional help."
).launch(share=True)  # Run locally or set `share=False`
