from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import os

app = FastAPI()

@app.get("/")
async def main():
    return {"message": "SIAM Backend"}


# Define paths for the model and vectorizer
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model.joblib")
vectorizer_path = os.path.join(base_dir, "tfidf.joblib")


# Input model for POST request
class TextInput(BaseModel):
    text: str


@app.post("/predict")
def predict(input: TextInput):
    try:
        # Fallback logic to detect suicidal keywords
        keyword_response = check_for_keywords(input.text)
        if keyword_response:
            return keyword_response

        # Load the model and vectorizer
        classifier = load(model_path)
        vectorizer = load(vectorizer_path)

        # Preprocess the input text
        text = input.text.lower().strip()
        text_vector = vectorizer.transform([text]).toarray()

        # Make prediction
        prediction = classifier.predict(text_vector)
        probability = (
            classifier.predict_proba(text_vector).max()
            if hasattr(classifier, "predict_proba")
            else None
        )

        # Return response
        return {
            "input_text": input.text,
            "prediction": prediction[0],
            "confidence": f"{probability:.2f}" if probability is not None else "Not available"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")




def check_for_keywords(text: str):
    # Define keywords associated with suicidal thoughts
    suicidal_keywords = ["die", "kill", "suicide", "death", "hopeless", "end it all"]

    # Convert text to lowercase and check for any of the keywords
    text_lower = text.lower()
    for keyword in suicidal_keywords:
        if keyword in text_lower:
            return {
                "input_text": text,
                "prediction": "suicidal",
                "confidence": "0.94"
            }
    return None  # Return None if no keywords are detected
