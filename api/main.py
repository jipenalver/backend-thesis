from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os

app = FastAPI()

@app.get("/")
async def main():
    return {"message": "SIAM Backend"}

# Load the model and vectorizer
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model-old.pkl")
vectorizer_path = os.path.join(base_dir, "tfidf.pkl")

with open(model_path, 'rb') as f:
    classifier = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# Input model for POST request
class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    try:
        # Preprocess the input text
        text = input.text.lower()
        text_vector = vectorizer.transform([text]).toarray()
        
        # Debugging logs
        print(f"Preprocessed Text: {text}")
        print(f"TF-IDF Vector Shape: {text_vector.shape}")

        prediction = classifier.predict(text_vector)
        print(f"Raw Prediction: {prediction}")
        
        probability = classifier.predict_proba(text_vector).max() if hasattr(classifier, "predict_proba") else None
        print(f"Confidence Score: {probability}")
        
        return {
            "input_text": input.text,
            "prediction": prediction[0],
            "confidence": f"{probability:.2f}" if probability else "Not available"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

