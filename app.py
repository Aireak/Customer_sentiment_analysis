from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load saved model, vectorizer, and label encoder
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Initialize Flask app
app = Flask(__name__)

# Define the keyword-based classification function
positive_keywords = {'excellent', 'great', 'amazing', 'awesome', 'good', 'love', 'best', 'fantastic', 'wonderful', 'clean', 'warm', 'friendly', 
                    'delight', 'smile', 'authentic', 'awesome', 'fabulous'}
negative_keywords = {'bad', 'horrible', 'awful', 'worst', 'terrible', 'poor', 'hate', 'disappointed', 'wrong', 'watery', 'over priced', 'unhealthy'}
neutral_keywords = {'okay', 'average', 'fine', 'decent', 'satisfactory', 'neutral', 'understandable', 'mild', 'subtle', 'acceptable', 'lukewarm'}

def keyword_classification(text):
    words = set(text.lower().split())  # Convert text into a set of words
    if words & positive_keywords:
        return 'positive'
    elif words & negative_keywords:
        return 'negative'
    elif words & neutral_keywords:
        return 'neutral'
    else:
        return 'unknown'  # If no matching keywords are found

@app.route('/')
def home():
    return "Sentiment Analysis API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON request data
        data = request.json
        review_text = data['review']

        # Transform text using the saved vectorizer
        review_vectorized = vectorizer.transform([review_text]).toarray()

        # Apply keyword classification
        keyword_sentiment = keyword_classification(review_text)

        # Encode the keyword-based classification
        keyword_encoded = label_encoder.transform([keyword_sentiment])[0]

        # Combine both features
        review_combined = np.hstack((review_vectorized, np.array([[keyword_encoded]])))

        # Predict sentiment
        prediction = model.predict(review_combined)[0]

        # Return result
        return jsonify({'sentiment': int(prediction)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
