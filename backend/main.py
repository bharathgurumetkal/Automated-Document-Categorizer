from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow all domains, or specify a list of allowed domains


# Load the model
def load_model():
    # Load the dataset and train the model again (or load a pre-trained model if saved)
    data = pd.read_csv('C:\\Users\\Bhara\\Downloads\\NewsCategorizer.csv.zip')

    # Prepare the data
    data['text'] = data['headline'] + " " + data['short_description']
    data = data.dropna(subset=['text', 'category'])

    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['category'], test_size=0.2, random_state=42
    )

    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)

    # Save the model if not already saved
    with open('news_category_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    return model

# Load the trained model (or load from saved file if available)
model = load_model()

@app.route('/predict', methods=['POST'])
def predict_category():
    # Get the text input from the request
    data = request.get_json()
    new_text = data.get('text', '')

    # Make prediction and get probability
    prediction = model.predict([new_text])[0]
    probability = model.predict_proba([new_text]).max() * 100  # Get max probability

    # Return the category and the confidence
    return jsonify({
        'category': prediction,
        'confidence': round(probability, 2)  # Round to 2 decimal places
    })

if __name__ == '__main__':
    app.run(debug=True)