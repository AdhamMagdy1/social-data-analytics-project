from flask import Flask, request, jsonify
from flask_cors import CORS
import fasttext
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import nltk


app = Flask(__name__)
CORS(app)


# Load FastText model
model = fasttext.load_model("coursara_model.bin")

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()


df=pd.read_csv("recommendations.csv")


def load_institutions(path):
    """
    load the institutions as numpy array with the same order of training
    """
    with open(path, 'rb') as file:
        # A new file will be loaded 
        model1 = pickle.load(file)
    return model1

model1 = load_model("predict_rate.h5")
institutions = load_institutions("institutions.pkl")



# Define API endpoint for sentiment analysis
@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    sentence = data['sentence']

    fasttext_result = get_fasttext(sentence)
    vader_result = get_vader(sentence)
    vader_sentiment = format_vader_sentiment(vader_result)

    response = {
        "sentence": "sentence is {}".format(sentence),
        "fasttext_prediction": fasttext_result,
        "vader_sentiment": vader_sentiment,
        # "vader_scores": vader_result  # Include VADER sentiment scores in the response
    }

    return jsonify(response)

@app.route('/recommendation', methods=['POST'])
def recommendation():
    data = request.get_json()
    course = data['course']

    recommendation = getrecommendation(course)

    response = {
        "course": course,
        "recommendation": recommendation
    }

    return jsonify(response)

@app.route('/predictrate', methods=['POST'])
def predict_rate():
    data = request.json
    review = data['review']
    institution = data['institution']
    predicted_rate = get_rate(review, institution, model=model1)
    predicted_rate = int(predicted_rate)
    return jsonify({'predicted_rate': "predicate rate is {}".format(predicted_rate),'review':review,'institution':institution})

# Function to perform sentiment analysis using FastText
def get_fasttext(sentence):
    prediction, _ = model.predict(sentence)
    # return "Sentence: {}\nFasttext Prediction: {}".format(sentence, prediction[0])
    return "Fasttext Prediction: {}".format(prediction[0])

# Function to perform sentiment analysis using VADER
def get_vader(sentence):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(sentence)
    return scores

def format_vader_sentiment(score):
    compound_score = score['compound']
    if compound_score > 0.5:
        return "sentence is positive and compound score is {}".format(compound_score)
    elif compound_score < 0.5 and compound_score > 0:
        return "sentence is neutral and compound score is {}".format(compound_score)
    else:
        return "sentence is negative and compound score is {}".format(compound_score)

def getrecommendation(course):
    name=df['Course'].tolist()
    rec=df['Recommendation'].tolist()
    for i,j in zip(name,rec):
        if i==course:
            return "The course {} is {}".format(i,j)

def replace_emojis_with_text(text):
    # Use the emoji library's `demojize` function to replace emojis with text
    # The `demojize` function returns a string where emojis are replaced with their short names
    # surrounded by colons, e.g., ":smile:", ":heart:", etc.
    # You can then remove the colons or customize the output to suit your needs
    return emoji.demojize(text, delimiters=("", ""))

def clean_text(series):
    series = series.apply(lambda x:  re.sub(r'_+', ' ', x))

    series = series.apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
    
    series = series.str.lower()
    
    series = series.apply(lambda x: ''.join([c for c in x if c not in punctuation]))
    
    # remove numbers
    series = series.apply(lambda x: re.sub(r'\d+', '', x))
    
    # remove extra whitespaces
    series = series.str.strip()
    
    # remove extra whitespaces
    series = series.apply(lambda x: re.sub(' +', ' ', x))
    
    return series

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_text = [token.lemma_ for token in doc]
    return ' '.join(lemmatized_text)

def get_institution(x, institutions = institutions):
    """
    one hot encoding to the given institution
    """
    output = [1  if institution == x else 0 for institution in institutions] 
    return output

def getVader(text):
    return analyzer.polarity_scores(text)["compound"]

def get_rate(review, institution, model = model1):
    """
    this function returns the predicted rate according to the model
    inputs :
    review(String)------------> review to be evaluated 
    institution(String)-------> institution that is responsible for the course
    model(Keras.layers.Model)-> Dnn model that is trained to predict rate
    output:
    rate(String)---------> classes ranges from 1 to 5
    """
    vader = getVader(review) 
    inputs  = get_institution(institution)
    inputs.insert(0,vader)
    inputs = np.reshape(np.array(inputs), (1,len(inputs)))
    y = model.predict(inputs, verbose = False)
    y = np.argmax(y)+1
    return y


if __name__ == '__main__':
    app.run(debug=True)
