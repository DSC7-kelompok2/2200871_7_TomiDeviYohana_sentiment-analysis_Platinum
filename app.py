# API library
from flask import jsonify
from flasgger import Swagger
from flasgger import swag_from
from template_swagger import app, swagger_config, swagger_template, request
from cleansing import apply_cleansing_file, apply_cleansing_text, pd

# machine learning library
import pickle
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


swagger = Swagger(app, template=swagger_template, config=swagger_config)

# fiture extraction
max_feature = 100000
tokenizer = Tokenizer(num_words=max_feature, split=' ', lower=True)

# label
sentiment = ['negative', 'neutral', 'positive']


# load model and resource feature enginering
# mlp
file = open("resources/resource_of_mlp/feature_count_vect.p", "rb")
count_vect = pickle.load(file)
file.close()

file = open("model/model_of_mlp/model.p", "rb")
mlp_model = pickle.load(file)
file.close()


# lstm
file = open("resources/resource_of_lstm/tokenizer.pickle", "rb")
tokenizer_file_from_lstm = pickle.load(file)
file.close()

file = open("resources/resource_of_lstm/x_pad_sequences.pickle", "rb")
feature_file_from_lstm = pickle.load(file)
file.close()

model_file_from_lstm = load_model("model/model_of_lstm/model.h5")


# ENDPOINT

# ========================Neural Network===========================
@swag_from("docs/mlp.yml", methods=['POST'])
@app.route('/mlp', methods=['POST'])
def mlp():
    # get text
    original_text = request.form.get('text')
    # cleansing text
    text = [apply_cleansing_text(original_text)]
    # feature extraction
    text_feature = count_vect.transform(text)
    # prediction
    prediction = mlp_model.predict(text_feature)[0]

    # response API
    json_response = {
        'status_code': 200,
        'description': 'Results of sentiment Analysis using LSTM',
        'data': {
            'original text': original_text,
            'clean text': text,
            'sentiment': prediction
        }
    }
    response_data = jsonify(json_response)
    return response_data

# ----------------------Neural Network File-----------------------


@swag_from("docs/mlp_file.yml", methods=['POST'])
@app.route('/mlp_file', methods=['POST'])
# processing text route
def mlp_file():

    # Upladed file
    file = request.files.getlist('file')[0]

    # Import file csv to Pandas
    df = pd.read_csv(file, sep="delimiter", encoding="latin-1")

    #  assertion
    assert any(df.columns == 'text')

    # apply cleansing
    df = apply_cleansing_file(df)

    # input text to list
    texts = df.text.to_list()

    # append prediction with text
    final_text_with_sentiment = []
    for text in texts:
        sentiment_result = []
        sentiment_result.append(text)

        text = [text]
        # feature extraction
        text_feature = count_vect.transform(text)
        # prediction
        prediction = mlp_model.predict(text_feature)[0]
        sentiment_result.append(prediction)

        final_text_with_sentiment.append(sentiment_result)

    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah di cleansing dan hasil prediksi",
        'data': final_text_with_sentiment
    }

    response_data = jsonify(json_response)
    return response_data


# ====================================== LSTM-text =======================================
@swag_from("docs/lstm.yml", methods=['POST'])
@app.route('/lstm', methods=['POST'])
def lstm():
    # get text
    original_text = request.form.get('text')
    # cleansing text
    text = [apply_cleansing_text(original_text)]
    # feature extraction
    feature = tokenizer_file_from_lstm.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
    # inference
    prediction = model_file_from_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    # response API
    json_response = {
        'status_code': 200,
        'description': 'Results of sentiment Analysis using LSTM',
        'data': {
            'original text': original_text,
            'clean text': text,
            'sentiment': get_sentiment
        }
    }
    response_data = jsonify(json_response)
    return response_data


# ------------------------------------LSTM-File ---------------------------------------------
@swag_from("docs/lstm_file.yml", methods=['POST'])
@app.route('/lstm_file', methods=['POST'])
# processing text route
def lstm_file():

    # Upladed file
    file = request.files.getlist('file')[0]

    # Import file csv to Pandas
    df = pd.read_csv(file, sep="delimiter", encoding="latin-1")

    #  assertion
    assert any(df.columns == 'text')

    # apply cleansing
    df = apply_cleansing_file(df)

    # input text to list
    texts = df.text.to_list()

    # append prediction with text
    final_text_with_sentiment = []
    for text in texts:
        sentiment_result = []
        sentiment_result.append(text)

        text = [text]
        feature = tokenizer_file_from_lstm.texts_to_sequences(text)
        feature = pad_sequences(
            feature, maxlen=feature_file_from_lstm.shape[1])

        model = model_file_from_lstm
        prediction = model.predict(feature)
        polarity = np.argmax(prediction[0])
        sentiment_result.append(sentiment[polarity])

        final_text_with_sentiment.append(sentiment_result)

    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah di cleansing dan hasil prediksi",
        'data': final_text_with_sentiment
    }

    response_data = jsonify(json_response)
    return response_data


if __name__ == '__main__':
    app.run()
