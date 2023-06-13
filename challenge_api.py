import pandas as pd
import re
from flask import Flask, jsonify
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
from flask import request
import os
import string

app = Flask(__name__)


# Template
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
    info={
        'title': LazyString(lambda: 'API Documentation for Data Processing and Modeling'),
        'version': LazyString(lambda: '1.0.0'),
        'description': LazyString(lambda: 'Dokumentasi API untuk Data Processing dan Modeling'),
    },
    host=LazyString(lambda: request.host)
)

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}

swagger = Swagger(app, template=swagger_template, config=swagger_config)


# pandas dataframe for kamusalay
df_kamusalay = pd.read_csv('asset_challenge/new_kamusalay.csv',
                           encoding='latin-1', names=['find', 'replace'])
# Mapping for kamusalay
kamusalay_mapping = dict(zip(df_kamusalay['find'], df_kamusalay['replace']))


# change to lower case
def text_lower(text):
    text = text.lower()
    return text


# processing text function
def remove_unnecessary_char(text):
    text = re.sub(r'\\+n', ' ', text)
    text = re.sub('user', ' ', text)  # remove every username
    text = re.sub(r'\n', " ", text)  # remove every '\n'
    text = re.sub(r'(rt)', ' ', text)  # remove every retweet symbol
    text = re.sub(r'\\x.{2}', ' ', text)  # remove emoji
    text = re.sub(
        r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' ', text)  # remove every URL
    text = re.sub(r'&amp;', ' dan ', text)  # remove ampersant
    text = re.sub(r'&', ' dan ', text)  # remove ampersant
    text = re.sub('[%s]' % re.escape(string.punctuation),
                  ' ', text)  # remove punctuation

    text = re.sub(r'[^a-z ]', ' ', text)  # remove another word

    # remove single character
    text = ' '.join([i for i in text.split() if len(i) > 1])

    text = re.sub(r'  +', ' ', text)  # remove extra spaces
    text = text.rstrip().lstrip()  # remove rstrip and lstrip

    return text


# Cleaning by replacing 'alay' words
def handle_from_kamusalay(text):
    wordlist = text.split()
    clean_alay = ' '.join([kamusalay_mapping.get(x, x) for x in wordlist])
    return clean_alay


# FUNCTION FOR CLEANSING TEXT
def apply_cleansing_text(text):
    text = text_lower(text)
    # text = ' '.join([re.sub(
    #     '\d{1,4}', lambda m: num2words(int(m.group()), lang='id'), word) for word in text.split()])
    text = remove_unnecessary_char(text)
    text = handle_from_kamusalay(text)
    return text


# FUNCTION FOR CLEANSING FILE
# apply function for cleansing data and kamusalay
def apply_cleansing_file(data):
    # delete duplicated data
    data = data.drop_duplicates()

    # cleansing text to lower
    data['text_lower'] = data['text'].apply(lambda x: text_lower(x))
    # drop text column
    data.drop(['text'], axis=1, inplace=True)

    data['text_clean'] = data['text_lower'].apply(
        lambda x: remove_unnecessary_char(x))

    # apply kamusalay function
    data['text'] = data['text_clean'].apply(lambda x: handle_from_kamusalay(x))

    # drop text clean column
    data.drop(['text_lower', 'text_clean'], axis=1, inplace=True)

    return data


# ROUTE
# text processing
@swag_from("docs/text_processing.yml", methods=['POST'])
@app.route('/text-processing', methods=['POST'])
def text_processing():

    text = request.form.get('text')

    text = apply_cleansing_text(text)

    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'data': text,
    }

    response_data = jsonify(json_response)
    return response_data


# File processing
@swag_from("docs/text_processing_file.yml", methods=['POST'])
@app.route('/text-processing-file', methods=['POST'])
# processing text route
def text_processing_file():

    # Upladed file
    file = request.files.getlist('file')[0]

    # Import file csv to Pandas
    df = pd.read_csv(file, sep="/t", names=['text', 'label'])

    #  assertion
    assert any(df.columns == 'text')

    # apply cleansing
    df = apply_cleansing_file(df)

    # input text to list
    texts = df['text'].to_list()

    # cleansing text from list texts
    cleaned_text = []
    for text in texts:
        cleaned_text.append(text)

    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'data': cleaned_text
    }

    response_data = jsonify(json_response)
    return response_data


if __name__ == '__main__':
    app.run()
