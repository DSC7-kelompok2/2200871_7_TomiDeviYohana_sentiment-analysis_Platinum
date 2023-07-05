import pandas as pd
import re
import string

# pandas dataframe for kamusalay
df_kamusalay = pd.read_csv('asset_challenge/new_kamusalay.csv',
                           encoding='latin-1', names=['find', 'replace'])
# Mapping for kamusalay
kamusalay_mapping = dict(zip(df_kamusalay['find'], df_kamusalay['replace']))


# merubah kalimat menjadi huruf kecil
def text_lower(text):
    text = text.lower()
    return text


# processing text function
def remove_unnecessary_char(text):

    text = re.sub(r'\\+n', ' ', text)  # remove every '\\n'
    text = re.sub(r'\n', " ", text)  # remove every '\n'
    # remove every username
    text = ' '.join(
        re.sub("([@#][A-Za-z0-9_]+)|(\w+:\/\/\S+)", " ", text).split())
    # text = ' '.join([i for i in text.split() if  i != 'rt'])   # remove every retweet symbol
    text = re.sub(r'\\x.{2}', ' ', text)  # remove emoji
    text = re.sub(
        r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', ' ', text)
    text = re.sub(r'&amp;', ' dan ', text)  # remove ampersant
    text = re.sub(r'&&+', ' ', text)  # remove ampersant
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
    # implement menghapus_unnecessary_char function
    data['text_clean'] = data['text_lower'].apply(
        lambda x: remove_unnecessary_char(x))
    # apply kamusalay function
    data['text'] = data['text_clean'].apply(lambda x: handle_from_kamusalay(x))
    # drop text clean column
    data.drop(['text_lower', 'text_clean'], axis=1, inplace=True)

    return data
