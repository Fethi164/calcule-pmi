from flask import Flask, render_template, request
import nltk
from collections import defaultdict
import pandas as pd
import urllib.request
import re
import math
import sqlite3
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the uploaded file and n value from the form
        file = request.files['file']
        n = int(request.form['n'])

        # Save the uploaded file to a temporary location
        filename = 'temp.csv'
        file.save(filename)

        # Process the dataset
        text_list, class_list, num_columns = read_dataset(filename)
        statistics = get_dataset_statistics(text_list, class_list, num_columns)
        top_pmi_words = calculate_top_pmi_words(text_list, class_list, n)

        # Save the file details to the database
        save_file_details(filename)

        # Delete the temporary file
        os.remove(filename)

        return render_template('index.html', statistics=statistics, top_pmi_words=top_pmi_words)

    return render_template('index.html')

def read_dataset(filename):
    if filename.endswith('.csv'):
        df = pd.read_csv(filename ,on_bad_lines='skip')
        columns_list = list(df.columns)
        if columns_list[0] in ['Unnamed: 0', 'Unnamed', 'unnamed: 0', 'unnamed: 0']:
            columns_list.pop(0)
        num_columns = len(columns_list)

    elif filename.endswith('.tsv'):
        df = pd.read_csv(filename, sep='\t' , encoding='utf-8', on_bad_lines='skip')
        columns_list = list(df.columns)
        if columns_list[0] in ['Unnamed: 0', 'Unnamed', 'unnamed: 0', 'unnamed: 0']:
            columns_list.pop(0)
        num_columns = len(columns_list)

    elif filename.endswith('.xlsx') or filename.endswith('.xls'):
        df = pd.read_excel(filename,on_bad_lines='skip')
        columns_list = list(df.columns)
        if columns_list[0] in ['Unnamed: 0', 'Unnamed', 'unnamed: 0', 'unnamed: 0']:
            columns_list.pop(0)
        num_columns = len(columns_list)

    else:
        raise ValueError('Unsupported file format')

    return df[columns_list[0]].tolist(), df[columns_list[1]].tolist(), num_columns

def get_dataset_statistics(text_list, class_list, num_columns):
    # Calculate dataset statistics
    num_rows = len(text_list)
    num_classes = len(set(class_list))

    class_counts = defaultdict(int)
    for class_name in class_list:
        class_counts[class_name] += 1

    statistics = {
        'num_rows': num_rows,
        'num_columns': num_columns,
        'num_classes': num_classes,
        'class_counts': class_counts
    }

    return statistics

def calculate_top_pmi_words(text_list, class_list, n):
    # Download stopwords and tokenize the text
    stopwords = download_stopwords()
    all_tokens = tokenize_text(text_list, stopwords)

    # Calculate word frequencies
    word_freqs = calculate_word_frequencies(text_list, class_list, all_tokens)

    # Get list of all unique words
    unique_words = sorted(set(all_tokens))

    # Calculate NPMI for each class
    top_pmi_words = {}
    for class_name in set(class_list):
        word_scores = calculate_pmi_scores(class_name, word_freqs, unique_words, all_tokens, class_list)
        top_words = get_top_n_words(word_scores, n)
        top_pmi_words[class_name] = top_words

    return top_pmi_words

def download_stopwords():
    # Download stopwords
    nltk.download('stopwords')
    url = 'https://raw.githubusercontent.com/mohataher/arabic-stop-words/master/list.txt'
    with urllib.request.urlopen(url) as response:
        arabic_stopwords = response.read().decode('utf-8').splitlines()

    url2 = 'https://raw.githubusercontent.com/Damazouz/Algerian-Arabic-stop-words/main/algerian_arabic_stopwords.txt'
    with urllib.request.urlopen(url2) as response:
        algerian_arabic_stopwords = response.read().decode('utf-8').splitlines()

    english_stopwords = nltk.corpus.stopwords.words('english')

    url3 = 'https://raw.githubusercontent.com/stopwords-iso/stopwords-fr/master/stopwords-fr.txt'
    with urllib.request.urlopen(url3) as response:
        french_stopwords = response.read().decode('utf-8').splitlines()

    stopwords = arabic_stopwords + algerian_arabic_stopwords + english_stopwords + french_stopwords

    return stopwords

def tokenize_text(text_list, stopwords):
    # Tokenize the text using NLTK and remove stop words
    all_tokens = []
    for text in text_list:
        text = re.sub(r'[^\w\s]', '', text)
        tokens = nltk.word_tokenize(text)
        tokens = [token for token in tokens if token.lower() not in stopwords]
        all_tokens.extend(tokens)

    return all_tokens

def calculate_word_frequencies(text_list, class_list, all_tokens):
    # Calculate word frequencies for each class
    word_freqs = defaultdict(lambda: defaultdict(int))
    for i, text in enumerate(text_list):
        tokens = nltk.word_tokenize(text)
        tokens = [token for token in tokens if token.lower() in all_tokens]
        for word in set(tokens):
            word_freqs[class_list[i]][word] += 1

    return word_freqs

def calculate_pmi_scores(class_name, word_freqs, unique_words, all_tokens, class_list):
    # Calculate PMI scores for each word in a given class
    word_scores = []
    for word in unique_words:
        if word_freqs[class_name][word] < 2:
            continue
        p_word_given_class = word_freqs[class_name][word] / sum(word_freqs[class_name].values())
        p_word = all_tokens.count(word) / len(all_tokens)
        p_class = class_list.count(class_name) / len(class_list)
        pmi = max(0, math.log(p_word_given_class / (p_word * p_class)))
        npmi = pmi / (-1 * math.log(p_word_given_class))
        word_scores.append((word, npmi))

    word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
    return word_scores

def get_top_n_words(word_scores, n):
    # Get the top n words based on PMI scores
    top_words = word_scores[:n]
    return top_words

def save_file_details(filename):
    # Connect to the database
    conn = sqlite3.connect('dataset.db')
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT
        )
    ''')

    # Insert the file details into the table
    cursor.execute('INSERT INTO files (filename) VALUES (?)', (filename,))
    conn.commit()

    # Close the database connection
    cursor.close()
    conn.close()

if __name__ == '__main__':
    app.run(debug=True)
