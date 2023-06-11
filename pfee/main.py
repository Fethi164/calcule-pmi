from flask import Flask, render_template, request
import nltk
from collections import defaultdict
import pandas as pd
import urllib.request
import re
import math

app = Flask(__name__)

# Read a CSV file into a list of strings
def read_dataset(file):
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        columns_list = list(df.columns)
        if columns_list[0] in ['Unnamed: 0', 'Unnamed', 'unnamed: 0', 'unnamed: 0']:
            columns_list.pop(0)
        num_columns = len(columns_list)
    elif file.filename.endswith('.tsv'):
        df = pd.read_csv(file, sep='\t')
        columns_list = list(df.columns)
        if columns_list[0] in ['Unnamed: 0', 'Unnamed', 'unnamed: 0', 'unnamed: 0']:
            columns_list.pop(0)
        num_columns = len(columns_list)
    elif file.filename.endswith('.xlsx') or file.filename.endswith('.xls'):
        df = pd.read_excel(file)
        columns_list = list(df.columns)
        if columns_list[0] in ['Unnamed: 0', 'Unnamed', 'unnamed: 0', 'unnamed: 0']:
            columns_list.pop(0)
        num_columns = len(columns_list)
    else:
        raise ValueError('Unsupported file format')
        
    return df[columns_list[0]].tolist(), df[columns_list[1]].tolist(), num_columns

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        n = int(request.form['n'])
        
        text_list, class_list, num_columns = read_dataset(file)
        
        # Download stopwords and perform dataset processing
        url = 'https://raw.githubusercontent.com/mohataher/arabic-stop-words/master/list.txt'
        with urllib.request.urlopen(url) as response:
            arabic_stopwords = response.read().decode('utf-8').splitlines()

        url2 = 'https://raw.githubusercontent.com/Damazouz/Algerian-Arabic-stop-words/main/algerian_arabic_stopwords.txt'
        with urllib.request.urlopen(url2) as response:
            algerian_arabic_stopwords = response.read().decode('utf-8').splitlines()

        nltk.download('stopwords')
        english_stopwords = nltk.corpus.stopwords.words('english')

        url3 = 'https://raw.githubusercontent.com/stopwords-iso/stopwords-fr/master/stopwords-fr.txt'
        with urllib.request.urlopen(url3) as response:
            french_stopwords = response.read().decode('utf-8').splitlines()

        stopwords = arabic_stopwords + algerian_arabic_stopwords + english_stopwords + french_stopwords

        all_tokens = []
        word_freqs = defaultdict(lambda: defaultdict(int))
        for i, text in enumerate(text_list):
            text = re.sub(r'[^\w\s]', '', text)
            tokens = nltk.word_tokenize(text)
            tokens = [token for token in tokens if token not in stopwords]  # Remove stop words
            all_tokens.extend(tokens)
            for word in set(tokens):
                word_freqs[class_list[i]][word] += 1
        
        unique_words = sorted(set(all_tokens))
        statistics = f"Number of rows: {len(text_list)}\n"
        statistics += f"Number of columns: {num_columns}\n"
        statistics += f"Number of classes: {len(set(class_list))}\n"
        
        class_counts = defaultdict(int)
        for class_name in class_list:
            class_counts[class_name] += 1
        
        statistics += "\nNumber of Rows in Each Class:\n"
        for class_name, count in class_counts.items():
            statistics += f"{class_name}: {count} rows\n"
        
        pmi_words = ""
        for class_name in set(class_list):
            pmi_words += f"\nTop {n} PMI words for class '{class_name}':\n"
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
            for i in range(n):
                pmi_words += f"{word_scores[i][0]:<15} {word_scores[i][1]:.3f}\n"

        return render_template('index.html', statistics=statistics, pmi_words=pmi_words)

    return render_template('index.html')

if __name__ == '__main__':
    app.run()
