import nltk
from collections import defaultdict
import pandas as pd
import urllib.request
import re
import math

# Read a CSV file into a list of strings
def read_dataset(filename):
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)
        columns_list = list(df.columns)
        if columns_list[0] in ['Unnamed: 0', 'Unnamed', 'unnamed: 0', 'unnamed: 0']:
            columns_list.pop(0)
        num_columns = len(columns_list)
        
    elif filename.endswith('.tsv'):
        df = pd.read_csv(filename, sep='\t')
        columns_list = list(df.columns)
        if columns_list[0] in ['Unnamed: 0', 'Unnamed', 'unnamed: 0', 'unnamed: 0']:
            columns_list.pop(0)
        num_columns = len(columns_list)
        
    elif filename.endswith('.xlsx') or filename.endswith('.xls'):
        df = pd.read_excel(filename)
        columns_list = list(df.columns)
        if columns_list[0] in ['Unnamed: 0', 'Unnamed', 'unnamed: 0', 'unnamed: 0']:
            columns_list.pop(0)
        num_columns = len(columns_list)
        
    else:
        raise ValueError('Unsupported file format')

    return df[columns_list[0]].tolist(), df[columns_list[1]].tolist(), num_columns

filename = 'C:/Users/fethi/Downloads/train_hate_speech.tsv'  # Replace with your dataset file path
text_list, class_list, num_columns = read_dataset(filename)

# Download Arabic stopwords list
url = 'https://raw.githubusercontent.com/mohataher/arabic-stop-words/master/list.txt'
with urllib.request.urlopen(url) as response:
    arabic_stopwords = response.read().decode('utf-8').splitlines()

# Download Algerian Arabic stopwords list
url2 = 'https://raw.githubusercontent.com/Damazouz/Algerian-Arabic-stop-words/main/algerian_arabic_stopwords.txt'
with urllib.request.urlopen(url2) as response:
    algerian_arabic_stopwords = response.read().decode('utf-8').splitlines()

# Download English stopwords list
nltk.download('stopwords')
english_stopwords = nltk.corpus.stopwords.words('english')

# Download French stopwords list
url3 = 'https://raw.githubusercontent.com/stopwords-iso/stopwords-fr/master/stopwords-fr.txt'
with urllib.request.urlopen(url3) as response:
    french_stopwords = response.read().decode('utf-8').splitlines()

# Combine all stopword lists
stopwords = arabic_stopwords + algerian_arabic_stopwords + english_stopwords + french_stopwords

# Tokenize the text using NLTK and remove stop words
all_tokens = []
word_freqs = defaultdict(lambda: defaultdict(int))
for i, text in enumerate(text_list):
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords]  # Remove stop words
    all_tokens.extend(tokens)
    for word in set(tokens):  # iterate through set of tokens to avoid double-counting
        word_freqs[class_list[i]][word] += 1


# Get list of all unique words
unique_words = sorted(set(all_tokens))

# Print the statistics of the dataset
print('Dataset Statistics:')
print('------------------')
print(f'Number of rows: {len(text_list)}')
print(f'Number of columns: {num_columns}')
print(f'Number of classes: {len(set(class_list))}')

# Count the number of rows in each class
class_counts = defaultdict(int)
for class_name in class_list:
    class_counts[class_name] += 1

# Print the number of rows in each class
print('\nNumber of Rows in Each Class:')
for class_name, count in class_counts.items():
    print(f'{class_name}: {count} rows')


# Calculate NPMI for each class
n = int(input("Enter the number of words you want to calculate the PMI for: "))  # number of top words to display for each class
for class_name in set(class_list):
    print(f"\nTop {n} PMI words for class '{class_name}':")
    word_scores = []
    for word in unique_words:
        if word_freqs[class_name][word] < 2:  # skip words with less than 2 occurrences
            continue
        p_word_given_class = word_freqs[class_name][word] / sum(word_freqs[class_name].values())
        p_word = all_tokens.count(word) / len(all_tokens)
        p_class = class_list.count(class_name) / len(class_list)
        pmi = max(0, math.log(p_word_given_class / (p_word * p_class)))
        npmi = pmi / (-1 * math.log(p_word_given_class))
        word_scores.append((word, npmi))
    word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
    for i in range(n):
        print(f"{word_scores[i][0]:<15} {word_scores[i][1]:.3f}")
