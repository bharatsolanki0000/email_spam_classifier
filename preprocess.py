import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# initialize
ps = PorterStemmer()
stopword_list = stopwords.words('english')
punctuation = string.punctuation


# preprocessing: lowercase, tokenize, remove special chars, stopwords, punctuation, stemming
def preprocess(row):

    # lowercase
    row = row.lower()

    # tokenize using a simple regex (avoids NLTK tokenizer dependencies)
    row = re.findall(r"\b\w+\b", row)

    # remove stopwords + punctuation + stemming
    y = [ps.stem(word) for word in row if word not in stopword_list and word not in punctuation]

    return " ".join(y)


def preprocess_transform(X):
    return [preprocess(text) for text in X]