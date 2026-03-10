import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# download required resources (first time only)
nltk.download('punkt')
nltk.download('stopwords')

# initialize
ps = PorterStemmer()
stopword_list = stopwords.words('english')
punctuation = string.punctuation


# preprocessing: lowercase, tokenize, remove special chars, stopwords, punctuation, stemming
def preprocess(row):

    # lowercase
    row = row.lower()

    # tokenize
    row = nltk.word_tokenize(row)

    # remove special characters
    y = []
    for i in row:
        if i.isalnum():
            y.append(i)

    row = y[:]
    y.clear()

    # remove stopwords + punctuation + stemming
    y = [ps.stem(word) for word in row if word not in stopword_list and word not in punctuation]

    return " ".join(y)


def preprocess_transform(X):
    return [preprocess(text) for text in X]