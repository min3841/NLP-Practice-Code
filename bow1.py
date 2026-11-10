from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

corpus = ['you know I want your love. because I love you.']

stopword = stopwords.words("english")

vector = CountVectorizer(stop_words=stopword)

print('bag of words vector :',vector.fit_transform(corpus).toarray())
print('vocabulary :',vector.vocabulary_)