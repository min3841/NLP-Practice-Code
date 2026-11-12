import pandas as pd # 데이터프레임
from math import log # IDF 계산

docs = [
  '먹고 싶은 사과',
  '먹고 싶은 바나나',
  '길고 노란 바나나 바나나',
  '저는 과일이 좋아요'
] 


vocab = list(set(i for word in docs for i in word.split()))

vocab.sort()


N = len(docs)

def tf(d,t):

    return d.count(t)

def idf(t):

    df = 0

    for dos in docs:

        df += t in dos

    return log(N/(1+df))


def tfidf(d,t):

    return tf(d,t) * idf(t)



result = []

for i in range(len(docs)):

    result.append([])

    d = docs[i]

    for j in range(len(vocab)):

        t = vocab[j]

        result[i].append(tf(d,t))


tf_ = pd.DataFrame(result, columns= vocab)



result = []

for j in range(len(vocab)):

    result.append([])

    t = vocab[j]

    result[j].append(idf(t))

idf_ = pd.DataFrame(result, columns=["IDF(역 문서 빈도)"])


result = []

for i in range(len(docs)):

    result.append([])

    d = docs[i]

    for j in range(len(vocab)):

        t = vocab[j]

        result[i].append(tfidf(d,t))


tf_idf = pd.DataFrame(result, columns= vocab)

print(tf_idf)




"""
#사이킷 런 사용

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',    
]

vector = CountVectorizer()

tfidfv = TfidfVectorizer().fit(corpus)


print(vector.fit_transform(corpus).toarray())
print(tfidfv.transform(corpus).toarray())
"""

