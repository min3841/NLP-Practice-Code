import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('dataset/movies_metadata.csv/movies_metadata.csv', low_memory= False)

data = data.head(20000)

data['overview'] = data['overview'].fillna('')

vector = TfidfVectorizer(stop_words='english')

tfidf_metrics = vector.fit_transform(data['overview'])

cosine_metrics = cosine_similarity(tfidf_metrics,tfidf_metrics)

data_index = dict(zip(data['title'], data.index))

def get_recommendations(title, cosine_metrics = cosine_metrics):

    index = data_index[title]

    ti_cosine = list(enumerate(cosine_metrics[index]))

    cosine_sort = sorted(ti_cosine, key= lambda x:x[1], reverse= True)

    cosin_top = cosine_sort[1:11]

    top_index = [i[0] for i in cosin_top]

    return data['title'].iloc[top_index]


print(get_recommendations('The Dark Knight Rises'))