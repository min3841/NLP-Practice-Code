from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical
import numpy as np

raw_text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."

stop_word = stopwords.words("english")

sentences = sent_tokenize(raw_text)

vocab = {}

preprocesed_sentences = []

for words in sentences:

    tokenized_word = word_tokenize(words)

    result = []

    for word in tokenized_word:

        word = word.lower()

        if word not in stop_word:

            if len(word) > 2:

                result.append(word)

                if word not in vocab:

                    vocab[word] = 0

                vocab[word] += 1

    preprocesed_sentences.append(result)

vocab = 5

tokenizer = Tokenizer(num_words = vocab + 2, oov_token = "OOV")

tokenizer.fit_on_texts(preprocesed_sentences)

encoding = tokenizer.texts_to_sequences(preprocesed_sentences)

encoding_word = tokenizer.word_index

padding = pad_sequences(encoding, padding="post", maxlen=5, truncating="post")

def one_hot_encoding(padding, vocab_size):

    result = []

    for preprocesed in padding:

        sentence_vectors = []

        for index in preprocesed:
                one_hot_vector = [0] * vocab_size
                one_hot_vector[index] = 1
                sentence_vectors.append(one_hot_vector)

        result.append(sentence_vectors)

    return result

hot_encodig = one_hot_encoding(padding, vocab + 2)

print(np.array(hot_encodig, dtype=float))



# or one_hot = to_categorical(padding) #케라스 원-핫 인코딩 함수 to_categorical()

# print(one_hot)




















