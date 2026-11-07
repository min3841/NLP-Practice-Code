from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

raw_text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."

sentences = sent_tokenize(raw_text)

stopword = stopwords.words("english")

vocab = {}

preprocessed_sentences = []

for sentence in sentences:

    tokenized_sentences = word_tokenize(sentence)
    result = []

    for word in tokenized_sentences:
        word = word.lower()

        if word not in stopword:

            if len(word) >2:

                result.append(word)

                if word not in vocab:
                    vocab[word] = 0
                vocab[word] += 1
    preprocessed_sentences.append(result)

vocab_sorted = sorted(vocab.items(), key=lambda x:x[1], reverse= True)

word_to_index = {word[0] : index + 1 for index, word in enumerate(vocab_sorted) if word[1] > 2}

word_to_index["oov"] = len(word_to_index) + 1

word_encodings = []

for words in preprocessed_sentences:

    word_encoding = []

    for word in words:

        try:

            word_encoding.append(word_to_index[word])
        except KeyError:

            word_encoding.append(word_to_index["oov"])

    word_encodings.append(word_encoding)
    
print(word_encodings)
