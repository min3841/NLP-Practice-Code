from konlpy.tag import Okt

def build_bag_of_words(dm):

    dm = dm.replace('.',"")
    okt = Okt()

    token = okt.morphs(dm)

    text_to_index = {}

    bow = []

    for word in token:

        if word not in text_to_index.keys():

            text_to_index[word] = len(text_to_index)

            bow.insert(len(text_to_index) - 1, 1)

        else:

            index = text_to_index.get(word)

            bow[index] = bow[index] + 1

    return text_to_index, bow

doc1 = "정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다."
doc2 = '소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.'

doc3 = doc1 + ' ' + doc2
vocab, bow = build_bag_of_words(doc3)

print('vocabulary :', vocab)
print('bag of words vector :', bow)