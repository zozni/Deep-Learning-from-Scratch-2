""" 유사 단어 검색 후 유사도가 높은 순서대로 출력 """

import sys
sys.path.append('..')
from common.util import preprocess, create_co_matrix, most_similar

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

most_similar('you', word_to_id, id_to_word, C, top=5)


# 유사도 결과가 이렇게 나오는 이유는 
# 말뭉치(corpus)의 크기가 너무 작다는 것이 원인.