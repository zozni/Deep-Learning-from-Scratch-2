# count_method_small.py
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, ppmi


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

# SVD
U, S, V = np.linalg.svd(W)

print(C[0])  # 동시발생 행렬
print(W[0])  # PPMI 행렬
print(U[0])  # SVD
# 2차원으로 차원 축소하기
print(U[0, :2])

# 플롯  
# 각 단어를 2차원 벡터로 표현한 후 그래프로 그려본 것.
for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))   # annotate(word,x,y) 메소드는 2차원 그래프상에서 좌표 x,y 지점에 word에 담긴 텍스트를 그린다.
plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()