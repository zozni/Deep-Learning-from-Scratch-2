# chap02/count_method_big.py
# PTB 데이터셋에 통계기반기법을 적용한 예제. 
# 큰 행렬에 SVD를 적용해야 하므로 고속 SVD를 이용. 그러려면 sklearn 모듈을 설치해야 한다.
import sys
sys.path.append('..')
import numpy as np
from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print('Create Co-occurrence Matrix...')
C = create_co_matrix(corpus, vocab_size, window_size)

print('PPMI 계산...')
W = ppmi(C, verbose=True)

try:
    # truncated SVD
    from sklearn.utils.extmath import randomized_svd 
    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5,   # 메소드 이용!
                             random_state=None)
except:
    # SVD
    U, S, V = np.linalg.svd(W)

    
word_vecs = U[:, :wordvec_size]
querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)


# 단어의 의미 혹은 문법적인 관점에서 비슷한 단어들이 가까운 벡터로 나타났다. 