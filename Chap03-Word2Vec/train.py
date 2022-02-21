from pickletools import optimize
import sys
sys.path.append('..')
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW
from common.util import preprocess, create_contexts_target, convert_one_hot

window_size = 1
hidden_size = 5
batch_size = 3 
max_epoch = 1000

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

# 학습 데이터로부터 미니배치를 선택한 다음, 신경망에 입력해 기울기를 구하고 그 기울기를 optimizer에 넘겨 매개변수를 갱신하는
# 일련의 작업을 수행.
# 학습을 거듭할수록 손실이 줄어드는 것을 알 수 있다.

# 학습이 끝난 후의 가중치 매개변수를 살펴보자.
# 입력 측 MatMul 계층의 가중치를 거내 실제 내용을 확인해 보자.
# 입력 측 MatMul 계층의 가중치는 인스턴스 변수 word_vecs에 저장되어 있다.

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])  # 이 코드는 word_vecs라는 이름으로 가중치를 꺼내느데, word_vecs의 각 행에는 대응하는 단어 ID의 분산표현이 저장되어 있다. 