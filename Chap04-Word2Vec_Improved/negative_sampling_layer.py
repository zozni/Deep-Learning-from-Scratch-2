# Chap04/negative_sampling_layer.py
import sys
sys.path.append('..')
import collections
from common.np import *
from common.layers import Embedding, SigmoidWithLoss

class EmbeddingDot:
    def __init__(self, W):             
        self.embed = Embedding(W)  # Embedding 계층의 계산결과를 잠시 유지.
        self.params = self.embed.params  # params에는 매개변수를 저장.
        self.grads = self.embed.grads  # grads에는 기울기를 저장.
        self.cache = None # 순전파시의 계산결과를 잠시 유지.
        
    def forward(self, h, idx):    # 순전파를 담당. h(은닉층뉴런), 단어ID의 넘파이배열(idx)
        target_W = self.embed.forward(idx) # 배열로 받는 이유는 데이터를 한꺼번에 처리하는 미니배치 처리를 가정했기 때문.
        out = np.sum(target_W * h, axis=1)   # forward 메소드에서는 우선 Embedding 계층의 forward(idx)를 호출한 다음 내적을 계산한다.
        
        self.cache = (h, target_W)
        return out
    
    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)
        
        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh


class UnigramSampler: # 이 메소드는 target 인수로 지정한 단어를 정답으로 해석하고 그 외의 단어 ID를 샘플링한다.
    def __init__(self, corpus, power, sample_size): # 단어ID목록(corpus), 확률분포에 제곱할 값 0.75 (power), 오답 샘플링 횟수(sample_size)
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None
        
        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1
            
        vocab_size = len(counts)
        self.vocab_size = vocab_size
        
        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]
            
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)
        
    def get_negative_sample(self, target):
        batch_size = target.shape[0]
        
        if not GPU:  # == CPU
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)
            
            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0  # target이 뽑히지 않게 하기 위함
                p /= p.sum()  # 다시 정규화 해줌
                negative_sample[i, :] = np.random.choice(self.vocab_size,
                                                         size=self.sample_size,
                                                         replace=False, p=p)
                
        else:
            # GPU(cupy)로 계산할 때는 속도를 우선한다.
            # 부정적 예에 타깃이 포함될 수 있다.
            negative_sample = np.random.choice(self.vocab_size, 
                                               size=(batch_size, self.sample_size), 
                                               replace=True, p=self.word_p)
            
        return negative_sample


class NegativeSamplingLoss: # 초기화 메소드
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size 
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)] # 원하는 계층을 리스트로 보관한다.
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]  # 원하는 계층을 리스트로 보관한다.
        
        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads
            
    def forward(self, h, target): # 은닉층 뉴런(h), 긍정적 예의 타깃을 뜻하는 target
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target) # 부정적 예를 샘플링하여 nega_sample에 저장.
        
        # 긍정적 예 순전파
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)
        
        # 부정적 예 순전파
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]  # embed_dot에 해당하는 타겟이라는 의미인 듯
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label) # 정답/오답 예에 대해 손실을 더함.
            
        return loss
    
    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)  # 여러개의 기울기값을 더해준다.
        
        return dh