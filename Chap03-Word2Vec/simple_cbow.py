# chap03/simple_cbow.py
import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss


class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):   # 인수로 어휘수와 은닉층의 뉴런수를 받는다.
        V, H = vocab_size, hidden_size
        
        # 가중치 초기화. 가중치를 2개 생성. 두 가중치는 각각 작은 무작위 값으로 초기화된다.
        W_in = 0.01 * np.random.randn(V, H).astype('f') # 넘파이 배열의 데이터 타입을 32비트 부동소수점 수로 초기화한다는 뜻
        W_out = 0.01 * np.random.randn(H, V).astype('f')
        
        # 레이어 생성
        self.in_layer0 = MatMul(W_in) # 입력 측 MatMul 계층들은 모두 같은 가중치를 이용하도록 초기화한다.
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()   # 여기서 입력 측의 맥락을 처리하는 MatMul 계층은 맥락에서 사용하는 단어의 수 (윈도우크기)만큼 만들어야 한다.
        
        # 모든 가중치와 기울기를 리스트에 모은다.
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], [] # 이 신경망에서 사용되는 매개변수와 기울기를 인스턴스 변수인 params와 grads 리스트에 각각 모아둔다.
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
            
        # 인스턴스 변수에 단어의 분산 표현을 저장한다.
        self.word_vecs1 = W_in
        self.word_vecs2 = W_out.T
        
    def forward(self, contexts, target):  # 인수 contexts는 3차원 넘파이 배열이라고 가정.
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss
    
    def backward(self, dout=1): # 역전파
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None

# 이미 각 매개변수의 기울기를 인스턴스 변수 grads에 모아뒀다. (코드 24번줄)
# 따라서 forward() 메소드를 호출한 다음 backward() 메소드를 실행하는 것만으로 grads 리스트의 기울기가 갱신된다.