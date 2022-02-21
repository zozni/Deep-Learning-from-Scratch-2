# CBOW모델의 추론 처리 구현 (추론 처리란 점수를 구하는 처리를 말한다.)
# chap03/cbow_predict.py
import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul

# 샘플 맥락 데이터
c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

# 가중치 초기화
W_in = np.random.randn(7, 3) # 필요한 가중치들 초기화
W_out = np.random.randn(3, 7)

# 계층 생성
in_layer0 = MatMul(W_in) # 입력층을 처리하는 MatMul 계층을 맥락 수만큼 생성 (여기서는 2개)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out) # 출력층의 MatMul 계층은 1개만 생성.
# 이때 입력층 측의 MatMul 계층은 가중치 W_in을 공유한다.

# 순전파
h0 = in_layer0.forward(c0)  # 입력층 측의 MatMul 계층들의 forward() 메소드를 호출해 중간 데이터를 계산하고, 출력층 측의 MatMul 계층을 통과시켜 
h1 = in_layer1.forward(c1)  # 각 단어의 점수를 구한다.
h = 0.5 * (h0 + h1)  # average
s = out_layer.forward(h)  # score

print(s)