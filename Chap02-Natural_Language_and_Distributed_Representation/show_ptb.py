import sys
sys.path.append('..')
from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data('train')  # ptb.load_data()는 데이터를 읽어들임.
                                                         # 인수로는 train, test, valid 중 하나를 지정할 수 있다.

# corpus는 단어 ID 목록.
# id_to_word는 단어 ID에서 단어로 변환하는 딕셔너리.
# word_to_id는 단어에서 단어 ID로 변환하는 딕셔너리.

print('말뭉치 크기: ', len(corpus))  
print('corpus[:30]:', corpus[:30])
print()
print('id_to_word[0]: ', id_to_word[0])
print('id_to_word[1]: ', id_to_word[1])
print('id_to_word[2]: ', id_to_word[2])
print()
print("word_to_id['car']: ", word_to_id['car'])
print("word_to_id['happy']: ", word_to_id['happy'])
print("word_to_id['lexus']: ", word_to_id['lexus'])
