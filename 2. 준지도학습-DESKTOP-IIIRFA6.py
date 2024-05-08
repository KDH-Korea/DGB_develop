# 모듈 임포트
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from transformers import GPT2Model, PreTrainedTokenizerFast, GPT2LMHeadModel,GPT2Tokenizer
import faiss
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
from keras.layers import GRU

# 데이터 불러오기
df = pd.read_csv(r'C:\Users\ehdgo\OneDrive - 계명대학교\대학\동아리\23~24년 2차 DGB 대회\본선 개발\형태소분석_20240507_204434.csv')
# 데이터 3% 추출
labeling_df = pd.DataFrame(df['text'].sample(int(0.03 * len(df))))

# 토크나이저 및 모델 지정
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')
model = GPT2Model.from_pretrained("skt/kogpt2-base-v2")
headmodel = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")  # 사용자가 정의한 모델 경로로 변경해야 합니다.
MAX_TOKENS = 1024

# 임베딩 추출 함수
def get_embeddings(input_ids):
    input_tensor = torch.tensor([input_ids])
    with torch.no_grad():
        outputs = model(input_tensor)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # 평균 풀링으로 임베딩 추출

# GPT-2로 은행 관련 여부 판단 함수
def is_bank_related_gpt2(text):
    prompt = f"당신은 사용자가 말한 주제에 적합한지 판단하는 라벨러입니다. 이 텍스트는 은행,금융등에 관련된 내용과 관련이 있습니까?예, 아니요로 대답하시오. '{text}'"
    inputs = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=950)
    # 예외 처리를 추가하여 문제 발생 시 원인을 파악하고 넘김
    try:
        outputs = headmodel.generate(inputs, max_length=inputs.shape[1] + 15, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return "있습니다" in response or "예" in response
    except IndexError as e:
        print(f"Error: {e}, Input Text: {text}")
        return False

# 텍스트 데이터를 FAISS에 저장
new_data = []
for text in labeling_df['text']:
    input_ids = tokenizer.encode(text, truncation=True, max_length=MAX_TOKENS)
    new_data.append(get_embeddings(input_ids))

# FAISS 인덱스 생성 및 임베딩 저장
embeddings = np.array(new_data)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# FAISS 인덱스에서 데이터 추출 및 GPT-2로 라벨링
labeled_data = []
for i in range(index.ntotal):
    original_text = labeling_df.iloc[i]['text']
    embedding = index.reconstruct(i)
    # 은행 관련 여부를 판단하고 라벨 적용
    label = 1 if is_bank_related_gpt2(original_text) else 0
    labeled_data.append((original_text, label))
# 라벨링 값 데이터 생성   
df_labeled = pd.DataFrame(labeled_data, columns=['original_text','Label'])
# 라벨링 값 포함한 데이터 생성
df_merged = pd.merge(df, df_labeled, left_on='text', right_on='original_text', how='left')
df_merged = df_merged.drop(columns=['original_text'])
print(df_merged)

# LSTM을 통한 나머지 값 라벨링
# 데이터 준비
texts = df_merged['tok_text'].apply(eval).astype(str)
labels = df_merged['Label']
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_seq_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_seq_length)
labeled_indices = labels.notna()
unlabeled_indices = ~labeled_indices
X_labeled = padded_sequences[labeled_indices]
y_labeled = labels[labeled_indices].astype(int)
X_unlabeled = padded_sequences[unlabeled_indices]
# LSTM모델 설정
model_LSTM = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_seq_length),
    LSTM(128),
    Dense(1, activation='sigmoid')
])
model_LSTM.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# LSTM 훈련
model_LSTM.fit(X_labeled, y_labeled, epochs=10, batch_size=32, validation_split=0.2)
# 의사 레이블 생성
LSTM_pseudo_labels = (model_LSTM.predict(X_unlabeled) > 0.5).astype(int)
# 준지도 학습
X_combined = np.vstack((X_labeled, X_unlabeled))
y_combined = np.concatenate((y_labeled, LSTM_pseudo_labels.flatten()))
model_LSTM.fit(X_combined, y_combined, epochs=10, batch_size=32, validation_split=0.2)
# 최종 예측 수행 및 데이터에 추가
LSTM_pseudo_labels = (model_LSTM.predict(X_unlabeled) > 0.5).astype(int)
df_merged.loc[unlabeled_indices, 'LSTM_Label'] = LSTM_pseudo_labels.flatten()

# RNN을 통한 나머지 값 라벨링
# 데이터 준비는 위와 동일
# RNN 모델 설정
model_RNN = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_seq_length),
    SimpleRNN(128),
    Dense(1, activation='sigmoid')
])
model_RNN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# RNN 모델 훈련
model_RNN.fit(X_labeled, y_labeled, epochs=10, batch_size=32, validation_split=0.2)
# 의사 레이블 생성
RNN_pseudo_labels = (model_RNN.predict(X_unlabeled) > 0.5).astype(int)
# 준지도 학습
X_combined = np.vstack((X_labeled, X_unlabeled))
y_combined = np.concatenate((y_labeled, RNN_pseudo_labels.flatten()))
model_RNN.fit(X_combined, y_combined, epochs=10, batch_size=32, validation_split=0.2)
# 최종 예측 수행 및 데이터에 추가
RNN_pseudo_labels = (model_RNN.predict(X_unlabeled) > 0.5).astype(int)
df_merged.loc[unlabeled_indices, 'RNN_Label'] = RNN_pseudo_labels.flatten()

# GRU를 통해 RNN, LSTM의 라벨링 값이 다른 녀석들에 대한 재계산
# LSTM과 RNN의 예측 결과가 다른 데이터 선택
different_indices = [i for i, (lstm_label, rnn_label) in enumerate(zip(LSTM_pseudo_labels.flatten(), RNN_pseudo_labels.flatten())) if lstm_label != rnn_label]
# GRU 모델을 사용해야 하는지 확인
if different_indices:
    # GRU 모델을 사용해야 할 경우에만 진행
    X_different = X_unlabeled[different_indices]
    # GRU 모델 설정 및 훈련
    model_GRU = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_seq_length),
        GRU(128),
        Dense(1, activation='sigmoid')
    ])
    model_GRU.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_GRU.fit(X_labeled, y_labeled, epochs=10, batch_size=32, validation_split=0.2)
    # GRU 모델을 사용한 예측
    gru_pseudo_labels = (model_GRU.predict(X_different) > 0.5).astype(int)
    # 예측 결과를 데이터에 추가
    df_merged.loc[unlabeled_indices[different_indices], 'GRU_Label'] = gru_pseudo_labels.flatten()
else:
    print("LSTM과 RNN 모델의 예측 결과가 모두 동일하여 GRU 모델을 사용하지 않습니다.")

print(df_merged)