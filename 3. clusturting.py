# 모듈 임포트
import pandas as pd
import numpy as np
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec   
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import GPT2Model, PreTrainedTokenizerFast, GPT2LMHeadModel,GPT2Tokenizer
import torch
import faiss

# 데이터 불러오기
df = pd.read_csv(r"C:\Users\ehdgo\OneDrive - 계명대학교\대학\동아리\23~24년 2차 DGB 대회\본선 개발\형태소분석_20240507_204434.csv")
desired_columns = ['keyword', 'date', 'text', 'tok_text', 'final_label']
df = df[desired_columns]

# Doc2Vec
# TaggedDocument 생성
tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(df['tok_text'])]  # 'tokenized_review'를 'tok_text'로 수정
# 모델 초기화 및 훈련
model = Doc2Vec(vector_size=50, min_count=2, epochs=40)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
# 벡터 추출 및 병합
df['vector'] = [model.dv[str(i)].tolist() for i in range(len(tagged_data))]

# PCA
# vector array 생성 및 PCA 수행
vector_array = np.array(df['vector'].tolist())
pca = PCA(n_components=2)
df[['PC1', 'PC2']] = pca.fit_transform(vector_array)

# KMeans
X = df[['PC1', 'PC2']]

# 엘보우 기법을 사용하여 최적의 k 찾기
sse = []
silhouette_coefficients = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)
    score = silhouette_score(X, kmeans.labels_)
    silhouette_coefficients.append(score)
# 엘보우 기법 시각화
plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), sse, marker='o')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()
# 실루엣 계수 시각화
plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), silhouette_coefficients, marker='o')
plt.title('Silhouette Coefficient')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()
# 실루엣 계수가 높은 상위 3개의 k 값을 추출하고 가장 작은 k 선택
top_k_indices = sorted(range(len(silhouette_coefficients)), key=lambda i: silhouette_coefficients[i], reverse=True)[:3]
optimal_k = min([k for k in range(3, 11) if k in top_k_indices])
print(f"Optimal number of clusters by top 3 highest silhouette scores: {optimal_k}")
# 최적의 k로 K-means 클러스터링 수행
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)
# 클러스터링 결과 시각화
plt.figure(figsize=(10, 6))
plt.scatter(df['PC1'], df['PC2'], c=df['Cluster'], cmap='viridis', marker='o', edgecolor='black')
plt.title('Clustering on PCA results')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(label='Cluster')
plt.show()

# TF-IDF
# 각 클러스터별로 TF-IDF 벡터라이저 적용
def cluster_top_tfidf_features(cluster_data, n_terms):
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # 최대 1000개의 단어를 고려
    tfidf_matrix = tfidf_vectorizer.fit_transform(cluster_data)
    feature_array = np.array(tfidf_vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    top_n = feature_array[tfidf_sorting][:n_terms]
    return top_n
# 각 클러스터별 상위 TF-IDF 단어 출력
print("Top terms per cluster:")
for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]['tok_text']
    top_terms = cluster_top_tfidf_features(cluster_data, 20)
    print(f"Cluster {i}: {', '.join(top_terms)}")
    
# gpt-2를 사용해 각 클러스터별 중요 TF-IDF단어 선정
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
  bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')
model = GPT2Model.from_pretrained("skt/kogpt2-base-v2")
# 주제 입력받기
user_topic = input("Enter a topic you are interested in: ")
def filter_important_words(texts, model, tokenizer, user_topic):
    # 각 텍스트에 대한 중요 단어 평가
    important_words = []
    for text in texts:
        input_ids = tokenizer.encode(user_topic + " " + text, return_tensors='pt')
        outputs = model.generate(input_ids, max_length=50)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        important_words.append(result.split())  # 결과에서 단어 분리
    return important_words
# 클러스터별 중요 단어 필터링
for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]['tok_text']
    top_terms = cluster_top_tfidf_features(cluster_data, 20)
    important_terms = filter_important_words(top_terms, model, tokenizer, user_topic)
    print(f"Cluster {i} important words: {', '.join(important_terms)}")
  
# 중요 단어를 포함한 문장 원문 반출
# 벡터화를 위한 준비
vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer.fit_transform(df['text']).toarray()
# FAISS 인덱스 생성
dimension = X_tfidf.shape[1]  # 특성의 수
index = faiss.IndexFlatL2(dimension)  # L2 거리를 사용하는 평면 인덱스
index.add(X_tfidf.astype(np.float32))  # FAISS는 float32를 사용
def search_documents(query, index, df, top_k=5):
    query_vec = vectorizer.transform([query]).toarray()
    distances, indices = index.search(query_vec.astype(np.float32), top_k)
    return df.iloc[indices[0]]
# 중요 단어 포함 문장 검색
for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]['tok_text']
    top_terms = cluster_top_tfidf_features(cluster_data, 20)
    important_terms = filter_important_words(top_terms, model, tokenizer, user_topic)
    for term in important_terms:
        print(f"Search results for {term} in Cluster {i}:")
        results = search_documents(term, index, df)
        print(results['text'])
