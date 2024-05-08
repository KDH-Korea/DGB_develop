#기본
import pandas as pd
import numpy as np
#임베딩
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec   
#PCA
from sklearn.decomposition import PCA
#클러스터링
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#시각화
import matplotlib.pyplot as plt
import seaborn as sns
#tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv(r"C:\Users\ehdgo\OneDrive - 계명대학교\대학\동아리\2023 1,2학기 학술\클롤링\전처리\아프니까사장이다 형태소 분석 완료_인터넷뱅킹.csv")
