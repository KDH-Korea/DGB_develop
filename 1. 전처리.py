# raw데이터 기본 전처리 + 형태소 분석
# 다음 진행 파일은 문장 분리 및 준지도 학습
# 모듈 임포트
import pandas as pd
import numpy as np
import re
from collections import Counter
from konlpy.tag import Okt, Komoran, Mecab, Hannanum, Kkma
from datetime import datetime

df = pd.read_csv(r'C:\Users\ehdgo\OneDrive - 계명대학교\대학\동아리\2023 1,2학기 학술\클롤링\아프니까 사장이다\아프니까 사장이다_하나은행 부터.csv')
# 필요없는 열 제거
df = df[['keyword', 'tit', 'body', 'comment', 'date']]

# 중복값 지우기
df = df.drop_duplicates(subset=['tit', 'body'])

# 결측치 채우기
df['body'] = df['body'].fillna('')

# tit, body, comment 합치기, 컬럼 삭제
df.loc[:,'text'] = df['tit'] + ' ' + df['body'] + ' ' + df['comment']
df=df.drop(['tit','body','comment'],axis=1)

# 불용어 처리 함수
def preprocess_text(text):
    if isinstance(text, str):
        # HTML/XML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        # 개행 문자를 공백으로 대체
        text = re.sub(r'\n', ' ', text)
        # 연속된 공백 문자를 하나의 공백으로 대체
        text = re.sub(r'\s+', ' ', text)
        # '갤'을 '개월'로 대체
        text = re.sub(r'[0-9]+갤', '개월', text)
        # 한글과 온점을 제외한 모든 문자 제거
        text = re.sub(r'[^가-힣.]', ' ', text)
        # '.' 다음에 나오는 공백을 없애기
        text = re.sub(r'\.\s+', '.', text)
        # 연속된 '.'을 하나의 '.'으로 대체
        text = re.sub(r'\.{2,}', '.', text)
        # '.' 다음에 나오는 공백을 없애기
        text = re.sub(r'\.\s+', '.', text)
        # 연속된 공백 문자를 하나의 공백으로 대체(한번 더)
        text = re.sub(r'\s+', ' ', text)   
    return text
# 불용어 처리
df.loc[:, 'text'] = df['text'].apply(preprocess_text)

# 특정 단어가 들어간 행 삭제(ex. 광고, 홍보글 등등)
words_to_delete = ['(협찬)', '체험단', '무료체험', '체험후기', '예약링크']
pattern = '|'.join(words_to_delete)
delete = df['text'].str.contains(pattern)
df.drop(df.index[delete], inplace = True)
df = df.reset_index(drop=True)

# 반복 문장 삭제
df.text = df.text.str.replace("긴글+ 사진폭탄 주의","")
df.text = df.text.str.replace("질문 있습니당.","")
df.text = df.text.str.replace("★","")
df.text = df.text.str.replace("댓글 없음","")
df.text = df.text.str.replace("\*주사기, 수액, 처방약 등 의약품 등록 불가* '채팅 거래 불가","")
df.text = df.text.str.replace("채팅 거래 불가 / 글 등록 후 채팅으로 연락오는 대부분이 사기며 모든 거래는 메모인증 등 실제 제품 확인해야 합니다. 채팅거래는 '사기치세요'와 같은 뜻입니다.","")
df.text = df.text.str.replace("\* 미준수 글은 활동정지/게시글 삭제","")
df.text = df.text.str.replace("안녕하세요","")
df.text = df.text.str.replace("안녕하세요!","")
df.text = df.text.str.replace("안녕하세요.","")
df.text = df.text.str.replace("감사합니다.","")
df.text = df.text.str.replace("챗드릴게요","")
df.text = df.text.str.replace("쪽지주세요","")
df.text = df.text.str.replace("쪽지드릴게요","")
df.text = df.text.str.replace("쪽지 드렸어요","")
df.text = df.text.str.replace("\* 규정 지킬 생각이 없으면 다른 카페에서 거래해 주세요.","")
df.text = df.text.str.replace("1\)메모인증은 수기작성(종이에 직접 아이디/닉네임/판매날짜/제품명 제품수량 제품용량)기입","")
df.text = df.text.str.replace("2\)작성 후 직접 촬영한 제품 사진과 함께 보이도록 사진 두 장 필수(제품 전체사진+메모인증 보이게 1장, 메모 잘 보이는 사진 1장) (상세내용은 배너클릭) ","")
df.text = df.text.str.replace("3\)판매할 제품은 모두 사진찍어 수량/용량 정확히 기입. ","")
df.text = df.text.str.replace("4\)벼룩 판매 가격규정 미준수 거래완료 판매자 아이디는 영구활동정지되니 반드시 공지 확인후 글등록","")
df.text = df.text.str.replace("완료되었습니다~.","")
df.text = df.text.str.replace("나눔후기\)","")
df.text = df.text.str.replace("\[체험후기\]","")
# 띄어쓰기 한번 더 처리
def re_spacing(text):
    result = text.strip()
    while "  " in result:
        result = result.replace("  ", " ")
    return result
df['text'] = df['text'].apply(lambda x : re_spacing(x))

# 형태소 분석 함수 지정
from konlpy.tag import Okt, Komoran, Mecab, Hannanum, Kkma
def get_tokenizer(tokenizer_name):
    if tokenizer_name == "komoran":
        tokenizer = Komoran()
    elif tokenizer_name == "okt":
        tokenizer = Okt()
    elif tokenizer_name == "mecab":
        tokenizer = Komoran()
    elif tokenizer_name == "Mecab":
        tokenizer = Komoran()
    elif tokenizer_name == "hannanum":
        tokenizer = Hannanum()
    elif tokenizer_name == "kkma":
        tokenizer = Kkma()
    else:
        tokenizer = Okt()
    return tokenizer
# 형태소 분석 진행
tokenizer = get_tokenizer(Mecab)
def pos_tagging_and_filter(text):
    # Mecab을 사용하여 텍스트의 품사 태깅을 수행합니다.
    pos_tokens = tokenizer.pos(text)
    # 'Josa' 품사를 가진 토큰(조사)을 필터링합니다.
    filtered_tokens = [token for token, pos in pos_tokens if pos != 'Josa']
    # 추가적인 단어들을 필터링합니다.
    filtered_tokens = [token for token in filtered_tokens if token not in ['그리고', '그러나', '그런데', '그래서', '또는', '혹은']]
    return filtered_tokens
df['tok_text'] = df['text'].apply(pos_tagging_and_filter)
#형태소 분석파일 저장 - 형태소 분석 완료된 시간으로 파일명
now = datetime.now()  
formatted_time = now.strftime("%Y%m%d_%H%M%S") 
filename = f"형태소분석_{formatted_time}.csv" 
df.to_csv(filename, index=False)