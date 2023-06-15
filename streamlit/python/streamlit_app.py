import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
#function to read dataset from csv file 
@st.cache
def read_data():
    df = pd.read_csv("train.csv")
    return df

#taking input from function
df = read_data()
# 데이터셋 로드

# 대시보드 제목
st.title("Titanic 생존률 대시보드")

# 사이드바에서 columns 갯수 선택
num_columns = st.sidebar.selectbox("Columns 갯수 선택", [1, 2, 3])

# 컬럼 수에 따른 레이아웃 설정
if num_columns == 1:
    col1 = st.container()
elif num_columns == 2:
    col1, col2 = st.columns(2)
else:
    col1, col2, col3 = st.columns(3)

# 생존자와 사망자 수 시각화
survived = df["Survived"].value_counts()

# col1에 생존자와 사망자 수 차트 추가
with col1:
    st.subheader("생존자와 사망자 수")
    st.bar_chart(survived)

# 성별 생존율 시각화
gender_survival = df.groupby("Sex")["Survived"].mean()

# col2에 성별 생존율 차트 추가
if num_columns >= 2:
    with col2:
        st.subheader("성별 생존율")
        st.bar_chart(gender_survival)

# 객실 등급에 따른 생존율 시각화
class_survival = df.groupby("Pclass")["Survived"].mean()

# col3에 객실 등급에 따른 생존율 차트 추가
if num_columns >= 3:
    with col3:
        st.subheader("객실 등급에 따른 생존율")
        st.bar_chart(class_survival)

# 나이별 생존율 시각화
# 결측치 처리
df["Age"].fillna(df["Age"].mean(), inplace=True)
# 나이 구간 생성
age_bins = [0, 18, 30, 50, 80]
age_labels = ["0-18", "18-30", "30-50", "50+"]
df["AgeGroup"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels)
age_survival = df.groupby("AgeGroup")["Survived"].mean()

# 나이별 생존율 차트 추가
st.subheader("나이별 생존율")
st.bar_chart(age_survival)

# 승선 항구에 따른 생존율 시각화
embark_survival = df.groupby("Embarked")["Survived"].mean()

# 승선 항구에 따른 생존율 차트 추가
st.subheader("승선 항구에 따른 생존율")
st.bar_chart(embark_survival)

# Fare 분포 시각화
st.subheader("Fare 분포")
fig, ax = plt.subplots()
sns.histplot(data=df, x="Fare", kde=True, ax=ax)
st.pyplot(fig)
