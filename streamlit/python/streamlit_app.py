import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# 데이터셋 로드
df = pd.read_csv("titanic.csv")

# 대시보드 제목
st.title("Titanic 생존률 대시보드")

# 생존자와 사망자 수 시각화
survived = df["Survived"].value_counts()

# 레이아웃 설정
col1, col2 = st.columns(2)

# col1에 생존자와 사망자 수 차트 추가
with col1:
    st.subheader("생존자와 사망자 수")
    st.bar_chart(survived)

# 성별 생존율 시각화
gender_survival = df.groupby("Sex")["Survived"].mean()

# col2에 성별 생존율 차트 추가
with col2:
    st.subheader("성별 생존율")
    st.bar_chart(gender_survival)

# 객실 등급에 따른 생존율 시각화
class_survival = df.groupby("Pclass")["Survived"].mean()

# 레이아웃 설정
col3, col4 = st.columns(2)

# col3에 객실 등급에 따른 생존율 차트 추가
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

# col4에 나이별 생존율 차트 추가
with col4:
    st.subheader("나이별 생존율")
    st.bar_chart(age_survival)

# 승선 항구에 따른 생존율 시각화
embark_survival = df.groupby("Embarked")["Survived"].mean()

# 레이아웃 설정
col5, col6 = st.columns(2)

# col5에 승선 항구에 따른 생존율 차트 추가
with col5:
    st.subheader("승선 항구에 따른 생존율")
    st.bar_chart(embark_survival)

# Fare 분포 시각화
st.subheader("Fare 분포")
sns.histplot(data=df, x="Fare", kde=True)
st.pyplot()

# 대시보드 실행
if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)
