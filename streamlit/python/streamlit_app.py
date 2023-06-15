import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 타이타닉 데이터를 불러옵니다.
data = pd.read_csv('train.csv')

# 타이틀을 설정합니다.
st.title('타이타닉 데이터 시각화 대시보드')

# 데이터프레임을 출력합니다.
st.subheader('데이터 개요')
st.write(data)

# 생존자와 사망자의 수를 계산합니다.
survived = data['Survived'].value_counts()
survived_df = pd.DataFrame({'Survived': survived.index, 'Passenger Count': survived.values})

# 막대 그래프로 생존자와 사망자를 시각화합니다.
st.subheader('생존자 vs 사망자')
st.bar_chart(survived_df['Passenger Count'])

# 객실 등급별 생존율을 계산합니다.
pclass_survived = data.groupby('Pclass')['Survived'].mean().reset_index()
pclass_survived.columns = ['Pclass', 'Survival Rate']

# 막대 그래프로 객실 등급별 생존율을 시각화합니다.
st.subheader('객실 등급별 생존율')
st.bar_chart(pclass_survived['Survival Rate'])

# 성별에 따른 생존자와 사망자의 수를 계산합니다.
sex_survived = data.groupby('Sex')['Survived'].value_counts().unstack().reset_index()
sex_survived.columns = ['Sex', 'Did Not Survive', 'Survived']
sex_survived = sex_survived.melt(id_vars='Sex', var_name='Survival Status', value_name='Passenger Count')

# 막대 그래프로 성별에 따른 생존자와 사망자를 시각화합니다.
st.subheader('성별에 따른 생존자 vs 사망자')
sns.barplot(x='Sex', y='Passenger Count', hue='Survival Status', data=sex_survived)
plt.xlabel('Sex')
plt.ylabel('Passenger Count')
plt.legend(title='Survival Status')
st.pyplot()

# 나이 분포를 시각화합니다.
st.subheader('나이 분포')
st.hist(data['Age'].dropna(), bins=20, edgecolor='k')
plt.xlabel('Age')
plt.ylabel('Count')
st.pyplot()
