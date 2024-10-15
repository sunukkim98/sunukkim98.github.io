---
layout: post
title: Overview of the Data Mining Process
description: |
  - Introduction
  - Core Ideas in Data Mining
  - The Steps in Data Mining
  - Preliminary Steps
  - Predictive Power and Overfitting
  - Building a Predictive Model
categories: Data_Mining
sitemap: false
hide_last_modified: true
---
# Overview of the Data Mining Process

## Introduction
- **<span style='background-color: #fff5b1'>데이터 모델링 과정</span>**

<p align="center">
<img src="/assets/img/blog/data-mining/OverviewOfTheDataMiningProcess/DataModelingProcess.png">
</p>

위 그림은 Data Modeling Process를 나타낸 것이다.
- 목적 정의
- 데이터 획득
- 데이터 탐색 및 정제
- DM 작업 결정
- DM 방법 선택
- 방법 적용, 최종 모델 선택
- 성능 평가
- 배포

Data Modeling Process는 이와 같이 이루어진다.

- Business Analytics 핵심 요소
    - 예측 분석(Predictive Analytics)
        - 분류(Classification) / 예측(Prediction)
    - 데이터 마이닝에서 많이 쓰이는 데이터베이스 기법
        - OLAP(Online Analytical Processing)과 SQL(Structured Query Language)
        - 예) 신용카드 고객 중에서 특정 지역에 살고, 연간지출액이 2만 달러를 넘고, 자기 주택을 소유하고 있고, 적어도 95%는 결제일을 맞춘 고객을 찾는 문제

## Core Ideas in Data Mining
- **<span style='background-color: #fff5b1'>Classification</span>**
    - 데이터 분석의 가장 많이 다루는 문제
    - 데이터를 집단으로 구분하기 위함
    - "응답" or "응답하지 않음" / "정상" or "사기" / "정상" or "고장" / "회복" or "진행중" or "사망"

<p align="center">
<img src="/assets/img/blog/data-mining/OverviewOfTheDataMiningProcess/Classification.png">
</p>

- **<span style='background-color: #fff5b1'>Prediction</span>**
    - 예측하고자 하는 변수가 숫자로 표현된 연속형 변수일 경우 예측 문제로 분류 <br>
     &#8251; 예측하고자 하는 변수가 범주형인 경우 분류 문제

<p align="center">
<img src="/assets/img/blog/data-mining/OverviewOfTheDataMiningProcess/Prediction.png">
</p>

- **Association Rules and Recommendation Systems**
    - 대량의 고객거래 데이터베이스는 구매항목들 간의 연관성, 즉 어떤 항목이 어떤 항목과 관련 되는지에 대한 분석
    - Collaborative Filtering: 개개인의 포괄적인 과거 구매정보와 다른 사람들의 구매정보를 이용하여 개개인의 구매성향을 예측하는 추천 시스템

<p align="center">
<img src="/assets/img/blog/data-mining/OverviewOfTheDataMiningProcess/Association Rules and Recommendation Systems.png">
</p>

- **Cluster Analysis(as Data reduction method)**
    - 비지도학습으로 데이터를 동질적인 군집들로 세분화
    - 측정값들로 구성된 레코드로부터 측정값들이 유사한 레코드의 모임 또는 군집을 형성하기 위해 사용

<p align="center">
<img src="/assets/img/blog/data-mining/OverviewOfTheDataMiningProcess/Cluster Analysis.png">
</p>

- **Dimension Reduction**
    - 변수의 개수를 줄이는 과정
    - 지도 학습 전에 수행되며, 예측 성능을 향상시키고 해석의 용이성 증대 목적

- **Data Exploration**
    - 데이터의 전반에 관한 이해와 이상치 탐지 목적
    - 수치적 혹은 시각적으로 데이터를 요약하는 방법론 사용

- **Data Visualization**
    - 수치형 변수: 히스토그램(histogram)과 상자그림(boxplot)을 이용하여, 변수값의 분포를 파악하고 극단치(outliers)를 찾음
    - 범주형 변수: 차트(charts)와 원형 차트(pie charts)를 이용. 변수 간의 가능한 관계들, 관계유형 그리고 극단치를 찾기 위해 한 쌍의 수치형 변수에 대한 산점도(scatter plots)을 조사

<p align="center">
<img src="/assets/img/blog/data-mining/OverviewOfTheDataMiningProcess/Histogram_and_Boxplot.png">
</p>

- **Supervised Learning**
    - 분류 또는 예측하고자 하는 변수가 존재할 경우 이를 labeled data(결과값)로 놓고 예측변수와의 관계를 통해 모델링
    - 학습 데이터를 이용하여 예측변수(종속변수)와 결과변수 간의 관계를 학습, 훈련함
    - 학습 데이터로부터 모델이 구축되면 결과를 알고 있는 검증 데이터를 사용하여 모델의 성능을 평가하고 다른 방법론과 비교
    - 다수의 모델을 비교하고자 할 경우, 평가 데이터를 통해 성능을 비교, 평가함
    - 최종적으로 검증이 끝난 모델은 예측변수의 값을 모르는 미래의 데이터 예측에 사용
    - 예) 단순 선형 회귀분석, 판별 분석, 역전파 신경망

- **Unsupervised Learning**
    - 예측 또는 분류를 위해 필요한 출력변수가 없는 경우에 사용되는 알고리즘
    - 예) 연관 규칙, 차원 축소 기법, 군집 분석 등

## The Steps in Data Mining

- **<span style='background-color: #dcffe4'>Step1. Develop an understanding of the putpose of the data mining project</span>**
    - 데이터 마이닝 프로젝트의 <span style='background-color: #fff5b1'>목적</span>을 정확히 설정
    > 목적: 수요가 필요한 쪽에 맞추어야 함
- **<span style='background-color: #dcffe4'>Step2. Obtain the dataset to be used in the analytics</span>**
    - 분석에 필요한 데이터 흭득
    - 대량의 데이터베이스에서 무작위 표본 추출 또는 서로 다른 데이터베이스에서 별도로 추출하여 통합
- **<span style='background-color: #dcffe4'>Step3. Explore, clean, and preprocess the data</span>**
    - 데이터의 탐색, 정제, 전처리
    - 데이터가 타당한 조건에 있는지를 검증: 결측치, 극단치 처리, 변수 간의 관계를 산점도 등으로 검토, 변수에 대한 정의, 측정단위, 측정 기간 등에 대한 일관성 체크
    > Data Pipeline: 데이터를 수집하여 전처리하고, 모델링하고, 결과를 시각화하는 과정
- **<span style='background-color: #dcffe4'>Step4. Reduce the data dimension, if necessary</span>**
    - 필요시 데이터 차원 축소
    - 불필요한 변수의 제거, 변수 값의 변환, 새로운 변수의 생성 등
- **<span style='background-color: #dcffe4'>Step5. Determine the data mining task(classification, prediction, clustering, etc.)</span>**
    - 데이터 마이닝 문제 결정(분류, 예측, 군집 등)
- **<span style='background-color: #dcffe4'>Step6. Partition the data(for supervised learning)</span>**
    - 데이터 분할(지도학습의 경우): 학습(training) / 검증(validation) / 평가(test)
- **<span style='background-color: #dcffe4'>Step7. Choose the data mining techniques to be used(regression, neural nets, hierarchical clustering, and so on)</span>**
    - 사용할 데이터 마이닝 기법 선택(회귀분석, 인공신경망, 계층 군집 등)
- **<span style='background-color: #dcffe4'>Step8. Use algorithms to perform the task</span>**
    - 데이터 마이닝 프로세스의 여러 단계를 반복적으로 수행하여 가장 좋은 알고리즘 탐색
    - 변수의 조합 시도, 알고리즘의 셋팅값 변경
- **<span style='background-color: #dcffe4'>Step9. Interpret the results of the algorithms</span>**
    - 가장 효율적인 알고리즘을 찾아내고, 검증 데이터를 이용하여 구축된 알고리즘의 성능을 평가
- **<span style='background-color: #dcffe4'>Step10. Deploy the model</span>**
    - 구축된 모델을 운용시스템에 탑재하여 실제 의사결정에 적용하는 단계
- **cf. SEMMA, methodology by SAS**
    - 표본 추출(Sampling): Training / Validation / Test data set
    - 탐색(Exploration): 데이터에 포함된 변수들이 어떠한 분포를 하고 있으며, 변수들 간의 관계가 어떠한 것인지를 파악
    - 변환(Modification) 및 변수 선정: 분석 목적에 적합한 형태로 변환
    - 모델링(Modeling): 문제의 특성에 맞는 적절한 기법을 통해서 모형 개발
    - 평가(Assessment): 검증 데이터(validation data set)를 통해 여러 종류의 모형을 비교평가

<p align="center"> 
<img src="/assets/img/blog/data-mining/OverviewOfTheDataMiningProcess/SEMMA.png">
</p>

## Preliminary Steps
- **Predicting Home Values in the West Roxbury Neighborhood**
    - 보스턴시에서 공개한 보스턴 지역 부동산 평가 데이터
    - 2014년 보스턴 남서부 웨스트 록스베리 지역의 단독주택 정보 (14개 변수 / 5,802개 주택)
    - 결과값: Total Value(주택 가격)

<p align="center">
<img src="/assets/img/blog/data-mining/OverviewOfTheDataMiningProcess/PredictingHomeValuesintheWestRoxburyNeighborhood.png">
</p>

- **Sampling from a Database**
    - 대개, 일부 레코드만 가지고 algorithm을 수행하고자 함
    - Computing power의 한계로 레코드나 variable의 개수에서 limitation을 가짐
    - 수백, 수천 개 정도의 레코드로도 정확한 model 수립이 가능
- **Oversampling Rare Events in Classification Tasks**
    1. 관심 대상 데이터가 희귀할 경우, 즉 정상 데이터는 많으나 비정상 데이터는 부족할 경우 &rarr; 샘플링시 <span style="color:red">소수 클래스에 보다 큰 가중치</span>를 주어 클래스 관측치 수에 균형을 맞출 수 있음
    2. 각 클래스의 <span style="color:red">오분류에 가중치</span>를 주어 해결 가능
    - 모델을 수립하는 데에 아무 정보를 주지 않는 수많은 non-rare event를 포함하는 sampling
            - 이 관심 대상 event에 대해 overweight하게 됨
            - Rare event를 찾기 위한 비용도 발생
            - 응답하지 않은 사람을 응답자로 잘못 분류하는 비용과, 응답자를 찾아내는 비용 간의 균형 필요
- **Preprocessing and Cleaning the Data**
    - **(1) Type of Variables**
        - numerical or text(or character) 
        - continuous(연속형: 대개 주어진 범위 내의 실수), integer(오직 정수 값), or categorical(범주형: 일정 범위의 값을 하나로 범주로 가정)
        - categorical: numerical(1,2,3) or text(현금결제, 비현금결제, 파산)
            - unordered(순위정보를 갖지 않는 범주형): "nominal"(명목형 변수) - Asia, Europe, North America
            - ordered(순위정보를 갖는 범주형): "ordinal"(순위형 변수) - high, medium, low
        - 각 방법마다 적용가능 variable에 제한이 있을 수 있음(ex. $$Na\ddot{i}ve Bayes$$는 categorical)

<p align="center">
<img src="/assets/img/blog/data-mining/OverviewOfTheDataMiningProcess/type_of_variables.png" width="200" height="400">
</p>

- 
    - **(2) Handling Categorical Variables**
        - 범주형 변수가 순위정보를 갖고 있는 경우(연령 구간, 신용등급 등): 연속형 변수인 것처럼 변수를 있는 그대로 사용
        - 범주형 변수가 명목형인 경우: 이진 분류의 더미변수로 분할 사용
            - 학생: 예/아니오 &nbsp;&nbsp;&nbsp;&nbsp; 무직: 예/아니오

<p align="center">
<img src="/assets/img/blog/data-mining/OverviewOfTheDataMiningProcess/handling_categorical_variables.png" width="400" height="200">
</p>

- 
    - **(3) Variable Selection**
        - 더 많은 변수의 활용이 더 나은 결과를 보장하지 않음 &rarr; 꼭 필요한 변수의 사용이 바람직함
        - 더 많은 변수를 모델에 사용하는 경우 변수 간의 상관관계 파악의 복잡성이 증가
            - 예를 들어 종속변수 Y와 하나의 변수 X의 상관관계를 알기 위해 15개의 데이터로 충분할 수 있음
            - 만약 15개의 변수X를 사용한다면 이보다 훨씬 많은 데이터 필요
    
    - **(4) How Many Variables and How Much Data?**
        - 경험에 의한 법칙(rules of thumb): 모든 예측변수는 각각 10개의 레코드 필요
        - <span style='background-color: #fff5b1'>Delmaster and Hancock(2001, p. 68): 최소한 $$6 \times m \times p$$개의 레코드 필요($$m$$: 클래스의 수, $$p$$: 변수의 개수)</span>
        - 도메인 지식을 가지고 있는 사람으로부터의 정보는 변수의 포함 여부를 결정할 때 중요하며 모델의 정확도를 높이고 오차를 줄일 수 있음
    - **(5) Outliers**
        - 대부분의 데이터로부터 멀리 떨어진 값들은 극단치(Outliers)로 불리운다.
        - Rule of thumb: 평균으로부터 표준편차의 3배보다 더 멀리 떨어져 있는 값은 극단치에 해당한다
        - 잘못된 데이터의 경우: 도메인 지식이나 상식을 활용하여 제거(체온 50도 등)
        - 극단치가 적을 경우 이를 제거
        - 각 column 별로 sorting하여 outlier 탐색, 또는 max-min value 검토
    - **(6) Missing Values**
        - 결측값을 갖는 레코드의 수가 적다면 그 레코드는 제외하고 분석
        - 변수의 개수가 많다면 단순 삭제에 문제가 있음
        - 결측값을 데이터에서 차지하는 비율이 낮더라도 많은 레코드에 영향
            - ex. 30개 변수, 5%의 결측값 &rarr; 거의 80%의 데이터 삭제 (주어진 레코드가 결측값을 가지고 있지 않을 확률: $$0.95^{30} = 0.215$$)
        - 결측값을 다른 변수의 해당 값을을 기반으로 대체하는 방법 (평균, 중앙값 등)
        - 변수의 중요도를 측정하여 판단: 해당 변수가 예측에 미치는 영향을 판단하여 삭제 또는 대체 여부 결정

---

~~~python
medianBedrooms = housing_df['BEDROOMS'].median()
housing_df.BEDROOMS = housing_df.BEDROOMS.fillna(value = medianBedrooms)
~~~

---