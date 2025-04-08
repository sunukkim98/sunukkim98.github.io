---
layout: post
title: (Paper Survey) Generating Sentences from a Continous Space
description: |
  * 해당 글에는 잘못된 해석이 있을 수 있습니다. 피드백은 언제든 환영입니다.
categories: Paper_Survey
sitemap: false
hide_last_modified: true
---
# Generating Sentences from a Continous Space

## 1. Introduction

- RNNLM: 자연어를 생설할 때 강력한 모델로, 긴 의존 관계를 잘 모델링할 수 있음. 하지만, **문법적 특성**이나 **주제와 같은 전역적인 정보**를 표현하는 데는 한계가 있다.
- 해당 연구는 RNNLM을 확장하여 잠재 변수(latent variable)를 사용해 전역적인 특징을 모델링하는 방안을 제시하며, 이를 위해 **변분 자동 인코더(variational autoencoder)**를 사용한다.
- 해당 연구에서는 **변분 자동 인코더(VAE)** 구조를 사용한 텍스트 모델을 제안하고, 이를 학습하는 데 있어 발생할 수 있는 문제와 해결책에 논의한다. 주요 기여는 다음과 같다:

    1. **언어 모델링 평가**: 글로벌 변수가 필요하지 않은 표준 언어 모델링 평가에서는 이 모델이 기존 RNNLM과 유사한 성능을 낸다고 제시.
    2. **누락된 단어 보간(task)**: 더 큰 코퍼스(더 많은 데이터)를 사용하여 누락된 단어를 예측하는 작업을 평가하며, 이를 위해 adversarial classifier(두 개의 모델이 서로 경쟁하며 학습하는 방식)를 사용하는 새로운 평가 전략을 도입한다. 이 방식을 계산이 어려운 우도 문제(확률 계산 문제)를 우회한다.
    3. **모델 분석**: 모델이 문장의 고차원적인 특징을 학습할 수 있는 능력을 분석하기 위한 정성적 기법(모델의 결과를 세부적으로 평가하는 방법)을 도입하고, 이 모델이 다양한 일관된 문장을 생성하며 문장간의 부드러운 보간(자연스러운 전환)이 가능함을 보여줌.

## 2. Preliminaries

### 2.1 Unsupervised sentence encoding

- 표준 RNN 언어 모델은 문장을 예측할 때 각 단어를 이전 단어와 변화하는 숨겨진 상태에 따라 예측하는 방식임. 하지만 표준 RNN 언어 모델은 문장 전체 **벡터 표현**을 학습하지 않으며, 이를 해결하기 위한 방법으로 **sequence autoencoder**, **skip-thought**, **paragraph vector**와 같은 non-generative techniques가 있다.
- **Sequence autoencoder**: 시퀀스 오토인코더는 지도학습을 위한 사전 훈련에 사용되엇으며, 문서 생성을 위한 모델로도 사용되었다. 오토인코더는 인코더 함수 `φenc`와 확률적 디코더 모델 `p(x|z~ = φenc(x))`로 구성되어 있으며, 예시 `x`에 대해 학습된 코드 `z~`를 조건으로 우도(확률)를 최대화한다. 시퀀스 오토인코더에서는 인코더와 디코더 모두 RNN이며, 예시는 토큰 시퀀스로 구성된다.

<p align="center">
<img src="/assets/img/blog/paper_survey/GeneratingSentencesfromaContinousSpace/tabel_1.png"><br>
<strong>Tabel 1</strong>
</p>

- **표준 오토인코더(Standard autoencoder)**가 문장의 글로벌 의미적 특징을 잘 추출하지 못한다. **Tabel 1**을 살펴보면 두 문장의 인코딩 사이에서 경로(homotopy)를 계산하고 각 중간 코드를 디코딩했을 때, 중간 문장들이 일반적으로 문법적으로 부자연스럽고 서로 부드럽게 전환되지 않는다. 또한, 새로운 문장을 생성하거나 확률을 할당하는데 사용할 수 없다.
- **Skip-thought** 모델은 비지도 학습 모델로, 시퀀스 오토인코더와 동일한 구조를 가지지만, 목표 문장 자체가 아닌 인접 문장을 조건으로 텍스트를 생성한다.
- **Paragraph vector**모델은 비순환적 문장 표현 모델로, 문장의 인코딩을 얻기 위해 그레디언트 기반 추론을 사용하여 문장의 단어들을 예측하는 데 필요한 인코딩 벡터를 학습한다.
- **Skip-thought**모델과 **Paragraph vector**모델은 문장 인코딩을 학습하는 데 유용하지만, 생성적 모델로 사용되진 않는다.

### 2.2 The variational autoencoder

- **변분 오토인코더(VAE)**는 표준 오토인코더의 **정규화된** 버전을 기반으로 하는 생성 모델이다. 이 모델은 숨겨진 코드 z~에 사전 확률 분포를 부여하여 코드들 간의 정규적인 기하학적 구조를 강제하며, 이를 통해 조상 샘플링(ancestral sampling)을 사용하여 모델로부터 적절한 샘플을 생성할 수 있게 한다.
- **VAE**는 결정론적 함수 φenc를 학습된 사후 인식 모델 `q(z~|x)`로 교체한다. 이 모델은 `z~`에 대한 근사 사후 분포(보통 **diagonal Gaussian**)를 신경망을 사용하여 `x`에 조건화하여 매개변수화한다. 직관적으로, VAE는 코드들을 단일점이 아니라 잠재 공간의 부드러운 타원체로 학습하여, 코드들이 훈련 데이터를 기억하는 것이 아니라 공간을 채우도록 만든다.
- 만약 VAE가 표준 오토인코더의 재구성 목표로 학습된다면 `q(z~|x)`의 분산이 매우 작아져 입력을 결정론적으로 인코딩 할 것이다. 하지만 VAE는 모델이 사후 분포를 사전 분포에 가깝게 유지하도록 유도하는 목표를 사용한다. 이 목표는 데이터의 실제 로그 우도(확률)에 대한 유효한 하한이 되어 VAE를 생성 모델로 만든다. **목표 함수**는 다음과 같다:

$$
\mathcal{L}(\theta; x) = -\text{KL}(q_{\theta}(\vec{z} \mid x) \| p(\vec{z})) + \mathbb{E}_{q_{\theta}(\vec{z} \mid x)}[\log p_{\theta}(x \mid \vec{z})] \leq \log p(x)
$$ 

- 이 목표 함수는 모델이 **잠재 공간**의 모든 지점에서 **그럴듯한 문장**을 디코딩할 수 있도록 강제한다.
- VAE 실험에서는 가우시안 분포를 사용하고, 경사 하강법으로 학습하며, 재구성 비용은 샘플을 이용해 추정하고, KL 발산 항은 닫힌 형태로 계산한다.

## 3. Proposed Method
## 4. Experiments
## 5. Conclusion