---
layout: post
title: (Paper Survey) FORWARD LEARNING OF GRAPH NEURAL NETWORKS
description: |
  * 해당 글에는 잘못된 해석이 있을 수 있습니다. 피드백은 언제든 환영입니다.<br>
  **Forward-GNN**은 그래프 신경망(GNN)을 **역전파 없이** 학습할 수 있는 새로운 학습 프레임워크로, 각 레이어가 지역적인 정보만으로 효율적으로 학습할 수 있도록 설계된 모델이다.
categories: Paper_Survey
sitemap: false
hide_last_modified: true
---

# FORWARD LEARNING OF GRAPH NEURAL NETWORKS (Park et al., 2024)

## INTRODUCTION
- GNN은 추천, 신약 개발, 교통 예측 등 다양한 그래프 기반 문제에 활용되며, GNN은 대부분 역전파 알고리즘(backpropagation)에 기반하고 있다.
- 역전파(BP)는 신경망의 출력을 정답과 맞추기 위해, 순전파 후 계산된 오차를 최소화하도록 역전파를 수행하고, 기울기를 이용해 파라미터를 조정하는 대표적인 학습 알고리즘이다.
- BP는 **forward pass 중 activation 저장**, **non-local signal에 의존한 parameter update**, 마지막 **layer부터 역순으로만 업데이트되는 구조적 제약**을 가지며, 이는 **scalability(확장성)**, **parallelism(병렬성)**, **flexibility(유연성)**을 제한한다.
- forward-forward algorithm(FF)는 BP의 제약을 피하기 위해 forward-backward 대신 positive-negative data에 대해 두 번의 forward pass만으로 layer-wise 학습을 수행하는 새로운 학습 방식이다.
- 기존 FF 방식은 **negative data 생성의 비효율성**과 **그래프 데이터에의 부적합**으로 인해 GNN 학습에 효과적으로 적용되기 어렵다.
- Forward-GNN은 negative data 없이 **single forward pass**와 **양방향 layer-wise 학습**을 통해 GNN을 효율적으로 학습하는 새로운 forward learning 프레임워크다.

논문의 주요 기여는 다음과 같다:
- **Forward Graph Learning**: node classification과 link prediction 등 핵심 그래프 학습 과제에서 forward-only 학습 방식의 가능성을 체계적으로 탐구.
- **새로운 학습 프레임워크 제안**: message passing 방식에 독립적이며, top-down 신호를 활용해 single forward pass만으로 학습 가능한 Forward-GNN 개발.
- **효과성 검증**:
  - Forward-GNN이 link-prediction과 node-classifiaction에서 BP 대비 더 나은 성능 또는 동등한 성능을 보이며, 메모리 효율은 더 뛰어남.
  - 기존 FF 기반 방법보다 single-forward 접근이 더 우수함을 실험으로 입증.

## BACKGROUND AND RELATED WORK
### GRAPH NEURAL NETWORKS

**GNN**은 그래프 구조에서 각 노드의 **이웃 정보(Neighborhood)**들을 **message passing**을 통해 반복적으로 **집계(aggregation)**하면서 노드 임베딩을 학습하는 모델이다.

- 각 layer에서 노드 i의 임베딩은 자신의 **이전 임베딩**과 **이웃 노드들의 메시지**를 조합해 업데이트됨.
- 이웃 **노드 j의 정보와 엣지 feature $$ e(j,i) $$**(있는 경우)를 이용해 메시지를 생성하고, mean, max 등 순서에 영향을 받지 않는 방식으로 이를 집계.
- CNN의 convolution연산을 노드 간 연결이 일정하지 않은 그래프 데이터로 확장한 개념으로 볼 수 잇음.

**대표적인 GNN 아키텍쳐**:
- **GCN**: mean aggregation
- **GraphSAGE**: mean, max pooling 등 다양한 aggregation 선택
- **GAT**: attention mechanism을 활용한 weighted aggregation

하지만 GNN 학습은 **BP(backpropagation)를 사용**하기 때문에, 앞서 논의된 BP의 제약(확장성, 유연성, 병렬성 제한)을 그대로 가짐.

### THE FORWARD-FORWARD ALGORITHM
- Forward-Forward Algorithm(FF): backpropagation 없이 학습하는 새로운 방식

FF란?

FF는 BP(backpropagation) 없이 neural network를 학습하는 layer-wise forward-only 학습 방법으로, Geoffrey Hinton이 2022년에 제안. 각 layer를 bottom-up 방식으로 학습하며, **이미 학습된 layer는 이후 고정**됨.

- 핵심 아이디어
  - 두 번의 forward pass:
    - **Positive data**: 올바른 label이 포함된 데이터
    - **Negative data**: 잘못된 label이 포함된 데이터
  - Backward pass 없이, positive는 high goodness, negative는 low goodness가 되도록 각 layer가 independently 학습.

- **Goodness** (좋음 정도)란?
  - 각 layer가 계산한 ReLU activation의 L2 norm 제곱값: $$ \mathcal{G}_i^{(\ell)} = \left\| \mathbf{a}_i^{(\ell)} \right\|_2^2 $$
  - Positive는 goodness &uarr;, Negative는 goodness &darr; 되도록 학습
  - Positive로 판별될 확률: $$ p(\text{positive}) = \sigma \left( \mathcal{G}_i^{(\ell)} - \theta \right) $$ ($$ \sigma $$: sigmoid함수, $$ \theta $$: threshold)

- **Normalization**
  - Activation의 길이만으로 positive-negative를 구분하는 걸 방지하기 위해
    - 다음 layer에 넘기기 전 activation을 L2 normalization
    - 다음 layer는 **벡터의 방향(orientation)**을 기준으로 학습

- **확장 및 응용**
  - CaFo: 각 layer에 class predictor 추가, weights 고정
  - SymBa: positive-negative goodness gap 최대화
  - Predictive FF: predictive coding과 FF 결합
  - GCF: graph classification 전용 FF 확장 (node classification, link prediction은 불가)

&rarr; GCF의 한계를 넘어 Forward-GNN이 최초로 node classification / link prediction에 forward learning 적용!

**한 줄 요약**: FF는 BP 없이, positive-negative data에 두 번의 forward pass를 수행하며, layer-wise로 독립적으로 학습하는 새로운 신경망 학습 방법이다.

## FORWARD LEARNING OF GRAPH NEURAL NETWORKS

해당 연구는 FF를 확장해 single forward pass와 bottom-up-top-down 신호를 활용하는 새로운 GNN 학습 절차를 제안한다.

### ADAPTING THE FORWARD-FORWARD ALGORITHM FOR GRAPH NEURAL NETWORKS

<p align="center">
<img src="/assets/img/blog/paper_survey/Forward-GNN/fig1.png">
</p>

**기존 FF 문제점**: 이미지 분류에서는 label을 overlay(덧씌워서)해서 positive/negative 샘플을 만들었지만, 그래프 데이터에서는 이런 방식이 적용되지 않음 &rarr; GNN에 맞는 샘플 생성 방식이 필요!

- **GNN에 FF를 적용하기 위한 두 가지 방법**
- **<span style="color:blue">Node Feature 확장 방식(Extending Node Features with Node Labels)</span>**
  - **How?**
    - 노드의 label을 auxiliary feature(보조 특징)로 변환해서 원래 feature와 concat
    - 수식: $$ x'_i = \left[ x_i \,\|\, \mathrm{Label\text{-}To\text{-}Feat}(i) \right] $$ ($$x_i$$: 원래 노드 feature, $$\mathrm{Label\text{-}To\text{-}Feat}(i)$$: label &rarr; one-hot encoding)
  - **Positive/Negative 샘플 생성**:
    - **Positive**: 올바른 label 사용
    - **Negative**: 랜덤하게 잘못된 label 사용
  - **Label없는 node처리**:
    - label이 없는 경우 &rarr; **uniform distribution over classes** 사용 (모든 클래스가 동일 비중)
  - **특징**:
    - 그래프 구조는 유지
    - 노드 feature만 변경해서 positive/negative 샘플 구분

- **<span style="color:blue">Graph Structure 확장 방식 (Extending Graph Structure with Virtual Nodes)</span>**
  - **How?**
    - **각 클래스마다 virtual node 생성** (class 수 k &rarr; virtual node도 K개)
    - Real node $$ \leftrightarrow $$ Virtual node 연결:
      - **Positive**: true label의 virtual node에 연결
      - **Negative**: 잘못된 label의 virtual node에 연결
  - **수식 (확장된 그래프 구조)**: 
$$
G' = \left( V \cup C, \, E \cup \left\{ (i, c) \,\middle|\, 
i \in V_{\mathrm{labeled}}, \, 
c \in C, \, 
\mathrm{Label\text{-}RE}(i) = \mathrm{Label\text{-}VR}(c) 
\right\} \right)
$$
    - $$C = {c1, ... , c_K}$$: virtual nodes (각 클래스 1개씩)
    - $$ \mathrm{Label\text{-}RE}(i) $$: real node i에 대한 label 선택 (positive: correct, negative: random incorrect)
    - $$ \mathrm{Label\text{-}VR}(c) $$: virtual node c의 label
  - **특징**:
    - 노드 feature는 동일,
    - 그래프 구조를 변경해서 positive/negative 구분
  - **효과**:
    - Virtual node가 class representative 역할을 함
    - positive/negative의 차이는 virtual node 연결 유무

- **Inference(추론방법)**
  - Test 시:
    - **각 label에 대해 goodness 계산**
    - feature 확장 방식 &rarr; label을 one-hot encoding해 feature에 추가
    - 구조 확장 방식 &rarr; test node를 해당 virtual node에 연결
  - **모든 layer의 goodness 누적 &rarr; 가장 높은 goodness label 선택 = 예측 결과!**

- **정리요약**:

|방식|positive/negative 구분 방법|변경 요소|특징|
|:--:|:--:|:--:|:--:|
|Node Feature 확장|label encoding 추가 (one-hot or uniform)|node feature|그래프 구조 유지|
|Graph Structure 확장 (Virtual Node)|virtual node 연결 여부로 구분|graph 구조|feature 고정, 구조 변경|

### FORWARD GRAPH LEARNING VIA A SINGLE FORWARD PASS

- **기존 FF 방식의 한계**
  - FF와 `GNN에 FF를 적용하기 위한 두 가지 방법`에서 설명된 방식들은
    - **positive/negative 샘플을 따로 생성**해야 하고,
    - 각각 **따로 forward pass**를 해야 하므로
    - &rarr; **메모리 사용량과 계산량이 큼** (비효율적)
- **이 논문의 제안: Positive/Negative 샘플 없이 학습 &rarr; Single Forward Pass**
  - 입력 단계에서 label 정보를 perturbing(교란)해서 positive/negative 만드는 대신,
  - GNN이 forward 과정 안에서 직접 학습 signal을 생성하도록 설계

---

- **핵심 아이디어: Virtual Node를 활용한 Self-supervised Target 생성**
  - **Virtual Node 연결을 이용해서**
    - 각 class에 해당하는 **class representative (가상 노드)**가
    - real node의 **학습 target** 역할을 하도록 설계
  - **GNN의 message passing 과정 자체가**
    - &rarr; real node와 virtual nodes 사이에 label-based 관계를 반영하게 함
    - &rarr; 따로 positive/negative graph를 만들지 않아도 됨

- **학습 방법 (Learning Signals)**
  - 목표: 같은 class의 node들은 embedding space에서 가깝게, 다른 class node들은 멀어지게 embedding 학습
  - **사용하는 loss: Contrastive Learning Objective**

---

$$
\mathcal{L}^{(\ell)} = \frac{1}{|V_{\mathrm{labeled}}|} 
\sum_{i \in V_{\mathrm{labeled}}} 
\mathcal{L} \left( h^{(\ell)}_i, i, \ell \right), 
\quad 
\mathcal{L}(h, i, \ell) = 
- \log 
\frac{
\exp \left( C(h, c^{(\ell)}_{\mathrm{Label}(i)}) / \tau \right)
}{
\sum_{k \in [1, K]} 
\exp \left( C(h, c^{(\ell)}_{k}) / \tau \right)
}
$$

- $$ h^{(\ell)}_i $$: i번째 layer에서 node i의 embedding
- $$ c^{(\ell)}_k $$: l번째 layer에서 class k (virtual node)의 embedding
- $$ C(\cdot , \cdot) $$: similarity function (dot product 등)
- $$ \tau $$: temperature (scailing factor)

- **무슨 의미?**
  - node i가 자신의 label에 해당하는 class representative와 더 가까워지도록,
  - 다른 클래스 representative와는 멀어지도록 학습
  - &rarr; Contrastive Learning의 대표적인 InfoNCE Loss 형태와 유사
  - &rarr; 하지만 GNN 구조 안에서 virtual nodes와 real nodes 관계를 이용해 이를 자연스럽게 구현

---

- Inference(추론 방법)
1. 학습된 GNN으로, test node와 각 class represantative 사이의 similarity 계산
2. softmax 형태의 수식으로 **각 class에 대한 확률 분포 계산**
3. 모든 GNN layer에서 얻은 확률들을 평균 내어,
4. 가장 확률이 높은 label을 예측값으로 선택

---

- 왜 중요한가? (기존 FF 대비 차별점)

|항목|기존 FF 방식|본 논문 방식 (FORWARDGNN)|
|:--:|:--:|:--:|
|샘플 생성|Positive/Negative input 각각 생성|샘플 생성 불필요, label info로 self-target 생성|
|Forward Pass 횟수|2번(positive, negative 각각)|1번(single forward pass)|
|Training signal|Goodness-based (L2 norm)|Contrastive loss, class representative 기반|
|Efficiency|메모리$$\cdot$$계산량 높음|효율적 (메모리$$\cdot$$계산량 절감)|

### INCORPORATING TOP-DOWN SIGNALS

- **문제 상황 (기존 FF/Forward-GNN의 한계)**
  - 기존 FF 기반 forward learning은 
    - **layer-wise, bottom-up만 사용**
    - 아래 layer 학습 후 고정 &rarr; 위 layer 학습
  - ❌ 하지만 **BP**는 **top-down error signal이 아래로 전달** &rarr; lower layers도 upper layers의 학습 상황을 반영 가능
  - 이 top-down 정보 없이 학습하면 일부 중요한 global 구조 정보를 놓칠 수 있음

---

- **이 논문의 제안: Top-Down Signals 두 가지 경로로 사용**
- **Top-To-Input Path(입력 쪽에 반영)**
  - 현재 layer l의 input을:
    - 하위 layer output $$h_{i,t}^{(l-1)}$$
    - 상위 layers $$\{h_{i,t-1}^{(l+1)},...,h_{i,t-1}^{(L)}\}$$ (이전 time step t-1에서 얻은 top-down signal) &rarr; 합쳐서 (Merge 함수 사용) 입력으로 사용
  - 수식:
$$
h_{i,t}^{(l-1)'}=\text{Merge}(h_{i,t}^{(l-1)},\{h_{i,t-1}^{(l+1)},...,h_{i,t-1}^{(L)}\})
$$
  - Merge 함수 예시:
    - average
    - concatenate
    - immediate upper layer만 사용 &rarr; 다양한 방식 가능 (실험적으로 선택)

---

- **Top-To-Loss Path (loss 계산 쪽에 반영)**
  - **Loss 계산 시에도 top-down 정보 활용!**
  - **layer l에서의 output $$h_{i,t}^{(l)}$$**에:
    - 상위 layers $$\{h_{i,t-1}^{(l+1)},...,h_{i,t-1}^{(L)}\}$$ &rarr; **합쳐서 augmented output $$h_{i,t}^{(l)"}$$ 생성 &rarr; 이를 loss에 사용**
  - 수식: $$h_{i,t}^{(l)"}=\text{Merge}(h_{i,t}^{(l)},\{h_{i,t}^{(l+1)},...,h_{i,t-1}^{(L)}\})$$
  - 이 $$h_{i,t}^{(l)"}$$로 contrastive loss 계산

---

- **핵심 포인트**: Gradient는 여전히 흘러가지 않음. Forward-only learning 유지! &rarr; top-down 정보는 단지 input augmentation과 loss augmentation을 통해 사용될 뿐, BP처럼 error가 아래로 전파되는 게 아님.

---

- **왜 중요한가?**

|방식|Top-Down 정보 사용 여부|Gradient flow 존재 여부|학습 signal 경로|
|:--:|:--:|:--:|:--:|
|Backpropagation(BP)|O(error signal)|O|backward error propagation|
|기존 Forward-GNN|X|X|bottom-up only|
|Forward-GNN + top-down|O(representation signal)|X|forward pass + input/loss augmentation|

**&rarr; BP없이 top-down 정보를 활용할 수 있는 forward learning 구조 완성!**

---

### APPLICATION TO LINK PREDICTION

- **배경: Link Prediction이란?**
  - 목표:
    - 두 노드 i, j 사이에 **edge가 존재하는지 예측**
    - Positive edge (실제 연결됨) vs. Negative edge (연결 안 됨)
  - 일반적인 LP GNN 구조:
    - 각 노드 i, j에 대해 embedding $$h_i, h_j$$ 생성
    - 두 node embedding을 조합 &rarr; edge embedding 만들기: $$\text{EdgeEmb}(h_i,h_j)$$ (예: element-wise product, concatenation, etc.)
    - Edge embedding &rarr; link probability 계산: $$p_{i,j}=\text{LinkProb}(\text{EdgeEmb}(h_i,h_j))$$ (예: sum + sigmoid)

---

- **이 논문의 Forward Learning 적용 방식**:
  - 기존 BP 기반 LP 학습도 사실상 &rarr; positive/negative edge 구분을 통해 학습됨 (이 점이 FF와 유사!)
  - 따라서 FF나 Forward-GNN 방식이 LP에 자연스럽게 적용 가능

---

- **Forward Learning LP 방식**:
  - Positive edges: 실제 존재하는 edge (ground-truth)
  - Negative edges: 샘플링된 nonexistent edges (random negative sampling)
  - &rarr; 이 두 가지를 single forward pass 안에서 한 번에 계산
- **기존 FF와의 차이**:
  - Node classification FF에서는 label-based perturbation으로 positive/negative 생성해야 했지만,
  - Link prediction은 edge 존재 여부로 바로 positive/negative가 결정됨 &rarr; 추가적인 feature perturbation 필요 없음 &rarr; 바로 single forward pass 가능

---

- **Training 목표 (Loss 관점)**:
  - Positive edge (i,j): $$p_{i,j} \uparrow$$
  - Negative edge (i,j): $$p_{i,j} \downarrow$$
  - Binary classification 형태의 loss (예: BCE loss) 사용 가능

---

- **Top-Down Signal 활용**:
  - Node embedding을 top-down 방식으로 학습하면 &rarr; 그 embedding으로 만들어지는 edge embedding도 자연스럽게 top-down 정보를 반영하게 됨
  - 따라서 Link Prediction에서도 top-down signal incorporation 사용 가능

---

- **Inference (추론 방법)**:
  - 학습될 각 GNN layer에서 edge probability 계산: $$ p_{i,j}^{(l)}=\text{LinkProb}(\text{EdgeEmb}(h_i^{(l)},h_j^{(l)})) $$
  - 모든 layer의 결과를 평균: $$p_{i,j} = \frac{1}{L} \sum_{\ell=1}^{L} p_{i,j}^{(\ell)}$$
  - 최종적으로 이 평균값이 link 존재 확률

## EXPERIMENTS

### EVALUATION SETUP
목표: FORWARD-GNN의 효과성과 범용성을 검증하기 위해

- Node Classification
- Link Prediction

두 가지 그래프 학습 과제에서 평가

---

- **Node Classification**
  - Task: 노드의 클래스를 예측
  - 데이터 분할: train/val/test = 64% / 16% / 20%
  - 평가 지표: Classification Accuracy (정확도)
  - 방식: 5회 random split &rarr; 평균 & 표준편차 측정

---

- **Link Prediction**
  - Task: edge가 존재하는지 예측 (positive/negative edge 구분)
  - 데이터 분할: positive edges &rarr; 64% / 16% / 20%, negative edges는 동일 수량 샘플링
  - Link probability: 두 노드 embedding의 **dot product** &rarr; **sigmoid 적용**
  - 평가 지표: ROC-AUC

---

- **사용 데이터셋(5개)**

|Domain|Dataset Name|설명|
|:--:|:--:|:--:|
|Citation|PUBMED,CITESEER,CORAML|논문 인용 네트워크|
|E-commerce|AMAZON|상품 co-purchase 네트워크|
|Social|GITHUB|팔로우 관계 그래프|

---

**사용한 GNN 모델 (3종)**
- GCN (Graph Convolutional Network)
- GraphSAGE
- GAT (Graph Attention Network)

&rarr; FORWARD-GNN vs BP 성능 비교 실험

### COMPARISON WITH THE BACKPROPAGATION ALGORITHM

<p align="center">
<img src="/assets/img/blog/paper_survey/Forward-GNN/fig2-1.png">
<img src="/assets/img/blog/paper_survey/Forward-GNN/fig2-2.png">
</p>

- 비교 기준:
  - Task performance(정확도, ROC-AUC)
  - GPU 메모리 사용량 (layer 수 증가 시)
- 결과 요약:
  - Node Classification: SF가 BP와 비슷하거나 더 좋은 성능, 특히 layer수가 많아질수록 BP는 메모리 사용 급증, SF는 안정적
  - Link Prediction: SF가 대부분 BP보다 더 높은 성능, BP는 layer가 많아질수록 성능 저하
  - Memory: BP는 최대 18배 메모리 증가, SF는 layer수와 무관하게 메모리 사용 거의 일정

### COMPARE AMONG FORWARD LEARNING APPROACHES

<p align="center">
<img src="/assets/img/blog/paper_survey/Forward-GNN/table1.png">
</p>

- **Forward Learning 관련 주요 모델 설명**

|모델명|개요|
|:--:|:--|
|SF(Single-Forward)|단일 forward pass만으로 학습하는 본 논문 제안 방식. positive/negative 샘플 생성 없이 학습 신호를 직접 생성|
|SF-TopDown|SF에 top-down 정보를 결합하여, 각 layer가 상위 layer의 표현도 입력이나 loss에 활용하도록 확장|
|FF-VN(Forward-Forward with Virtual Nodes)|각 class에 대응되는 가상 노드를 그래프에 추가해 FF 방식을 그래프에 적용|
|FF-LA(Forward-Forward with Label Appending)|노드 feature에 label 정보를 부착하여 FF 방식으로 학습하는 그래프 변형 방식|
|FF-Symba|FF의 확장으로, positive/negative 샘플 간의 goodness 차이를 직접 최대화하는 objective를 사용|
|CaFo|FF 계열 접근으로, GNN의 각 layer에 개별적인 classifier를 붙여 layer-wise로 학습. GNN 가중치는 고정됨|
|PEPITA|FF 이전에 제안된 forward-only 방식으로, 생물학적 신경망 원리에 영감을 받은 학습 절차|

- **Link Prediction 관련 변형**

|모델명|개요|
|:--:|:--|
|ForwardGNN-CE|본 논문의 LP용 버전으로, cross-entropy loss를 적용한 single forward 학습 방식|
|ForwardGNN-FF|기존 FF objective를 link prediction에 맞게 적용한 방식|
|ForwardGNN-SymBa|FF-SymBa의 objective를 LP에 맞춰 변형한 학습 방식|
|CaFo (LP)|link-level predictor만 layer-wise로 학습하는 구조. GNN 자체는 고정된 상태로 사용됨|

---

<p align="center">
<img src="/assets/img/blog/paper_survey/Forward-GNN/fig3.png">
</p>

- **Node Classification 실험 결과:**
  - 가장 안정적이고 성능 좋은 방법: SF (Single-Forward), SF-TopDown
  - FF 계열(FF-VN, FF-LA, FF-SymBa): 높은 메모리 소모 + 학습 불안정
  - CaFo: 메모리는 적게 쓰지만 성능 매우 낮음 (GNN 자체를 학습하지 않음)
  - PEPITA: node classification에 효과적이지 않음

<p align="center">
<img src="/assets/img/blog/paper_survey/Forward-GNN/fig4.png">
</p>

- **Link Prediction 실험 결과**:
  - ForwardGNN-CE (BCE loss 기반): 성능 + 안정성 모두 가장 우수
  - ForwardGNN-SymBa: 성능 편차 큼
  - CaFo: layer-wise predictor만 학습 &rarr; 성능 매우 낮음

&rarr; 전반적으로 SF 기반 방법들이 가장 우수한 성능과 메모리 효율성을 동시에 보여줌

---

## CONCLUSION

- 문제의식: Backpropagation(BP)은 GNN 학습의 표준이지만, 확장성, 병렬성, 생물학적 타당성 측면에서 제약이 존재
- 본 연구의 기여:
  - Forward Graph Learning: Node Classification, Link Prediction 같은 핵심 과제에서 forward-only 학습 가능성을 체계적으로 탐구
  - 새로운 학습 프레임워크: 다양한 message passing 방식을 가진 GNN에 적용 가능한 FORWARDGNN 개발
  - 효과성 검증: FORWARDGNN은 BP보다 더 낮은 메모리 사용으로 비슷하거나 더 나은 성능을 보였으며, 제안한 single-forward 접근법이 기존 FF 기반 방법들보다 성능을 향상시킴