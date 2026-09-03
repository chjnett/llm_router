# KSC Overleaf 논문 작성 완전 가이드

이 문서는 현재 저장소의 실험만으로 한국정보과학회 한국소프트웨어종합학술대회(KSC)
논문을 Overleaf에서 작성하기 위한 단일 작업 지침서다. 파일 구성, 논문 주장 범위,
섹션별 초안, 표·수식·그림 위치, LaTeX 코드, 참고문헌, 최종 검수 순서까지 포함한다.

> **가장 빠른 방법:** `paper/overleaf/main_single.tex`의 내용을 Overleaf `main.tex`에
> 통째로 붙여넣고 `paper/figures/*.pdf` 네 개를 `figures/` 폴더에 업로드한다. 이 경우
> section 파일과 `references.bib`를 따로 만들 필요가 없다.

> **형식 확인 상태 (2026-09-03)**  
> 한국정보과학회 학술대회용 `kcc` LaTeX 템플릿과 Overleaf 공개 템플릿은 확인했다.
> 다만 KSC 2026의 공식 논문 모집 페이지와 페이지 제한은 아직 검색 결과에서 확인되지
> 않았다. 따라서 지금은 `kcc.cls` 기반으로 작성하고, 모집 공고가 게시되면 **페이지 수,
> 익명 심사 여부, 제출 파일 형식, 저자 표기 규칙**만 마지막에 다시 확인한다. 결과를 본
> 뒤 실험 기준을 바꾸지 않는 것과 마찬가지로, 형식을 추측해서 확정하지 않는다.

> **심사 피드백 반영 상태:** 통계표 수치는 원본 결과와 대조했으며 값은 바꾸지 않았다.
> 표 2는 열 넘침을 막도록 축소 조판하고, Figure 2는 양단 폭으로 배치한다. 또한
> self-consistency와의 차이, 별도 검증 모델을 쓰지 않는 이유, 위험 제약 최적화의 후속
> 절차를 본문에 명시했다.

---

## 1. 논문의 한 문장 정의

> 소형 모델의 첫 출력 confidence와 5~7토큰 규모의 독립 answer-only 검증을 결합하면,
> 대형 모델의 품질을 대부분 유지하면서 전체 풀이를 다시 생성하는 기존 cascade보다
> 실제 GPU 지연시간과 정규화 비용을 크게 줄일 수 있다.

논문의 중심은 **안전 인증**이 아니라 **성능–비용 frontier 개선**이다. 5% 안전 기준은
결과를 본 뒤 완화하지 않는다. 95% unsafe-risk 상한 5.195%는 한계로 명시한다.

### 허용되는 핵심 주장

1. 동일 RTX 3090에서 answer-only 검증은 full second pass 대비 latency를 94.09% 줄였다.
2. 실측 latency 환산 시 Always-Upper 대비 cascade 비용은 독립 인증 split에서 22.29%,
   SVAMP 공식 test에서 19.42% 감소했다.
3. C3 대비 exact McNemar 검정에서 유의한 정확도 차이가 관측되지 않았다
   (`p=0.625`, `p=0.146`).
4. 출력 기반의 짧은 검증은 query-only semantic routing이 놓치는 실제 모델 수행 가능성
   정보를 저비용으로 보완한다.

### 쓰면 안 되는 주장

- “정확도가 동일함을 증명했다.” → 유의차가 없다는 것이 동등성 증명은 아니다.
- “5% 이하의 안전성을 보장한다.” → 95% 상한이 5.195%라서 인증에 실패했다.
- “일반적인 모든 LLM과 과제에서 효과적이다.” → 현재 모델 한 쌍과 산술 과제 중심이다.
- “비용이 94.09% 감소했다.” → 94.09%는 **두 번째 호출 latency** 감소다. 전체 cascade
  비용 감소는 19.42~22.29%다.
- “0.12/0.82가 더 좋으므로 최종 정책이다.” → 결과 확인 후 발견한 점이며 본 정책은
  사전에 고정한 0.12/0.80이다.

---

## 2. 사용할 제목

### 권장 국문 제목

**출력 신뢰도와 저비용 정답 검증을 이용한 적응형 SLM–LLM 라우팅**

### 권장 영문 제목

**Adaptive SLM–LLM Routing with Output Confidence and Low-Cost Answer Verification**

### 대안 제목

- 정답 전용 검증을 활용한 비용 효율적 소형–대형 언어 모델 라우팅
- Output-Aware Answer Verification for Cost-Efficient SLM–LLM Cascades

제목에는 “safe”, “certified”, “guaranteed”를 넣지 않는다.

### 국문 용어 통일표

영문은 첫 등장 때만 괄호로 병기하고 이후에는 아래 국문 표현을 쓴다. C3,
Always-Upper처럼 실험 방법을 식별하는 고유 이름은 표 안에서 유지해도 된다.

| 원 표현 | 본문 권장 표현 |
|---|---|
| confidence | 신뢰도(confidence) → 이후 `신뢰도` |
| answer-only verifier | 정답 전용(answer-only) 검증 → 이후 `정답 전용 검증` |
| escalation | 상위 모델 전달 |
| split | 분할 |
| routing score | 라우팅 점수 |
| normalized cost | 정규화 비용 |
| unsafe routing | 위험 라우팅 |
| latency | 지연시간 |

---

## 3. Overleaf 프로젝트 만들기

1. 한국정보과학회 학술대회 공개 Overleaf 템플릿을 연다.
2. `Open as Template`를 선택하여 새 프로젝트를 만든다.
3. 템플릿 안의 `kcc.cls`, 기본 package 파일과 bibliography 설정을 유지한다.
4. 저장소의 `paper/figures/*.pdf` 네 개를 Overleaf의 `figures/` 폴더에 업로드한다.
5. 아래 구조로 `.tex` 파일을 만든다.
6. 컴파일러는 우선 템플릿 기본값을 사용한다. 한글이 깨질 때만 템플릿 설명에 따라
   XeLaTeX 또는 한글 package 설정을 조정한다.

권장 프로젝트 구조:

```text
overleaf-project/
├── main.tex
├── kcc.cls
├── packages.tex             # 템플릿에 존재하면 유지
├── definitions.tex          # 템플릿에 존재하면 유지
├── sections/
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_method.tex
│   ├── 04_experiments.tex
│   ├── 05_results.tex
│   └── 06_conclusion.tex
├── figures/
│   ├── fig1_architecture.pdf
│   ├── fig2_latency_tokens.pdf
│   ├── fig3_pareto_frontier.pdf
│   └── fig4_method_comparison.pdf
└── references.bib
```

PDF를 우선 사용하는 이유는 벡터 선과 글자가 확대해도 선명하고 `pdflatex` 호환성이
좋기 때문이다. SVG는 편집용, PNG는 GitHub/미리보기용으로 남긴다.

---

## 4. `main.tex` 시작 코드

공개 템플릿의 명령 이름이 다르면 템플릿 예제를 우선한다. 확인된 공개 템플릿은
`\documentclass[preprint]{kcc}`를 심사용, `\documentclass{kcc}`를 출판용으로 안내한다.

```latex
\documentclass[preprint]{kcc} % 심사용; 최종본 규칙은 해당 연도 공고 확인

\input{packages}
\input{definitions}

% packages.tex에 없을 때만 추가
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{multirow}

\title{출력 신뢰도와 저비용 정답 검증을 이용한 적응형 SLM--LLM 라우팅}
\author{
제1저자$^{\circ}$, 공동저자, 교신저자\\
소속기관명\\
\{first, second, corresponding\}@example.ac.kr
}

\engtitle{Adaptive SLM--LLM Routing with Output Confidence and Low-Cost Answer Verification}
\engauthor{
First Author, Second Author, Corresponding Author\\
Affiliation\\
}

\abstract{\input{sections/00_abstract}}

\begin{document}
\maketitle

\section{서론}
\input{sections/01_introduction}

\section{관련 연구}
\input{sections/02_related_work}

\section{제안 방법}
\input{sections/03_method}

\section{실험 설정}
\input{sections/04_experiments}

\section{실험 결과}
\input{sections/05_results}

\section{결론}
\input{sections/06_conclusion}

\bibliographystyle{ieeetr}
\bibliography{references}
\end{document}
```

---

## 5. 페이지 구성 전략

공식 페이지 제한이 공개되기 전에는 아래 4쪽 버전을 기준으로 작성한다. 2쪽 제한이면
관련 연구와 실험 설정을 합치고 그림 4를 제거한다.

| 부분 | 4쪽 목표 | 2쪽 압축 시 |
|---|---:|---:|
| 제목·초록 | 0.35쪽 | 0.25쪽 |
| 서론·관련 연구 | 0.75쪽 | 0.40쪽 |
| 제안 방법 | 0.85쪽 | 0.45쪽 |
| 실험 설정 | 0.55쪽 | 0.25쪽 |
| 결과·분석 | 1.10쪽 | 0.50쪽 |
| 결론·참고문헌 | 0.40쪽 | 0.15쪽 |

그림 우선순위는 `fig1 > fig2 > fig3 > fig4`다. 공간이 부족하면 fig4를 먼저 삭제하고
방법 비교는 표 하나로 남긴다.

---

## 6. 초록: 그대로 붙여넣을 초안

`sections/00_abstract.tex`에 아래 내용을 넣는다.

```latex
대규모 언어 모델은 높은 추론 성능을 제공하지만 모든 요청을 대형 모델에 할당하면
계산 비용과 지연시간이 증가한다. 본 연구는 소형 모델의 질문 임베딩뿐 아니라 실제
생성 출력의 confidence와 짧은 독립 검증 결과를 이용하는 적응형 SLM--LLM 라우팅
방법을 제안한다. 제안 방법은 Qwen2.5-1.5B가 먼저 답을 생성하고, 경계 사례에 대해서만
설명을 제외한 answer-only 검증을 수행한 후 두 숫자 답의 일치 여부에 따라
Qwen2.5-7B로 escalation한다. GSM8K에서 학습한 confidence router를 SVAMP로 전이하여
분리된 정책 선택 및 인증 split과 공식 test에서 평가하였다. RTX 3090 동일 조건
실험에서 answer-only 검증은 기존 두 번째 전체 풀이보다 문항당 지연시간을
492.97 ms에서 29.15 ms로 94.09\% 감소시켰다. 실측 지연시간으로 환산한 cascade
비용은 독립 인증 split과 공식 test에서 Always-Upper 대비 각각 22.29\%와 19.42\%
감소했으며, 정확도는 각각 88.00\%와 85.67\%를 기록하였다. C3 기준선과의 exact
McNemar 검정에서는 유의한 정확도 차이가 관측되지 않았다. 한편 95\% unsafe-risk
상한은 5.195\%로 사전 설정한 5\% 기준을 근소하게 충족하지 못했다. 결과적으로
제안 방법은 관측된 성능--비용 frontier를 개선하지만, 안전 보장과 정확도 비열등성의
확증에는 더 큰 독립 표본이 필요하다.
```

초록에는 인용, 표, 그림 번호를 넣지 않는다. 저자·소속이 확정되면 마지막으로 180~250
단어 또는 학회 지정 길이에 맞춘다.

---

## 7. 서론: 문단별로 무엇을 쓸지

### 문단 1 — 문제와 중요성

```latex
대형 언어 모델(LLM)은 다양한 추론 과제에서 높은 성능을 보이지만, 모든 요청을 대형
모델로 처리하는 방식은 불필요한 계산 비용과 응답 지연을 유발한다. 실제 요청 중 일부는
소형 언어 모델(SLM)로도 정확히 처리할 수 있으므로, 입력별로 적절한 모델을 선택하는
라우팅은 품질과 비용을 동시에 관리하기 위한 핵심 기술이다.
```

### 문단 2 — 기존 방식의 한계

```latex
기존 LLM 라우팅은 질문 임베딩, 선호 데이터 또는 별도의 분류기를 이용해 강한 모델의
필요성을 예측한다~\cite{chen2024frugalgpt,ong2025routellm}. 그러나 의미적으로 유사한
질문도 SLM 관점의 난이도가 다를 수 있으며, query-only 표현은 SLM이 실제로 생성한
답의 불확실성과 자기 일관성을 직접 반영하지 못한다. 또한 경계 사례마다 전체 풀이를
두 번 생성하면 검증 자체의 decode 비용이 커져 cascade의 경제성이 약해진다.
```

### 문단 3 — 제안 아이디어

```latex
본 연구는 첫 번째 SLM 출력에서 계산한 confidence와 독립 answer-only 검증을 결합한다.
라우터가 확실한 사례는 첫 답을 즉시 채택하고, 경계 사례는 동일 SLM이 숫자 답만 짧게
재생성하도록 한다. 두 답이 일치하면 SLM 출력을 채택하고, 불일치하거나 confidence가
낮으면 Upper LLM으로 escalation한다. 이 구조는 출력 정보를 사용하면서도 두 번째
검증의 생성 길이를 최소화한다.
```

### 문단 4 — 기여점

```latex
본 연구의 기여는 다음과 같다. 첫째, 질문 의미와 실제 SLM 출력 confidence를 결합한
두 임계값 라우팅 구조를 제시한다. 둘째, 전체 재풀이 대신 평균 5.40토큰의 answer-only
검증을 사용해 동일 GPU에서 두 번째 호출 지연시간을 94.09\% 줄인다. 셋째, 정책 선택과
평가 split을 분리하고 paired bootstrap 및 McNemar 검정을 통해 정확도--비용 trade-off를
정량적으로 분석한다. 넷째, 안전 기준의 근소 실패도 함께 보고하여 방법의 적용 범위를
명확히 한다.
```

---

## 8. 관련 연구: 세 덩어리로 작성

### 8.1 비용 효율적 LLM cascade와 routing

FrugalGPT는 여러 모델을 순차적으로 호출하는 cascade를 통해 품질과 비용을 함께
최적화하는 문제를 제시했다. RouteLLM은 강한/약한 모델 사이의 선택을 선호 데이터로
학습했다. 두 연구를 소개한 다음, 본 연구는 **라우팅 점수뿐 아니라 SLM이 방금 생성한
답의 confidence와 저비용 독립 검증을 사용한다**는 차이를 쓴다.

```latex
FrugalGPT는 이질적인 비용 구조를 가진 모델을 cascade로 결합하여 비용--품질
trade-off를 최적화하였다~\cite{chen2024frugalgpt}. RouteLLM은 인간 선호 데이터와
증강을 이용해 강한 모델과 약한 모델 사이의 라우터를 학습하였다~\cite{ong2025routellm}.
이들과 달리 본 연구는 입력 표현만으로 모델을 선택하지 않고, 약한 모델이 생성한
출력의 confidence와 독립적인 짧은 숫자 검증을 라우팅 신호로 사용한다.
```

### 8.2 출력 검증과 선택적 예측

GSM8K 원 논문은 후보 풀이의 verifier가 수학 추론 성능을 개선할 수 있음을 보였다.
Selective Generation은 생성 모델에서 위험 통제의 필요성을 다룬다. 본 연구의 검증기는
후보를 점수화하는 별도 대형 모델이 아니라 **동일 SLM의 매우 짧은 별도 프롬프트 답 생성**이다.

```latex
GSM8K 연구는 생성 후보를 평가하는 verifier의 효과를 보였다~\cite{cobbe2021gsm8k}.
선택적 생성 연구는 신뢰할 수 없는 출력을 거부하거나 선택할 때 위험 통제가 필요함을
보였다~\cite{lee2024selective}. 본 연구는 별도의 검증 모델을 학습하지 않고 동일
SLM의 정답 전용 재생성을 이용해 첫 출력과의 숫자 일치를 확인한다. 전통적
self-consistency가 동일 프롬프트에서 확률적으로 여러 출력을 표본화해 다수결하는 것과
달리~\cite{wang2023selfconsistency}, 본 방법은 greedy decoding을 유지하면서 풀이 프롬프트와 정답 전용 프롬프트라는
서로 다른 두 관점을 사용하고 신뢰도 경계 사례에만 두 번째 호출을 수행한다. 따라서
별도 가중치의 적재·상주 비용 없이 평균 5.40토큰의 추가 디코딩만 사용한다.
```

여기서 `독립`이라는 단어는 통계적 독립으로 오해될 수 있으므로 검증 출력에는 쓰지
않는다. 데이터 분할에만 `독립 인증 분할`이라는 표현을 사용한다.

### 8.3 의미 임베딩과 능력 경계

BGE는 일반적인 semantic embedding 기준선이다. 현재 실험에서 frozen BGE router AUC가
낮았던 결과를 관련 연구의 결함처럼 쓰지 말고, **의미 유사성과 모델 수행 가능성은 다른
신호**라는 동기로만 사용한다.

```latex
범용 임베딩 모델은 의미적 유사성을 효과적으로 표현하지만~\cite{xiao2023bge}, 같은
주제의 질문도 SLM이 해결할 수 있는 난이도는 다를 수 있다. 본 연구의 초기 GSM8K
실험에서 frozen BGE 기반 logistic router의 AUC는 0.589였고, 출력 confidence 기반
router는 0.722를 기록하였다. 이는 의미 표현을 폐기해야 한다는 의미가 아니라,
출력 행동 신호가 능력 경계를 보완할 수 있음을 시사한다.
```

---

## 9. 제안 방법

### 9.1 전체 구조와 Figure 1 위치

이 절 첫 문단 직후 `fig1_architecture.pdf`를 넣는다.

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/fig1_architecture.pdf}
  \caption{출력 confidence와 answer-only 검증을 결합한 제안 라우팅 구조. 높은
  confidence의 요청은 즉시 채택하고, 경계 사례는 짧게 재검증하며, 불일치 또는 낮은
  confidence 사례만 Upper LLM으로 전달한다.}
  \label{fig:architecture}
\end{figure}
```

본문에서 반드시 `그림~\ref{fig:architecture}`로 먼저 호출한다.

### 9.2 출력 confidence router

첫 SLM 출력에서 얻는 11차원 특징 벡터를 다음처럼 정의한다.

```latex
\begin{equation}
\mathbf{x}_i = [\mu_{\log p},\min_{\log p},\sigma_{\log p},
\mu_H,\max_H,\mu_M,\min_M,T,C,N,F],
\end{equation}
```

- `\mu_{\log p},\min_{\log p},\sigma_{\log p}`: 생성 토큰 log-probability의 평균,
  최솟값, 표준편차
- `\mu_H,\max_H`: token entropy의 평균과 최댓값
- `\mu_M,\min_M`: top-1/top-2 probability margin의 평균과 최솟값
- `T,C`: completion token 수와 문자 수
- `N`: 출력에서 추출된 숫자의 개수
- `F`: `Final answer` 형식 존재 여부

이는 `src/run_confidence_router.py`의 `FEATURES` 11개와 같은 순서다. 최종 원고에서도
존재하지 않는 feature를 추가하지 않는다.

Logistic regression router:

```latex
\begin{equation}
q_i = \sigma(\mathbf{w}^{\top}\mathbf{x}_i+b),
\end{equation}
```

여기서 `q_i`는 Lower가 안전하게 처리 가능한 정도를 나타내는 라우팅 점수다. 논문에서는
확률 보정이 완벽하다고 주장하지 말고 “routing score”라고 부르는 편이 안전하다.

### 9.3 두 임계값 정책

고정 정책은 `\tau_l=0.12`, `\tau_h=0.80`이다.

```latex
\begin{equation}
\pi(q_i)=
\begin{cases}
\text{accept Lower}, & q_i \ge \tau_h,\\
\text{answer-only verify}, & \tau_l \le q_i < \tau_h,\\
\text{escalate Upper}, & q_i < \tau_l.
\end{cases}
\end{equation}
```

중간 영역에서는 첫 숫자 답 `a_i^{(1)}`와 answer-only 숫자 답 `a_i^{(v)}`가 모두
추출 가능하고 동일할 때만 Lower를 채택한다.

```latex
\begin{equation}
\text{accept}_i = [q_i\ge\tau_h]\;\lor\;
([\tau_l\le q_i<\tau_h]\land[a_i^{(1)}=a_i^{(v)}]).
\end{equation}
```

### 9.4 비용 함수

```latex
\begin{equation}
C_i = C_L + I_i^{(v)} C_V + I_i^{(U)} C_U,
\qquad
\tilde{C}=\frac{1}{N}\sum_{i=1}^{N}\frac{C_i}{C_U}.
\end{equation}
```

- `C_L=1`
- `C_U=4.667`
- 보수적 분석의 `C_V=0.35`
- 동일 GPU latency 실측 환산의 `C_V=0.0591`

본문 주 결과는 실측 환산 비용을 쓰되, 표 각주에 보수적 0.35 가정에서도 인증 split
비용이 0.830이라고 병기한다. latency 비율이 API 가격이나 에너지 비용과 같다고 쓰지
않는다.

---

## 10. 실험 설정

### 10.1 데이터와 split

```latex
라우터 학습에는 GSM8K를 사용하고, 교차 과제 평가는 SVAMP에서 수행하였다
~\cite{cobbe2021gsm8k,patel2021svamp}. GSM8K pilot은 2,000문항을 router train
1,000, distillation 400, validation 200, test 400으로 분리하였다. SVAMP train 700문항은
초기 validation 200, 정책 선택 250, 독립 risk certification 250으로 겹치지 않게
분리했으며, 공식 test 300문항은 최종 진단에만 사용하였다. 모든 split은 고정 seed 42와
문항 ID로 관리하였다.
```

중요: 공식 test 결과를 보고 0.12/0.80을 선택했다고 쓰면 안 된다. 정책은 별도 selection
split에서 선택하고 certification 전에 고정했다.

### 10.2 모델과 하드웨어

```latex
Lower와 Upper는 각각 Qwen2.5-1.5B-Instruct와 Qwen2.5-7B-Instruct를 사용하였다
~\cite{yang2024qwen25}. 두 모델은 4-bit로 로드했으며 단일 NVIDIA RTX 3090 24GB에서
평가하였다. 생성은 greedy decoding을 사용하였다. 첫 답은 최대 256토큰, answer-only
검증은 최대 64토큰으로 제한했으나 실제 검증 출력은 평균 5.40토큰이었다.
```

### 10.3 평가 지표

- Accuracy: 최종 cascade 정답률
- Upper-call rate: Upper LLM으로 escalation한 비율
- Unsafe routing: Lower를 채택했으나 Lower 답이 틀린 전체 문항 비율
- Normalized cost: Always-Upper 비용을 1로 둔 상대 비용
- Latency: 워밍업 후 동일 128문항, batch 16, 조건별 2회 ABBA 순서의 문항당 시간
- 통계: 동일 문항 10,000회 paired bootstrap CI와 exact McNemar 검정
- 위험 상한: 독립 250문항에서 one-sided 95% Clopper--Pearson upper bound

### 10.4 비교 방법

1. Always Lower
2. Always Upper
3. Query-only BGE + logistic/kNN 초기 기준선
4. C3: confidence + full second-pass consistency
5. Answer-only: confidence + 짧은 정답 전용 consistency (제안 방법)

공간이 부족한 KSC 단문에서는 3번은 서론의 AUC 한 줄로 줄이고, 결과표에는 C3,
Answer-only, Always Upper만 남긴다.

---

## 11. 결과 섹션과 그림 배치

### 11.1 Figure 2 — latency와 token

결과 절의 첫 번째 그림으로 배치한다. 두 개의 패널이 한 column에서 과도하게 축소되지
않도록 양단 폭 `figure*`를 사용한다.

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=0.88\textwidth]{figures/fig2_latency_tokens.pdf}
  \caption{동일 RTX 3090에서 두 번째 전체 풀이와 정답 전용 검증의 문항당
  지연시간 및 생성 토큰 수 비교. 128문항을 batch size 16, 조건별 2회 ABBA 순서로
  측정하였다.}
  \label{fig:latency}
\end{figure*}
```

설명 문단:

```latex
그림~\ref{fig:latency}에서 answer-only 검증은 두 번째 전체 풀이의 평균 124.55토큰을
5.40토큰으로 줄였다. 이에 따라 문항당 지연시간은 492.97 ms에서 29.15 ms로 94.09\%
감소했고, 처리량은 2.03 items/s에서 34.32 items/s로 증가하였다. 이는 검증 호출을
제거한 결과가 아니라 동일 Lower 모델의 decode 길이를 줄인 결과다.
```

### 11.2 주 결과표

표는 Figure 2 직후 또는 다음 column 상단에 넣는다.

```latex
\begin{table}[t]
\centering
\caption{SVAMP에서 고정 정책의 정확도 및 실측 latency 환산 비용}
\label{tab:main}
\small
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{llrrr}
\toprule
Split & Method & Acc.(\%) & Unsafe(\%) & Cost \\
\midrule
Cert. & Always Upper & 88.80 & -- & 1.000 \\
      & C3           & 88.80 & 2.00 & 1.008 \\
      & Answer-only  & 88.00 & 2.80 & \textbf{0.777} \\
\midrule
Test  & Always Upper & 88.00 & -- & 1.000 \\
      & C3           & 87.67 & 1.00 & 1.045 \\
      & Answer-only  & 85.67 & 3.00 & \textbf{0.806} \\
\bottomrule
\end{tabular}
\end{table}
```

표 설명:

```latex
표~\ref{tab:main}에서 제안 방법은 Always-Upper 대비 certification과 공식 test의
정규화 비용을 각각 22.29\%와 19.42\% 줄였다. C3 대비 정확도는 각각 0.8\%p와
2.0\%p 낮았으나 exact McNemar 검정은 유의하지 않았다. 그러나 이 결과는 정확도
동등성 또는 비열등성을 입증하지 않는다.
```

### 11.3 Figure 3 — Pareto frontier

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=0.92\columnwidth]{figures/fig3_pareto_frontier.pdf}
  \caption{정책 선택 split의 성능--비용 Pareto frontier. 보라색 점은 독립 평가 전에
  고정한 임계값 0.12/0.80 정책이다.}
  \label{fig:pareto}
\end{figure}
```

```latex
그림~\ref{fig:pareto}는 threshold 변화에 따라 비용과 정확도가 함께 증가하는 양상을
보인다. 본 연구는 selection split에서 quality와 unsafe 제약을 만족한 정책 0.12/0.80을
고정하였다. 그림의 0.12/0.82 지점은 frontier 설명용이며 사후에 최종 정책으로
교체하지 않았다.
```

### 11.4 통계표

```latex
\begin{table}[t]
\centering
\caption{정답 전용 검증과 C3의 대응 비교(단위: \%p)}
\label{tab:paired}
\small
\setlength{\tabcolsep}{2.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lrrr}
\toprule
분할 & $\Delta$정확도 [95\% CI] & $p_{McN}$ & $\Delta$비용 [95\% CI] \\
\midrule
Cert. & $-0.8$ [$-2.4,0.8$] & .625 & $-23.06$ [$-28.70,-17.38$] \\
Test  & $-2.0$ [$-4.33,0.01$] & .146 & $-23.87$ [$-29.58,-18.27$] \\
\bottomrule
\end{tabular}
}
\end{table}
```

두 행의 값은 결과 JSON과 다시 대조한 값이며, 인증 분할은 `-0.8 [-2.4, 0.8]`, 공식
테스트는 `-2.0 [-4.33, 0.01]`이다. 신뢰구간과 차이는 %p 기준임을 caption 또는 본문에 쓴다. 비용 신뢰구간이 0을
포함하지 않는다고 “통계적으로 비용이 감소했다”고 쓸 수 있지만, latency 반복이 2회뿐인
점도 제한사항에 남긴다.

### 11.5 Figure 4 — 공간이 남을 때만

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/fig4_method_comparison.pdf}
  \caption{독립 certification split과 SVAMP 공식 test에서 C3와 answer-only 정책의
  정확도 및 정규화 비용 비교.}
  \label{fig:comparison}
\end{figure}
```

4쪽 이하에서 표~\ref{tab:main}과 정보가 중복되므로 지면이 부족하면 이 그림을 뺀다.

---

## 12. 위험 결과와 한계 작성법

반드시 결과 또는 결론 앞에 아래 문단을 넣는다.

```latex
독립 certification 250문항에서 answer-only 정책의 unsafe 오류는 7건(2.80\%)이었다.
이에 대한 단측 95\% Clopper--Pearson 상한은 5.195\%로, 사전 설정한 5\% 기준을
0.195\%p 초과하였다. 따라서 본 결과를 5\% 위험 보장으로 해석하지 않는다. 반면
안전 인증을 통과한 C3 정책은 정규화 비용이 1.008로 Always-Upper보다 높았다. 이는
현재 표본과 구조에서 엄격한 안전성과 비용 절감을 동시에 달성하지 못했음을 보여준다.
```

제한사항 문단:

```latex
본 연구에는 네 가지 제한이 있다. 첫째, 산술 문장제와 하나의 Qwen 모델 계열만 평가하여
다른 도메인과 모델 조합으로의 일반화가 확인되지 않았다. 둘째, 공식 test가 300문항으로
정확도 비열등성을 검정하기에 충분하지 않다. 셋째, latency 측정은 단일 RTX 3090과
조건별 2회 반복에 한정되며 API 가격, 에너지 소비 또는 다중 사용자 serving 비용과
동일하지 않다. 넷째, answer-only 정책은 5\% 위험 인증을 근소하게 통과하지 못했다.
후속 연구에서는 더 큰 사전 등록 표본과 다양한 모델·도메인에서 비열등성 및 위험
통제를 재평가할 필요가 있다.
```

위 제한사항 직후에는 사후적으로 5% 기준을 완화하지 않고, 다음과 같은 위험 제약
최적화를 후속 방법으로 제시한다.

```latex
후속 연구에서는 정책 선택 분할에서
$\min_{\tau_l,\tau_h}\widetilde{C}(\tau_l,\tau_h)$를 최적화하되 위험률의 단측
신뢰상한이 목표 $\alpha$ 이하라는 제약을 동시에 적용하고, 고정된 임계값을 새로운 독립
분할에서 다시 인증할 예정이다.
```

핵심은 현재 공식 테스트 결과를 보고 임계값을 다시 고르는 것이 아니다. 새 정책 선택
분할에서 임계값을 고정하고, 그 뒤에 열지 않은 인증 분할에서 다시 평가해야 한다.

---

## 13. 결론 초안

`sections/06_conclusion.tex`:

```latex
본 연구는 SLM의 출력 confidence와 짧은 answer-only 검증을 결합한 적응형 SLM--LLM
라우팅 방법을 제안하였다. 전체 풀이를 다시 생성하는 C3 방식과 비교해 answer-only
검증은 동일 GPU에서 두 번째 호출 latency를 94.09\% 줄였으며, 실측 latency 환산
cascade 비용을 Always-Upper 대비 19.42--22.29\% 절감하였다. 동일 문항 paired
검정에서 C3와 유의한 정확도 차이는 관측되지 않았지만, 이는 동등성 증명을 의미하지
않는다. 또한 95\% unsafe-risk 상한은 5.195\%로 사전 안전 기준을 근소하게 초과했다.
향후에는 더 큰 독립 표본과 다양한 모델 조합에서 비열등성, 위험 통제 및 serving 환경의
실제 비용을 함께 검증할 예정이다.
```

---

## 14. `references.bib` 초안

아래 BibTeX는 시작점이다. 최종 제출 전 저자명, venue, 페이지, DOI를 원 논문에서 다시
확인한다. 특히 arXiv 논문이 정식 학회에 게재된 경우 정식 venue를 우선한다.

```bibtex
@article{chen2024frugalgpt,
  title={FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance},
  author={Chen, Lingjiao and Zaharia, Matei and Zou, James},
  journal={Transactions on Machine Learning Research},
  year={2024}
}

@inproceedings{ong2025routellm,
  title={RouteLLM: Learning to Route LLMs with Preference Data},
  author={Ong, Isaac and Almahairi, Amjad and Wu, Vincent and Chiang, Wei-Lin and Wu, Tianhao and Gonzalez, Joseph E. and Kadous, M. Waleed and Stoica, Ion},
  booktitle={International Conference on Learning Representations},
  year={2025}
}

@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}

@inproceedings{patel2021svamp,
  title={Are NLP Models Really Able to Solve Simple Math Word Problems?},
  author={Patel, Arkil and Bhattamishra, Satwik and Goyal, Navin},
  booktitle={Proceedings of NAACL-HLT},
  year={2021}
}

@article{yang2024qwen25,
  title={Qwen2.5 Technical Report},
  author={Yang, An and others},
  journal={arXiv preprint arXiv:2412.15115},
  year={2024}
}

@article{xiao2023bge,
  title={C-Pack: Packed Resources for General Chinese Embeddings},
  author={Xiao, Shitao and Liu, Zheng and Zhang, Peitian and Muennighoff, Niklas and Lian, Defu and Nie, Jian-Yun},
  journal={arXiv preprint arXiv:2309.07597},
  year={2023}
}

@inproceedings{lee2024selective,
  title={Selective Generation for Controllable Language Models},
  author={Lee, Minjae and Kim, Kyungmin and Kim, Taesoo and Park, Sangdon},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}

@article{clopper1934use,
  title={The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial},
  author={Clopper, Charles J. and Pearson, Egon S.},
  journal={Biometrika},
  volume={26},
  number={4},
  pages={404--413},
  year={1934}
}

@inproceedings{wang2023selfconsistency,
  title={Self-Consistency Improves Chain of Thought Reasoning in Language Models},
  author={Wang, Xuezhi and Wei, Jason and Schuurmans, Dale and Le, Quoc V. and Chi, Ed H. and Narang, Sharan and Chowdhery, Aakanksha and Zhou, Denny},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

---

## 15. 그림 재생성 방법

저장소 루트 `C:/llm_models/llm_router`에서 실행한다. 현재 PC에서는 Matplotlib이 base
Anaconda에 설치되어 있다.

```powershell
C:\Users\uns\anaconda3\python.exe -m src.render_paper_figures
```

입력 데이터:

```text
paper/data/paper_results.json
```

출력:

```text
paper/figures/fig1_architecture.{pdf,svg,png}
paper/figures/fig2_latency_tokens.{pdf,svg,png}
paper/figures/fig3_pareto_frontier.{pdf,svg,png}
paper/figures/fig4_method_comparison.{pdf,svg,png}
```

수치가 바뀌었을 때는 `paper_results.json`을 임의로 손으로 고치지 말고 원본
`artifacts/results/*.json`과 대조한 후 업데이트한다. 그림 생성 뒤 PNG를 눈으로 확인하고
PDF를 Overleaf에 업로드한다.

---

## 16. 저장소 결과와 논문 수치 대응표

| 논문 내용 | 원본/요약 파일 |
|---|---|
| 통합 연구 흐름 | `adaptive_slm_llm_routing_architecture_v1.html` |
| KSC 결과 요약 | `RESULTS_KSC_PILOT.md` |
| figure 입력 snapshot | `paper/data/paper_results.json` |
| latency 원본 | `artifacts/results/verifier_latency_benchmark.json` |
| paired 통계 원본 | `artifacts/results/verifier_performance_analysis.json` |
| answer-only risk 결과 | `artifacts/results/svamp_low_cost_verifier.json` |
| C3 split risk 결과 | `artifacts/results/svamp_split_risk_certification.json` |
| figure 생성 코드 | `src/render_paper_figures.py` |

`artifacts/`는 용량과 모델 출력 때문에 Git에서 제외되어 있다. 논문에 사용되는 고정 수치는
추적 가능한 `paper/data/paper_results.json`에 별도로 저장했다.

---

## 17. Overleaf에서 자주 생기는 문제

### `%` 때문에 컴파일 오류

LaTeX 본문에서 퍼센트는 `94.09\%`처럼 쓴다.

### 밑줄 때문에 컴파일 오류

일반 문장에서는 `risk\_certification`처럼 escape하거나 `\texttt{risk\_certification}`을
사용한다.

### 그림이 안 보임

- 파일이 `figures/` 안에 있는지 확인한다.
- 확장자까지 `figures/fig1_architecture.pdf`로 쓴다.
- 파일 이름 대소문자를 정확히 맞춘다.
- `\usepackage{graphicx}`가 로드되었는지 확인한다.

### 표가 column 밖으로 나감

순서대로 해결한다.

1. `\small` 또는 `\scriptsize`
2. `\setlength{\tabcolsep}{3pt}`
3. 열 제목 축약
4. 불필요한 digit 제거
5. 최후에만 `\resizebox{\columnwidth}{!}{...}` 사용

### 참고문헌이 `?`로 표시됨

Overleaf에서 재컴파일하고 `references.bib`의 key와 `\cite{}` key가 같은지 확인한다.

### 한글 글꼴 오류

임의 package를 추가하기 전에 `kcc` 템플릿 자체가 요구하는 compiler와 package를
확인한다. 공개 Overleaf 한글 가이드는 XeLaTeX와 `xeCJK`를 사용할 수 있다고 안내하지만,
학회 class와 충돌할 수 있으므로 공식 템플릿 설정이 우선이다.

---

## 18. 최종 제출 전 숫자 검수표

아래 값이 원고 전체에서 동일해야 한다.

| 항목 | 확정 값 |
|---|---:|
| Lower / Upper | Qwen2.5-1.5B / Qwen2.5-7B |
| GPU | RTX 3090 24GB |
| 고정 threshold | 0.12 / 0.80 |
| Certification / test 크기 | 250 / 300 |
| Answer-only 평균 token | 5.40 (벤치마크 128문항) |
| Full / answer-only latency | 492.97 / 29.15 ms |
| 두 번째 호출 latency 감소 | 94.09% |
| Certification accuracy | 88.00% |
| Test accuracy | 85.67% |
| Certification / test cost | 0.777 / 0.806 |
| Always-Upper 대비 절감 | 22.29% / 19.42% |
| Unsafe | 2.80% / 3.00% |
| 95% unsafe upper bound | 5.195% |
| McNemar p | 0.625 / 0.146 |

`5.40 tokens`는 128문항 latency benchmark 값이고, 전체 certification cache의 평균
`6.54 tokens`와 표본 범위가 다르다. 논문 latency 그림에서는 반드시 5.40을 사용하고,
전체 cache 통계를 별도로 언급할 때만 6.54라고 쓴다.

---

## 19. 제출 직전 체크리스트

### 연구 내용

- [ ] 논문 제목에 safe/certified/guaranteed가 없다.
- [ ] 정책 0.12/0.80이 certification 전에 고정됐다고 썼다.
- [ ] official test를 정책 선택에 사용하지 않았다고 썼다.
- [ ] 5.195% 위험 상한 실패를 결과와 한계에 모두 썼다.
- [ ] “유의차 없음”을 “동등함”으로 과장하지 않았다.
- [ ] 94.09%를 전체 비용이 아니라 두 번째 호출 latency 감소라고 썼다.
- [ ] 전체 비용 절감은 22.29%와 19.42%로 구분했다.

### 형식

- [ ] 해당 연도 KSC 공식 모집 공고의 페이지 제한을 확인했다.
- [ ] 심사본의 저자 익명 여부를 확인했다.
- [ ] `preprint`/publication class 옵션을 공고에 맞췄다.
- [ ] 모든 그림 글자가 100% 확대에서 읽힌다.
- [ ] 표와 그림을 본문에서 번호 순서대로 언급했다.
- [ ] 참고문헌의 저자·연도·venue를 원 논문과 대조했다.
- [ ] PDF에 깨진 한글, overfull box, 빈 페이지가 없다.
- [ ] 제출 PDF 파일명 규칙과 최대 용량을 확인했다.

### 재현성

- [ ] `python -m pytest -q`가 통과한다.
- [ ] figure가 `src.render_paper_figures`로 재생성된다.
- [ ] `paper_results.json`과 본문 표의 숫자가 일치한다.
- [ ] GitHub `main`에 마지막 문서와 figure가 push되어 있다.

---

## 20. 실제 작성 순서

1. Overleaf에서 `kcc` 템플릿 프로젝트를 만든다.
2. 저자·소속·이메일만 먼저 입력한다.
3. 이 문서의 초록, 서론, 방법, 실험, 결과, 결론 초안을 각 `.tex`로 복사한다.
4. `paper/figures`의 PDF 네 개를 업로드한다.
5. 표 1과 통계표를 넣고 먼저 4쪽 버전을 완성한다.
6. `references.bib`를 넣고 모든 citation을 해결한다.
7. KSC 해당 연도 페이지 제한에 맞춰 fig4와 관련 연구부터 압축한다.
8. PDF를 내려받아 숫자 검수표와 체크리스트를 한 줄씩 확인한다.
9. 지도교수/공동저자에게는 “주장 범위”와 “쓰면 안 되는 주장”을 함께 전달한다.
10. 피드백 반영 후 GitHub와 Overleaf 최종본의 figure/data 버전을 맞춘다.

이 순서를 따르면 새 분석을 추가하지 않아도 현재 결과로 일관된 성능 중심 KSC 초안을
완성할 수 있다.
