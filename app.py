import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

# =========================================================
# App Config
# =========================================================
st.set_page_config(
    page_title="K-water 수도관리 AI 봇 (요약 · 예측 · 운영보조) 26.01.22 4pm",
    page_icon="💧",
    layout="wide",
)

# =========================================================
# OpenAI Client
# =========================================================
def init_openai() -> Tuple[OpenAI, str]:
    api_key = st.secrets.get("OPENAI_API_KEY")
    model = st.secrets.get("OPENAI_MODEL", "gpt-5.2")

    if not api_key:
        st.error("OPENAI_API_KEY가 설정되지 않았습니다 (Streamlit Secrets 확인).")
        st.stop()

    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI(), model


client, DEFAULT_MODEL = init_openai()

# =========================================================
# System Prompt (수도관리 봇 핵심)
# =========================================================
SYSTEM_PROMPT = """
당신은 K-water 상하수도 분야를 지원하는 수도관리 AI 봇이다.

[목적]
1) 연구자를 위한 공정 AI 하이브리드 예측 지원
2) 현장 자료 수집을 통한 모형 성능 향상
3) 수도 운영 의사결정의 안전한 보조

[원칙]
- 의사결정 보조자이며 최종 결정자는 인간이다.
- 모든 운영 조치는 근거와 불확실성을 명시한다.
- 현장 추가 업무를 최소화하는 방향을 우선한다.
- 단정적 표현을 금지한다.

[응답 구조]
근거 → 해석 → 제안 → 리스크 → 추가 확인사항
"""

# =========================================================
# Tool Schemas (B안)
# =========================================================
TOOLS = [
    {
        "name": "query_document",
        "description": "K-water 보고서/운영기준/매뉴얼 검색",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "doc_type": {
                    "type": "string",
                    "enum": ["report", "manual", "regulation"]
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "analyze_timeseries",
        "description": "수질·수요·에너지 시계열 예측",
        "parameters": {
            "type": "object",
            "properties": {
                "target_variable": {"type": "string"},
                "time_horizon": {"type": "string"},
                "model_type": {
                    "type": "string",
                    "enum": ["physical", "ml", "hybrid"]
                }
            },
            "required": ["target_variable", "time_horizon"]
        }
    },
    {
        "name": "diagnose_anomaly",
        "description": "이상 원인 후보 진단",
        "parameters": {
            "type": "object",
            "properties": {
                "symptom": {"type": "string"},
                "location": {"type": "string"}
            },
            "required": ["symptom"]
        }
    },
    {
        "name": "recommend_action",
        "description": "SOP 기반 운영 조치안 제안(보조)",
        "parameters": {
            "type": "object",
            "properties": {
                "issue": {"type": "string"},
                "urgency": {
                    "type": "string",
                    "enum": ["monitor", "check", "urgent"]
                },
                "human_approval_required": {"type": "boolean"}
            },
            "required": ["issue"]
        }
    },
    {
        "name": "collect_field_feedback",
        "description": "현장 관찰·조치 결과 수집",
        "parameters": {
            "type": "object",
            "properties": {
                "observation": {"type": "string"},
                "action_taken": {"type": "string"},
                "outcome": {"type": "string"}
            },
            "required": ["observation"]
        }
    }
]

# =========================================================
# Utility Functions
# =========================================================
def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            pages.append(f"[page {i+1}]\n{txt}")
    return normalize_text("\n\n".join(pages))


def chunk_text(text: str, size: int = 8000, overlap: int = 800) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def call_llm(prompt: str, model: str) -> str:
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.2,
    )
    return (resp.output_text or "").strip()


# =========================================================
# Summarization Logic
# =========================================================
@dataclass
class SummaryResult:
    merged: str
    key_points: str
    glossary: str


def summarize_report(text: str, model: str) -> SummaryResult:
    chunks = chunk_text(text)
    chunk_summaries = []

    for i, ch in enumerate(chunks, 1):
        prompt = f"""
다음은 K-water 상하수도 보고서 일부이다 (chunk {i}/{len(chunks)}).
- 수치/지표/공정 중심 요약
- 연구 및 운영 관점 포함
- 10~12줄 이내

[원문]
{ch}
"""
        chunk_summaries.append(call_llm(prompt, model))

    merge_prompt = f"""
다음은 보고서 청크 요약이다.
이를 바탕으로 통합 요약을 작성하라.

[출력]
1) 통합 요약 (800~1200자)
2) 연구·운영 시사점 TOP 7
3) 데이터/변수 후보 목록

[청크 요약]
{chr(10).join(chunk_summaries)}
"""
    merged = call_llm(merge_prompt, model)

    key_prompt = f"""
다음 요약을 기반으로 실무자용 브리프 작성:

- 한 페이지 요약
- 즉시 실험 가능한 아이디어 5개

[요약]
{merged}
"""
    key_points = call_llm(key_prompt, model)

    glossary_prompt = f"""
다음 요약에서 핵심 용어 20개 내외 용어집 작성:

[요약]
{merged}
"""
    glossary = call_llm(glossary_prompt, model)

    return SummaryResult(merged, key_points, glossary)


# =========================================================
# Bot Draft Generation
# =========================================================
def generate_bot_draft(summary: str, model: str) -> str:
    prompt = f"""
{SYSTEM_PROMPT}

다음 보고서 요약을 근거로
"K-water 수도관리 AI 봇" 기획 초안을 작성하라.

[필수 포함]
- 공정 AI 하이브리드 예측 구조
- 현장 자료 수집 → 성능 향상 루프
- 사용자 유형별 기능
- 안전장치 및 휴먼 인 더 루프
- 8주 구축 로드맵

[보고서 요약]
{summary}
"""
    return call_llm(prompt, model)


# =========================================================
# UI
# =========================================================
st.title("💧 K-water 수도관리 AI 봇")

with st.sidebar:
    st.header("설정")
    model = st.text_input("모델", DEFAULT_MODEL)

tab1, tab2, tab3 = st.tabs(
    ["1️⃣ 보고서 요약", "2️⃣ 수도관리 봇 초안", "3️⃣ 수도관리 챗봇"]
)

# Session State
if "summary" not in st.session_state:
    st.session_state.summary = None
if "bot_draft" not in st.session_state:
    st.session_state.bot_draft = ""
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# TAB 1: Summary
# ---------------------------
with tab1:
    uploaded = st.file_uploader("K-water 상하수도 PDF 업로드", type=["pdf"])
    if uploaded and st.button("요약 생성"):
        with st.spinner("요약 생성 중..."):
            raw_text = extract_pdf_text(uploaded)
            st.session_state.summary = summarize_report(raw_text, model)

    if st.session_state.summary:
        s = st.session_state.summary
        st.subheader("통합 요약")
        st.write(s.merged)
        st.subheader("실무 브리프")
        st.write(s.key_points)
        st.subheader("용어집")
        st.write(s.glossary)

# ---------------------------
# TAB 2: Bot Draft
# ---------------------------
with tab2:
    if not st.session_state.summary:
        st.warning("먼저 보고서를 요약하세요.")
    else:
        if st.button("수도관리 봇 초안 생성"):
            with st.spinner("초안 생성 중..."):
                st.session_state.bot_draft = generate_bot_draft(
                    st.session_state.summary.merged, model
                )

        if st.session_state.bot_draft:
            st.subheader("수도관리 봇 기획 초안")
            st.write(st.session_state.bot_draft)

# ---------------------------
# TAB 3: Chatbot (Mock Tool Mode)
# ---------------------------
with tab3:
    st.caption("⚠️ 현재는 Tool 호출을 '설계 수준'으로만 시뮬레이션")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("수도 운영 / 예측 / 이상 진단 질문 입력")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("응답 생성 중..."):
            prompt = f"""
{SYSTEM_PROMPT}

[대화 맥락]
{st.session_state.messages}

[사용자 질문]
{user_input}
"""
            answer = call_llm(prompt, model)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.write(answer)

