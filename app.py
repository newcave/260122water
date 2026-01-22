import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

# =========================================================
# Sample PDFs (repo 내 포함)
# =========================================================
SAMPLE_PDFS = {
    "배수지 최적 운영 (기본)": "배수지최적운영.pdf",
    "머신러닝 기반 지방상수도 누수관리": "머신러닝 기반의 지방상수도 관 파손사고 감지 및 누수관리 시스템 개발.pdf",
}

DEFAULT_SAMPLE_KEY = "배수지 최적 운영 (기본)"

# =========================================================
# App Config
# =========================================================
st.set_page_config(
    page_title="K-water 수도관리 AI 봇 (요약 · 예측 · 운영보조)",
    page_icon="💧",
    layout="wide",
)

# --- Sidebar Hide ---
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
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
# System Prompt
# =========================================================
SYSTEM_PROMPT = """
당신은 K-water 상하수도 분야를 지원하는 수도관리 AI 봇이다.

[원칙]
- 의사결정 보조자이며 최종 결정자는 인간
- 모든 제안은 근거와 불확실성을 명시
- 단정적 표현 금지

[응답 구조]
근거 → 해석 → 제안 → 리스크 → 추가 확인사항
"""

# =========================================================
# Utility Functions
# =========================================================
def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(file_obj) -> str:
    reader = PdfReader(file_obj)
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
- 수치/지표/운영 중심 요약
- 실무/연구 관점 포함
- 10~12줄 이내

[원문]
{ch}
"""
        chunk_summaries.append(call_llm(prompt, model))

    merged = call_llm(
        f"""다음 청크 요약을 통합 요약하라 (800~1200자).

{chr(10).join(chunk_summaries)}
""",
        model,
    )

    key_points = call_llm(
        f"""다음 요약을 실무자용 브리프로 재작성하라.

[요약]
{merged}
""",
        model,
    )

    glossary = call_llm(
        f"""다음 요약에서 핵심 용어 20개 내외 용어집 작성.

[요약]
{merged}
""",
        model,
    )

    return SummaryResult(merged, key_points, glossary)

# =========================================================
# UI
# =========================================================
st.title("💧 K-water 수도관리 AI 봇 (샘플), @'25.01.22 수도관리처-연구원 워크숍")

tab1, tab2 = st.tabs(["1️⃣ 보고서 요약", "2️⃣ 수도관리 봇 초안"])

# ---------------------------
# TAB 1: Summary
# ---------------------------
with tab1:
    st.subheader("보고서 선택")

    use_sample = st.checkbox("📄 샘플 보고서 사용", value=True)

    uploaded = None
    sample_loaded = False
    selected_sample_name = None

    if use_sample:
        selected_sample_name = st.radio(
            "샘플 보고서 선택",
            list(SAMPLE_PDFS.keys()),
            index=list(SAMPLE_PDFS.keys()).index(DEFAULT_SAMPLE_KEY),
        )

        selected_path = SAMPLE_PDFS[selected_sample_name]

        if os.path.exists(selected_path):
            uploaded = selected_path
            sample_loaded = True
            st.success(f"선택된 샘플: {selected_sample_name}")
        else:
            st.error(f"샘플 PDF 파일을 찾을 수 없습니다: {selected_path}")
    else:
        uploaded = st.file_uploader("PDF 업로드", type=["pdf"])

    if uploaded and st.button("요약 생성"):
        with st.spinner("요약 생성 중..."):
            if sample_loaded:
                with open(uploaded, "rb") as f:
                    raw_text = extract_pdf_text(f)
            else:
                raw_text = extract_pdf_text(uploaded)

            st.session_state.summary = summarize_report(raw_text, DEFAULT_MODEL)
            st.session_state.sample_name = selected_sample_name

    if "summary" in st.session_state:
        s = st.session_state.summary
        if "sample_name" in st.session_state and st.session_state.sample_name:
            st.caption(f"📘 사용된 보고서: {st.session_state.sample_name}")

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
    if "summary" not in st.session_state:
        st.warning("먼저 보고서를 요약하세요.")
    else:
        if st.button("수도관리 봇 초안 생성"):
            with st.spinner("초안 생성 중..."):
                draft = call_llm(
                    f"""{SYSTEM_PROMPT}

이 보고서는 '{st.session_state.get("sample_name", "사용자 업로드")}' 기반이다.
이를 바탕으로 K-water 수도관리 AI 봇 기획 초안을 작성하라.

[보고서 요약]
{st.session_state.summary.merged}
""",
                    DEFAULT_MODEL,
                )
                st.session_state.bot_draft = draft

        if "bot_draft" in st.session_state:
            st.subheader("수도관리 봇 기획 초안")
            st.write(st.session_state.bot_draft)
