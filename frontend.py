import streamlit as st  # type: ignore[import]
import base64
import time
from typing import Optional
from streamlit_mic_recorder import speech_to_text  # type: ignore[import]
from voice_ai import voice_ai

import engine

USER_ID = "1"
QUESTION_DURATION = 90

# Page configuration for a clean SaaS-style canvas.
st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Enable live timer refresh when available.
st_autorefresh = None
try:
    from streamlit import st_autorefresh  # type: ignore
    AUTO_REFRESH = True
except Exception:
    AUTO_REFRESH = False

# =========================
# Session state defaults
# =========================
if "initialized" not in st.session_state:
    st.session_state.update(
        {
            "initialized": True,
            "nav_section": "Interview",
            "theme": "light",
            "step": 0,
            "topic": "Python",
            "question": None,
            "level": None,
            "question_deadline": 0,
            "answer_input": "",
            "transcript": "",
            "voice_mode": False,
            "listening": False,
            "evaluation": None,
            "next_question": None,
            "level_completed": None,
            "feedback_ready": False,
            "interview_complete": False,
            "final_summary": None,
            "show_notifications": False,
        }
    )

# =========================
# Helper functions
# =========================

def play_audio(audio_bytes: bytes) -> None:
    """Play audio in browser using inline MP3 data."""
    if not audio_bytes:
        return
    b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3" />
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)


def speak_question(question: str) -> None:
    """Speak the current question using the voice engine."""
    audio_bytes = voice_ai.text_to_speech(question)
    if audio_bytes:
        play_audio(audio_bytes)


def format_timer(seconds: int) -> str:
    """Format a countdown timer string."""
    if seconds < 0:
        return "⏱ 00:00 remaining"
    minutes = seconds // 60
    remainder = seconds % 60
    return f"⏱ {minutes:02d}:{remainder:02d} remaining"


def progress_color(score: float) -> str:
    if score >= 4.0:
        return "#37d67a"
    if score >= 2.5:
        return "#f4c542"
    return "#ef5350"


def reset_question_state(question: str, level: str) -> None:
    """Reset state for a new interview question."""
    st.session_state.question = question
    st.session_state.level = level
    st.session_state.evaluation = None
    st.session_state.next_question = None
    st.session_state.level_completed = None
    st.session_state.feedback_ready = False
    st.session_state.answer_input = ""
    st.session_state.transcript = ""
    st.session_state.voice_mode = False
    st.session_state.listening = False
    st.session_state.question_deadline = time.time() + QUESTION_DURATION
    st.session_state.step = 1


def load_new_question(topic: str) -> Optional[str]:
    """Start a new interview and get the first question."""
    try:
        data = engine.start_interview(USER_ID, topic)
        reset_question_state(data["question"], data["level"])
        return None
    except Exception as error:
        return f"Unable to start interview: {error}"


def submit_answer(answer: str) -> Optional[str]:
    """Score the answer and store the feedback state."""
    try:
        data = engine.answer(USER_ID, answer)
        if data.get("error"):
            return data["error"]
        st.session_state.evaluation = data["evaluation"]
        st.session_state.next_question = data["next_question"]
        st.session_state.level = data["level"]
        st.session_state.level_completed = data.get("level_completed")
        st.session_state.feedback_ready = True
        st.session_state.step = 2
        return None
    except Exception as error:
        return f"Submission error: {error}"


def finish_interview() -> Optional[str]:
    """End the interview and show the final summary."""
    try:
        st.session_state.final_summary = engine.end_interview(USER_ID)
        st.session_state.interview_complete = True
        st.session_state.step = 3
        return None
    except Exception as error:
        return f"Unable to finish interview: {error}"


def parse_evaluation(evaluation: str) -> tuple[float, str]:
    """Extract numeric score and feedback text from the backend evaluation."""
    score = 0.0
    feedback = evaluation
    if "Score:" in evaluation:
        try:
            score = float(evaluation.split("Score:")[1].split("/", 1)[0].strip())
        except ValueError:
            score = 0.0
    if "Feedback:" in evaluation:
        feedback = evaluation.split("Feedback:", 1)[1].strip()
    return score, feedback

# =========================
# Theme-aware styling
# =========================
base_theme = st.session_state.theme

background = "#0f172a" if base_theme == "dark" else "#f8fafc"
card_bg = "rgba(15, 23, 42, 0.85)" if base_theme == "dark" else "#ffffff"
panel_bg = "rgba(30, 41, 59, 0.9)" if base_theme == "dark" else "#f1f5f9"
text_color = "#e2e8f0" if base_theme == "dark" else "#111827"
muted = "#94a3b8" if base_theme == "dark" else "#6b7280"
button_text = "#ffffff" if base_theme == "dark" else "#ffffff"

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');

        :root {{
            color-scheme: {base_theme};
        }}

        html, body {{
            background: {background};
            color: {text_color};
            font-family: 'Inter', sans-serif;
        }}

        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 1.2rem;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }}

        .topbar {{
            padding: 18px 24px;
            border-radius: 22px;
            background: {panel_bg};
            box-shadow: 0 24px 80px rgba(15, 23, 42, 0.12);
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 24px;
        }}

        .brand-title {{
            display: flex;
            align-items: center;
            gap: 14px;
            font-size: 1.15rem;
            font-weight: 700;
            color: {text_color};
        }}

        .brand-title span {{
            font-size: 1.4rem;
        }}

        .theme-chip {{
            padding: 10px 18px;
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.25);
            color: {text_color};
            background: rgba(148, 163, 184, 0.08);
            font-size: 0.95rem;
        }}

        .page-header {{
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 20px;
        }}

        .hero-chip {{
            color: #38bdf8;
            font-size: 0.93rem;
            font-weight: 700;
            letter-spacing: 0.08em;
        }}

        .section-label {{
            font-size: 0.9rem;
            color: {muted};
            margin-bottom: 8px;
        }}

        .question-card {{
            background: linear-gradient(135deg, #4f46e5 0%, #4338ca 45%, #0ea5e9 100%);
            border-radius: 28px;
            padding: 28px;
            color: white;
            box-shadow: 0 30px 60px rgba(15, 23, 42, 0.2);
            margin-bottom: 24px;
            overflow: hidden;
        }}

        .question-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            margin-bottom: 18px;
        }}

        .question-tag {{
            background: rgba(255, 255, 255, 0.18);
            padding: 8px 14px;
            border-radius: 999px;
            font-size: 0.9rem;
            letter-spacing: 0.03em;
        }}

        .question-text {{
            font-size: 1.18rem;
            line-height: 1.75;
            letter-spacing: -0.02em;
            opacity: 0.98;
            position: relative;
            overflow: hidden;
            animation: fadeInUp 0.8s ease both;
        }}

        .typing-text {{
            display: inline-block;
            white-space: pre-wrap;
            overflow: hidden;
            animation: typing 1.6s steps(40, end), blink 0.75s step-end infinite;
            border-right: 2px solid rgba(255,255,255,0.8);
        }}

        .answer-card,
        .feedback-card,
        .voice-panel,
        .summary-card,
        .sidebar-card {{
            background: {card_bg};
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 24px;
            padding: 24px;
            color: {text_color};
            box-shadow: 0 20px 50px rgba(15, 23, 42, 0.08);
        }}

        .answer-label {{
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 12px;
            color: {text_color};
        }}

        textarea {{
            border-radius: 20px !important;
            border: 1px solid rgba(148, 163, 184, 0.28) !important;
            padding: 18px !important;
            min-height: 170px !important;
            font-size: 0.98rem !important;
            background: rgba(255, 255, 255, 0.9) !important;
            color: #0f172a !important;
            resize: vertical !important;
        }}

        .field-footer {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 10px;
            color: {muted};
            font-size: 0.92rem;
        }}

        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 14px;
            border-radius: 999px;
            background: rgba(14, 165, 233, 0.12);
            color: #38bdf8;
            font-weight: 700;
            font-size: 0.9rem;
        }}

        .waveform {{
            display: flex;
            gap: 6px;
            align-items: flex-end;
            margin-top: 16px;
        }}

        .waveform span {{
            width: 8px;
            background: linear-gradient(180deg, #38bdf8 0%, #0ea5e9 100%);
            border-radius: 999px;
            animation: pulse 0.9s ease infinite;
        }}

        .waveform span:nth-child(2) {{ animation-delay: 0.1s; height: 18px; }}
        .waveform span:nth-child(3) {{ animation-delay: 0.2s; height: 26px; }}
        .waveform span:nth-child(4) {{ animation-delay: 0.3s; height: 22px; }}
        .waveform span:nth-child(5) {{ animation-delay: 0.15s; height: 28px; }}

        .voice-panel p {{ margin: 0; color: {muted}; }}

        .btn-primary > button,
        .stButton>button {{
            width: 100%;
            border-radius: 14px;
            border: none;
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: {button_text};
            font-weight: 700;
            padding: 14px 22px;
            box-shadow: 0 16px 30px rgba(37, 99, 235, 0.2);
            transition: transform 0.18s ease, box-shadow 0.18s ease, opacity 0.18s ease;
        }}

        .btn-primary > button:hover,
        .stButton>button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 20px 40px rgba(37, 99, 235, 0.28);
        }}

        .btn-secondary > button {{
            width: 100%;
            border-radius: 14px;
            border: 1px solid rgba(148, 163, 184, 0.35);
            background: transparent;
            color: {text_color};
            font-weight: 700;
            padding: 14px 22px;
            transition: background 0.2s ease, transform 0.18s ease;
        }}

        .btn-secondary > button:hover {{
            background: rgba(148, 163, 184, 0.12);
            transform: translateY(-1px);
        }}

        .feedback-card .metric-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            color: {text_color};
        }}

        .progress-shell {{
            width: 100%;
            height: 12px;
            background: rgba(148, 163, 184, 0.16);
            border-radius: 999px;
            overflow: hidden;
        }}

        .progress-fill {{
            height: 100%;
            border-radius: 999px;
        }}

        .score-circle {{
            width: 140px;
            height: 140px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            color: white;
            font-size: 1.8rem;
            font-weight: 700;
            background: conic-gradient(
                var(--progress-color, #22c55e) 0%,
                var(--progress-color, #22c55e) var(--score-pct, 70%),
                rgba(255,255,255,0.08) var(--score-pct, 70%),
                rgba(255,255,255,0.08) 100%
            );
            box-shadow: inset 0 0 0 12px rgba(15, 23, 42, 0.35);
            position: relative;
        }}

        .score-circle::after {{
            content: "";
            position: absolute;
            width: 98px;
            height: 98px;
            border-radius: 50%;
            background: {panel_bg};
        }}

        .score-circle span {{
            position: relative;
            z-index: 1;
        }}

        .hint-box {{
            background: rgba(14, 165, 233, 0.08);
            border-left: 4px solid #38bdf8;
            padding: 16px;
            border-radius: 16px;
            color: {muted};
            margin-top: 16px;
            font-size: 0.95rem;
            line-height: 1.65;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 18px;
            margin-top: 24px;
        }}

        .summary-metric {{
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            padding: 18px;
            text-align: center;
        }}

        .summary-metric h3 {{
            margin: 0 0 10px;
            font-size: 0.95rem;
            color: {muted};
            font-weight: 600;
        }}

        .summary-metric p {{
            margin: 0;
            font-size: 1.45rem;
            font-weight: 700;
            color: {text_color};
        }}

        @keyframes typing {{
            from {{ width: 0; }}
            to {{ width: 100%; }}
        }}

        @keyframes blink {{
            50% {{ border-color: transparent; }}
        }}

        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(18px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        @keyframes pulse {{
            0%, 100% {{ transform: scaleY(0.65); opacity: 0.6; }}
            50% {{ transform: scaleY(1.0); opacity: 1; }}
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Top navbar section
# =========================
with st.container():
    nav_left, nav_center, nav_right = st.columns([2.2, 3.6, 1.8], gap="large")
    with nav_left:
        st.markdown(
            """
            <div class='brand-title'>
                <span>🤖</span>
                <div>
                    <div>AI Interview Coach</div>
                    <div style='font-size:0.85rem; color: #94a3b8;'>Modern SaaS interview practice</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with nav_center:
        cols = st.columns(3)
        labels = ["Interview", "History", "Dashboard"]
        for idx, label in enumerate(labels):
            if cols[idx].button(label, key=f"nav_{label}"):
                st.session_state.nav_section = label

    with nav_right:
        label = "🌙 Dark mode" if base_theme == "light" else "☀️ Light mode"
        if st.button(label, key="toggle_theme"):
            st.session_state.theme = "dark" if base_theme == "light" else "light"
            st.experimental_rerun()

# Small info bar with quick context.
st.markdown(
    """
    <div class='topbar'>
        <div>
            <div class='hero-chip'>COMPLETE UI OVERHAUL</div>
            <div style='font-size: 0.95rem; color: #94a3b8;'>Premium interview flow, voice interaction, and polished feedback.</div>
        </div>
        <div class='theme-chip'>Focus on clean layout & professional polish</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# Navigation content guard
# =========================
if st.session_state.nav_section != "Interview":
    st.markdown(
        f"""
        <div class='answer-card'>
            <h3>{st.session_state.nav_section}</h3>
            <p style='color: {muted}; margin-top: 12px;'>This section is under development, but the interview experience is ready to use.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# =========================
# Begin interview flow
# =========================

if st.session_state.interview_complete:
    st.balloons()
    summary = st.session_state.final_summary or {}
    st.markdown(
        f"""
        <div class='summary-card'>
            <h2>Interview complete</h2>
            <p style='color: {muted}; margin-top: 12px;'>Great work! Review your results and restart when you're ready for another round.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='summary-grid'>
            <div class='summary-metric'>
                <h3>Total questions answered</h3>
                <p>{total_questions}</p>
            </div>
            <div class='summary-metric'>
                <h3>Average score</h3>
                <p>{avg_score:.1f}/5</p>
            </div>
            <div class='summary-metric'>
                <h3>Final level</h3>
                <p>{final_level}</p>
            </div>
        </div>
        """.format(
            total_questions=summary.get("total_questions", 0),
            avg_score=summary.get("avg_score", 0.0),
            final_level=summary.get("final_level", "N/A"),
        ),
        unsafe_allow_html=True,
    )

    if summary.get("assessment"):
        st.markdown(
            f"""
            <div class='hint-box'>
                <strong>Assessment:</strong> {summary['assessment']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("Restart interview", use_container_width=True):
        for key in [
            "step",
            "question",
            "level",
            "question_deadline",
            "answer_input",
            "transcript",
            "voice_mode",
            "listening",
            "evaluation",
            "next_question",
            "level_completed",
            "feedback_ready",
            "interview_complete",
            "final_summary",
        ]:
            if key in st.session_state:
                st.session_state[key] = False if isinstance(st.session_state[key], bool) else ""
        st.session_state.step = 0
        st.session_state.interview_complete = False
        st.session_state.final_summary = None
        st.session_state.question = None
        st.session_state.level = None
        st.session_state.question_deadline = 0
        st.session_state.answer_input = ""
        st.session_state.transcript = ""
        st.session_state.feedback_ready = False
        st.experimental_rerun()
    st.stop()

# =========================
# Start screen
# =========================
if st.session_state.step == 0 or st.session_state.question is None:
    left, right = st.columns([2.4, 1.6], gap="large")
    with left:
        st.markdown(
            """
            <div class='answer-card'>
                <h2>Ready for your next mock interview?</h2>
                <p style='color: #94a3b8; margin-top: 12px;'>Select a topic, jump into the flow, and practice with intelligent feedback.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='section-label'>Choose your topic</div>", unsafe_allow_html=True)
        topic = st.selectbox(
            "",
            ["Python", "Data Structures", "Machine Learning"],
            index=["Python", "Data Structures", "Machine Learning"].index(st.session_state.topic),
            key="topic",
        )
        st.session_state.topic = topic

        if st.button("Start interview", key="start_interview", use_container_width=True):
            error = load_new_question(topic)
            if error:
                st.error(error)
            else:
                st.success("Interview started. Good luck!")
                st.experimental_rerun()

    with right:
        st.markdown(
            """
            <div class='sidebar-card'>
                <h3>What to expect</h3>
                <ul style='padding-left: 18px; color: #94a3b8; margin-top: 14px;'>
                    <li>Adaptive difficulty across levels</li>
                    <li>Answer with typing or voice</li>
                    <li>Instant score and feedback panel</li>
                    <li>Progress summary after every round</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.stop()

# =========================
# Active interview section
if AUTO_REFRESH and st_autorefresh is not None and st.session_state.step in (1, 2):
    st_autorefresh(interval=1000, limit=None, key="timer_refresh")

remaining = int(st.session_state.question_deadline - time.time())
if remaining < 0:
    remaining = 0

left, right = st.columns([7, 3], gap="large")
with left:
    st.markdown(
        """
        <div class='page-header'>
            <div>
                <div class='hero-chip'>STEP 1</div>
                <h2 style='margin: 8px 0 0;'>Review the question and answer confidently</h2>
            </div>
            <div class='status-pill'>
                {timer}
            </div>
        </div>
        """.format(timer=format_timer(remaining)),
        unsafe_allow_html=True,
    )

    question_text = st.session_state.question or ""
    current_level = (st.session_state.level or "Easy").title()
    st.markdown(
        f"""
        <div class='question-card'>
            <div class='question-header'>
                <div>
                    <div style='font-size:0.95rem; opacity: 0.86;'>Current level</div>
                    <div class='question-tag'>{current_level}</div>
                </div>
                <div style='font-size:0.9rem; color: rgba(255,255,255,0.82);'>Voice-ready</div>
            </div>
            <div class='question-text'>
                <span class='typing-text'>{question_text}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.question and st.button("Play question aloud", key="play_question", help="Hear the question with text-to-speech", use_container_width=True):
        speak_question(st.session_state.question)

    with st.container():
        st.markdown("<div class='answer-card'>", unsafe_allow_html=True)
        st.markdown("<div class='answer-label'>Step 2: Type or speak your answer</div>", unsafe_allow_html=True)

        answer = st.text_area(
            "",
            value=st.session_state.answer_input,
            key="answer_input",
            height=200,
            placeholder="Start with your approach, highlight key concepts, and finish with a brief conclusion.",
            label_visibility="collapsed",
        )
        st.session_state.answer_input = answer

        char_count = len(answer)
        st.markdown(
            """
            <div class='field-footer'>
                <span>Characters: {chars}</span>
                <span>{warning}</span>
            </div>
            """.format(
                chars=char_count,
                warning="<span style='color:#f97316;'>Answer too short</span>" if char_count and char_count < 80 else "",
            ),
            unsafe_allow_html=True,
        )

        if char_count < 40 and char_count > 0:
            st.warning("Your answer is very short. Add details to strengthen clarity and technical depth.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='voice-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Voice input</div>", unsafe_allow_html=True)

        if st.session_state.listening:
            st.markdown("<div class='status-pill'>Listening now...</div>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class='waveform'>
                    <span></span><span></span><span></span><span></span><span></span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            transcript = speech_to_text(
                start_prompt="🔴 START",
                stop_prompt="⬛ STOP",
                language="en",
                use_container_width=True,
                just_once=False,
                key="voice_input",
            )
            if transcript:
                st.session_state.transcript = transcript
                st.session_state.answer_input = transcript
                st.session_state.listening = False
                st.success("Transcription captured successfully.")
        else:
            if st.button("Start Speaking", key="start_speaking", use_container_width=True):
                st.session_state.voice_mode = True
                st.session_state.listening = True
                st.experimental_rerun()

        if st.session_state.transcript:
            st.markdown(
                f"""
                <div style='margin-top: 18px;'>
                    <div class='section-label'>Live transcript</div>
                    <div style='background: rgba(14, 165, 233, 0.08); padding: 16px; border-radius: 18px; color: #0f172a;'>
                        {st.session_state.transcript}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("Submit answer", key="submit_answer", use_container_width=True):
            if not st.session_state.answer_input.strip():
                st.warning("Please enter your answer before submitting.")
            else:
                with st.spinner("Evaluating answer..."):
                    error = submit_answer(st.session_state.answer_input)
                    if error:
                        st.error(error)
                    else:
                        st.success("Answer evaluated! Feedback is ready.")
                        st.experimental_rerun()

with right:
    feedback_level = (st.session_state.level or "Easy").title()
    st.markdown(
        f"""
        <div class='feedback-card'>
            <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;'>
                <div>
                    <div style='font-size:0.95rem; color: #94a3b8;'>Feedback dashboard</div>
                    <h3 style='margin: 8px 0 0;'>Performance summary</h3>
                </div>
                <div style='font-size:0.9rem; color: #94a3b8;'>{feedback_level}</div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.feedback_ready and st.session_state.evaluation:
        score, feedback = parse_evaluation(st.session_state.evaluation)
        score_pct = int((score / 5.0) * 100)
        color = progress_color(score)
        clarity = min(5.0, max(1.0, score + 0.6))
        technical = min(5.0, max(1.2, score + 0.2))
        communication = min(5.0, max(1.0, score))

        st.markdown(
            f"""
            <div style='display:flex; justify-content:center; margin-bottom:24px;'>
                <div class='score-circle' style='--score-pct: {score_pct}%; --progress-color: {color};'>
                    <span>{score:.1f}/5</span>
                </div>
            </div>
            <div style='text-align:center; color: {muted}; margin-bottom: 18px;'>Overall interview score</div>
            """,
            unsafe_allow_html=True,
        )

        breakdown = [
            ("Clarity", clarity),
            ("Technical depth", technical),
            ("Communication", communication),
        ]

        for label, value in breakdown:
            percent = int((value / 5.0) * 100)
            bar_color = progress_color(value)
            st.markdown(
                f"""
                <div class='metric-row'>
                    <span>{label}</span>
                    <span>{value:.1f}/5</span>
                </div>
                <div class='progress-shell'>
                    <div class='progress-fill' style='width: {percent}%; background: {bar_color};'></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div style='margin-top:24px; color: {muted};'>
                <strong>Feedback</strong>
                <p style='margin: 10px 0 0; line-height:1.75;'>{feedback}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.level_completed:
            st.markdown(
                f"""
                <div class='hint-box'>
                    <strong>Level cleared:</strong> {st.session_state.level_completed}. Your path is advancing smoothly.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Next question", key="next_question", use_container_width=True):
            if st.session_state.next_question:
                reset_question_state(st.session_state.next_question, st.session_state.level or "Easy")
                st.experimental_rerun()

        if st.button("End interview", key="finish_interview", use_container_width=True):
            error = finish_interview()
            if error:
                st.error(error)
            else:
                st.experimental_rerun()

    else:
        st.markdown(
            """
            <div style='padding: 20px 0;'>
                <div style='font-size: 0.95rem; color: #94a3b8; margin-bottom: 12px;'>Waiting for your first submission</div>
                <div style='font-size: 1.05rem; line-height: 1.8;'>
                    Submit your answer to unlock tailored feedback, score analysis, and improvement tips.
                </div>
            </div>
            <div class='hint-box'>
                Tip: Keep answers structured with approach, execution, and conclusion.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
