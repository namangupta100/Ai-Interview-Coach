import streamlit as st
import requests
import base64
from streamlit_mic_recorder import speech_to_text  # type: ignore
from voice_ai import voice_ai

API_URL = "http://127.0.0.1:8000"

# Wide layout for less scrolling
st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for compact design
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    .stButton>button {width: 100%;}
    .question-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-size: 18px;
        margin: 10px 0;
    }
    .score-card {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .level-badge {
        background: #00d4aa;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# =========================
# 🔹 VOICE HELPER FUNCTIONS
# =========================
def play_audio(audio_bytes):
    """Play audio in browser using HTML audio tag."""
    if audio_bytes:
        b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f'''
            <audio autoplay>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        '''
        st.markdown(audio_html, unsafe_allow_html=True)


def speak_question(question: str):
    """Convert question to speech and play it."""
    audio_bytes = voice_ai.text_to_speech(question)
    if audio_bytes:
        play_audio(audio_bytes)
        return True
    return False


def transcribe_audio(audio_bytes) -> str:
    """Convert recorded audio to text."""
    if audio_bytes:
        return voice_ai.speech_to_text_from_audio(audio_bytes)
    return ""


# =========================
# 🔹 SESSION INIT (must be before any UI elements that use session state)
# =========================
if "question" not in st.session_state:
    st.session_state.question = None
    st.session_state.level = None
    st.session_state.evaluation = None
    st.session_state.next_question = None
    st.session_state.level_completed = None
    st.session_state.show_level_popup = False
    st.session_state.interview_complete = False
    st.session_state.final_summary = None
    st.session_state.voice_mode = False
    st.session_state.transcribed_answer = ""
    st.session_state.auto_speak = True
    st.session_state.prev_q_hash = None
    st.session_state.last_spoken_q = None


# =========================
# 🔹 HEADER BAR
# =========================
st.title("🤖 AI Interview Coach")

# Voice Settings Box - Always visible
st.markdown("### 🎙️ Voice Settings")
voice_col1, voice_col2 = st.columns(2)
with voice_col1:
    voice_on = st.checkbox("🎤 Enable Voice Mode", value=st.session_state.get("voice_mode", False), key="voice_check")
    st.session_state.voice_mode = voice_on
with voice_col2:
    if st.session_state.voice_mode:
        auto_speak = st.checkbox("🔊 Auto-Speak Questions", value=st.session_state.get("auto_speak", True), key="auto_speak_check")
        st.session_state.auto_speak = auto_speak

if st.session_state.voice_mode:
    st.success("✅ Voice Mode is ON! Questions will be spoken and you can record answers.")
else:
    st.info("💡 Enable Voice Mode to use speech features")

st.divider()


# =========================
# 🔹 INTERVIEW COMPLETE
# =========================
if st.session_state.interview_complete:
    st.balloons()
    
    if st.session_state.final_summary:
        summary = st.session_state.final_summary
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📝 Questions", summary['total_questions'])
        with col2:
            st.metric("⭐ Avg Score", f"{summary['avg_score']:.1f}/5")
        with col3:
            st.metric("🎯 Level", summary['final_level'].title())
        
        st.success(f"🎉 **Assessment:** {summary.get('assessment', 'Great effort!')}")
    
    if st.button("🔁 Start New Interview", use_container_width=True):
        st.session_state.clear()
        st.rerun()


# =========================
# 🔹 START SCREEN
# =========================
elif not st.session_state.question:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🎯 Ready to practice?")
        st.markdown("Select a topic and let's begin!")
        
        topic = st.selectbox("📚 Topic", ["Python", "Data Structures", "Machine Learning"])
        
        if st.button("🚀 Start Interview", use_container_width=True, type="primary"):
            res = requests.get(f"{API_URL}/start-interview", params={"user_id": "1", "topic": topic})
            data = res.json()
            st.session_state.question = data["question"]
            st.session_state.level = data["level"]
            st.rerun()


# =========================
# 🔹 INTERVIEW IN PROGRESS
# =========================
else:
    # Level popup
    if st.session_state.show_level_popup and st.session_state.level_completed:
        st.balloons()
        st.success(f"🎉 {st.session_state.level_completed} Completed!")
        st.session_state.show_level_popup = False

    # Two column layout: Question+Answer | Evaluation
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        # Level indicator
        level_text = st.session_state.level.upper() if st.session_state.level else "EASY"
        st.markdown(f"**Level:** `{level_text}`")
        
        # Question box
        st.markdown(f"""
        <div class="question-box">
            <strong>❓ Question:</strong><br>{st.session_state.question}
        </div>
        """, unsafe_allow_html=True)
        
        # Voice features for question
        if st.session_state.voice_mode:
            # Manual speak button
            if st.button("🔊 Speak Question", use_container_width=True):
                speak_question(st.session_state.question)
            
            # Auto-speak on new question
            if st.session_state.auto_speak:
                if st.session_state.last_spoken_q != st.session_state.question:
                    st.session_state.last_spoken_q = st.session_state.question
                    speak_question(st.session_state.question)
        
        # Answer section
        current_q_hash = hash(str(st.session_state.question))
        
        if st.session_state.voice_mode:
            # Reset on question change
            if "prev_q_hash" not in st.session_state:
                st.session_state.prev_q_hash = current_q_hash
            if st.session_state.prev_q_hash != current_q_hash:
                st.session_state.prev_q_hash = current_q_hash
                st.session_state.transcribed_answer = ""
            
            # Voice recording section
            st.markdown("---")
            st.markdown("#### 🎤 Record Your Answer")
            st.caption("Click START, speak your answer, then click STOP")
            
            # Voice recorder
            transcribed_text = speech_to_text(
                start_prompt="🔴 START Recording",
                stop_prompt="⬛ STOP Recording",
                language="en",
                use_container_width=True,
                just_once=False,
                key=f"stt_{current_q_hash}"
            )
            
            if transcribed_text:
                st.session_state.transcribed_answer = transcribed_text
            
            current_answer = st.session_state.transcribed_answer
            if current_answer:
                st.success("✅ Answer Recorded!")
            
            answer_key = f"ans_{current_q_hash}_{hash(current_answer)}"
            answer = st.text_area("✍️ Your Answer (edit if needed)", value=current_answer, height=120, key=answer_key)
        else:
            answer = st.text_area("✍️ Type Your Answer", height=120, key=f"ans_text_{current_q_hash}")
        
        # Submit button
        if st.button("🚀 Submit", use_container_width=True, type="primary"):
            if not answer or not answer.strip():
                st.warning("Please enter an answer")
            else:
                with st.spinner("Evaluating..."):
                    res = requests.post(f"{API_URL}/answer", params={"user_id": "1", "answer": answer})
                
                if res.status_code == 200:
                    data = res.json()
                    if "error" not in data:
                        st.session_state.evaluation = data["evaluation"]
                        st.session_state.next_question = data["next_question"]
                        if data.get("level_completed"):
                            st.session_state.level_completed = data["level_completed"]
                            st.session_state.show_level_popup = True
                        st.session_state.level = data["level"]
                        st.rerun()
    
    with right_col:
        st.markdown("### 📊 Feedback")
        
        if st.session_state.evaluation:
            eval_text = st.session_state.evaluation
            
            # Parse score
            score = None
            if "Score:" in eval_text:
                try:
                    score = float(eval_text.split("Score:")[1].split("/")[0].strip())
                except:
                    pass
            
            if score is not None:
                # Score display
                color = "🟢" if score >= 4 else "🟡" if score >= 2 else "🔴"
                st.markdown(f"### {color} {score}/5")
                st.progress(min(score / 5, 1.0))
            
            # Feedback
            if "Feedback:" in eval_text:
                feedback = eval_text.split("Feedback:")[1].strip()
                st.info(feedback)
            elif score is None:
                st.write(eval_text)
            
            st.divider()
            
            # Action buttons
            if st.button("➡️ Next Question", use_container_width=True, type="primary"):
                st.session_state.question = st.session_state.next_question
                st.session_state.evaluation = None
                st.session_state.transcribed_answer = ""
                st.session_state.prev_q_hash = None
                st.rerun()
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🏁 End", use_container_width=True):
                    res = requests.get(f"{API_URL}/end-interview", params={"user_id": "1"})
                    if res.status_code == 200:
                        st.session_state.final_summary = res.json()
                        st.session_state.interview_complete = True
                        st.rerun()
            with col_b:
                if st.button("🔁 Restart", use_container_width=True):
                    st.session_state.clear()
                    st.rerun()
        else:
            st.caption("Submit your answer to see feedback here")