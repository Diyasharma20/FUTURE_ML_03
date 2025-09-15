# app.py
import os
import time
import json
from typing import Tuple, Dict, Any, List

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import openai

# -------------------------
# Config (ENV variables)
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # required for GPT fallback
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # optional (for telegram_bridge.py)
FAQ_PATH = "data/faqs.csv"  # sample FAQ dataset
LOG_PATH = "chat_logs.jsonl"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # replace if needed

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# -------------------------
# Minimal intent recognizer
# -------------------------
INTENT_KEYWORDS = {
    "billing": ["bill", "invoice", "payment", "charge", "refund", "subscription", "plan", "renew"],
    "technical": ["error", "issue", "bug", "fail", "not working", "crash", "slow", "timeout"],
    "account": ["login", "password", "reset", "account", "username", "signup", "sign up"],
    "product": ["feature", "price", "available", "supported", "integrate", "integration"],
    "greeting": ["hi", "hello", "hey", "good morning", "good evening"],
    "goodbye": ["bye", "goodbye", "thanks", "thank you", "see you"]
}

# -------------------------
# Load FAQs (simple CSV with 'question','answer')
# -------------------------
@st.cache_data
def load_faqs(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        # create a tiny sample FAQ dataset
        sample = pd.DataFrame({
            "question": [
                "How do I reset my password?",
                "How can I get a refund?",
                "What payment methods do you accept?",
                "How to contact support?",
                "Does the product support integrations?"
            ],
            "answer": [
                "To reset your password: Go to Settings → Account → Reset Password. We'll send a reset link to your registered email.",
                "Refunds are processed within 5-7 business days after approval. Submit a refund request from Billing → Request Refund.",
                "We accept major credit cards (Visa/MasterCard/Amex) and PayPal. For invoices, contact billing@company.com.",
                "You can contact support via support@company.com or use the 'Contact' form on our website. For urgent issues call +1-555-5555.",
                "Yes — we support integrations via REST API and pre-built connectors for popular services. See Integrations → API docs."
            ]
        })
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sample.to_csv(path, index=False)
    return pd.read_csv(path)

faqs_df = load_faqs(FAQ_PATH)

# Build TF-IDF over FAQ questions for quick retrieval
@st.cache_resource
def build_tfidf(docs: List[str]):
    vect = TfidfVectorizer(stop_words='english').fit(docs)
    mat = vect.transform(docs)
    return vect, mat

tfidf_vectorizer, faq_matrix = build_tfidf(faqs_df['question'].tolist())

# -------------------------
# Helper functions
# -------------------------
def detect_intent(text: str) -> Tuple[str, float]:
    """Return best intent and a simple confidence score based on keyword overlap."""
    text_low = text.lower()
    scores = {}
    for intent, keys in INTENT_KEYWORDS.items():
        scores[intent] = sum(text_low.count(k) for k in keys)
    best_intent = max(scores, key=scores.get)
    max_score = scores[best_intent]
    # simple normalization to [0,1]
    confidence = min(1.0, max_score / 3.0)
    return best_intent, confidence

def faq_lookup(question: str, top_k:int=1) -> List[Dict[str,Any]]:
    """Return top_k FAQ matches with similarity scores."""
    q_vec = tfidf_vectorizer.transform([question])
    sims = cosine_similarity(q_vec, faq_matrix).flatten()
    idxs = sims.argsort()[::-1][:top_k]
    results = []
    for i in idxs:
        results.append({"question": faqs_df.iloc[i]['question'],
                        "answer": faqs_df.iloc[i]['answer'],
                        "score": float(sims[i])})
    return results

def openai_chat_fallback(user: str, prompt: str, system_prompt: str = None) -> str:
    """Call OpenAI to generate a response; short wrapper. Requires OPENAI_API_KEY."""
    if not OPENAI_API_KEY:
        return "Smart fallback requires an OPENAI_API_KEY set in your environment."

    system_prompt = system_prompt or (
        "You are a helpful customer support assistant. Answer concisely, be polite, ask clarifying questions only when necessary."
    )
    # Lightweight chat completion
    try:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=400
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI API error: {e}"

def log_conversation(record: Dict[str,Any]):
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# -------------------------
# Business logic: handle message
# -------------------------
def handle_message(user_id: str, message: str) -> Dict[str,Any]:
    t0 = time.time()
    intent, intent_conf = detect_intent(message)
    faqs = faq_lookup(message, top_k=2)
    best_faq = faqs[0] if faqs else None

    chosen_answer = None
    used_faq = False
    used_gpt = False
    actions = []

    # 1) If intent is greeting or goodbye -> canned responses
    if intent == "greeting" and intent_conf >= 0.1:
        chosen_answer = "Hello! 👋 How can I help you today?"
    elif intent == "goodbye" and intent_conf >= 0.1:
        chosen_answer = "Thanks for reaching out. If you need anything else, feel free to message us again. Goodbye!"
    # 2) If FAQ similarity high -> return FAQ answer
    elif best_faq and best_faq['score'] > 0.45:
        chosen_answer = best_faq['answer'] + f"\n\n(Referenced FAQ: \"{best_faq['question']}\")"
        used_faq = True
    # 3) If intent recognized strongly, use canned flows
    elif intent in ("billing", "account", "technical", "product") and intent_conf >= 0.2:
        if intent == "billing":
            chosen_answer = (
                "I see you have a billing question. Could you tell me whether it's about charges, invoices, or refunds? "
                "If you want, type 'invoice' or 'refund' to get quick steps."
            )
            actions.append("ask_specific_billing_type")
        elif intent == "account":
            chosen_answer = "If you're having login issues try: reset your password from the login page. Did you get any error message?"
            actions.append("offer_password_reset_link")
        elif intent == "technical":
            chosen_answer = "Sorry you're facing technical trouble. Can you share the exact error message or a screenshot URL? Meanwhile try clearing cache and retrying."
            actions.append("collect_error_details")
        else:
            chosen_answer = "Can you tell me which feature or product you're asking about? I can share docs or pricing info."
            actions.append("ask_product_name")
    # 4) Otherwise: use GPT as smart fallback
    else:
        used_gpt = True
        prompt = (
            f"User message: \"{message}\"\n\n"
            "You are a helpful, short customer support agent. If the user asks for steps, list them in 3 bullet points. "
            "If the question is ambiguous ask 1 clarifying question. Keep tone friendly and concise."
        )
        ai_resp = openai_chat_fallback(user_id, prompt)
        chosen_answer = ai_resp

    # build response
    response = {
        "reply": chosen_answer,
        "intent": intent,
        "intent_confidence": intent_conf,
        "faq_score": best_faq['score'] if best_faq else None,
        "faq_question": best_faq['question'] if best_faq else None,
        "used_faq": used_faq,
        "used_gpt": used_gpt,
        "actions": actions,
        "latency_s": time.time() - t0
    }

    # log
    log_record = {
        "timestamp": int(time.time()),
        "user_id": user_id,
        "user_message": message,
        "response": response
    }
    log_conversation(log_record)

    return response

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="SupportBot — Demo", page_icon="🤖")
st.title("SupportBot — Customer Support Chatbot (Demo)")

col1, col2 = st.columns([3,1])
with col2:
    st.markdown("**Status**")
    st.write("OpenAI Key set" if OPENAI_API_KEY else "OpenAI Key NOT set (GPT fallback disabled)")
    st.write(f"FAQ items: {len(faqs_df)}")
    st.write("Logs: " + LOG_PATH)

with col1:
    st.markdown("Type your message and press Send. The bot will try FAQ retrieval, intent rules, then fallback to GPT.")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

user_id = st.text_input("User ID (any string)", value="user_001")

msg = st.text_area("Your message", value="", height=120)
send = st.button("Send")

if send and msg.strip():
    with st.spinner("Thinking..."):
        res = handle_message(user_id, msg.strip())
    st.session_state.conversation.append(("user", msg.strip()))
    st.session_state.conversation.append(("bot", res['reply']))

# show conversation
for speaker, text in st.session_state.conversation[-10:]:
    if speaker == "user":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Bot:** {text}")

st.markdown("---")
st.markdown("**FAQ (You can edit data/faqs.csv to add company FAQs)**")
st.table(faqs_df)
