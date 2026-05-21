"""
IncidentIQ - AI-powered Incident Intelligence
Production-ready Streamlit application
"""

import os
import re
import json
import time
import base64
import tempfile
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import streamlit as st
from openai import OpenAI

from langchain.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage
from pinecone import Pinecone

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white
from reportlab.pdfgen import canvas as rl_canvas

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# ── Environment ────────────────────────────────────────────────────────────────
load_dotenv()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT']    = 'incidentiq-agent'
if os.getenv('LANGSMITH_API_KEY'):
    os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGSMITH_API_KEY')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IncidentIQ",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=DM+Mono:wght@400;500&display=swap');

*, html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 1.5rem !important; max-width: 100% !important; }

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #eeeeee;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0; }

/* Chat bubbles */
.bubble-user {
    background: #C0392B;
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px;
    margin: 8px 0 8px 15%;
    font-size: 14px;
    line-height: 1.65;
    word-break: break-word;
}
.bubble-agent {
    background: #f7f7f7;
    border: 1px solid #efefef;
    color: #1a1a1a;
    border-radius: 4px 18px 18px 18px;
    padding: 10px 16px;
    margin: 8px 15% 8px 0;
    font-size: 14px;
    line-height: 1.65;
    word-break: break-word;
}

/* Trace cards */
.trace-user {
    background: #f9f9f9;
    border-left: 3px solid #ddd;
    border-radius: 0 6px 6px 0;
    padding: 7px 10px;
    margin-bottom: 5px;
    font-size: 12px;
    color: #777;
}
.trace-done {
    background: #f0faf5;
    border-left: 3px solid #1D9E75;
    border-radius: 0 6px 6px 0;
    padding: 7px 10px;
    margin-bottom: 5px;
    font-size: 12px;
    color: #333;
}
.trace-pro {
    background: #fafafa;
    border-left: 3px solid #C0392B;
    border-radius: 0 6px 6px 0;
    padding: 7px 10px;
    margin-bottom: 5px;
    font-size: 11px;
    color: #555;
    font-family: 'DM Mono', monospace !important;
}
.badge-ok { background: #e1f5ee; color: #0f6e56; padding: 1px 7px; border-radius: 4px; font-size: 10px; font-family: 'DM Mono', monospace; }
.badge-cached { background: #e6f1fb; color: #185fa5; padding: 1px 7px; border-radius: 4px; font-size: 10px; font-family: 'DM Mono', monospace; }
.badge-time { color: #bbb; font-size: 10px; font-family: 'DM Mono', monospace; }

/* Status pill */
.status-loaded { background: #e8f5e9; color: #2e7d32; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 500; }
.status-empty  { background: #f5f5f5; color: #999; padding: 3px 10px; border-radius: 20px; font-size: 11px; }

/* Section label */
.sec-label { font-size: 10px; letter-spacing: 0.09em; color: #bbb; font-weight: 600; margin: 14px 0 6px; }

/* Timeline */
.tl-header {
    background: linear-gradient(135deg, #1C2833 0%, #2C3E50 100%);
    padding: 20px 22px;
    border-radius: 12px;
    margin-bottom: 18px;
}
.metric-card {
    background: #f8f8f8;
    border-radius: 10px;
    padding: 14px;
    text-align: left;
    border-bottom: 3px solid #ddd;
}
.tl-card {
    background: white;
    border-radius: 10px;
    padding: 14px 16px;
    border: 1px solid #f0f0f0;
    margin-bottom: 4px;
}
.learning-card {
    background: #f8f8f8;
    border-radius: 10px;
    padding: 14px;
    height: 100%;
}
</style>
""", unsafe_allow_html=True)

# ── Language config ────────────────────────────────────────────────────────────
LABELS = {
    "Nederlands": {
        "chat_placeholder": "Stel een vraag of plak een YouTube URL...",
        "btn_pdf":     "📄  Key Concepts PDF",
        "btn_visual":  "📊  Visuele tijdlijn",
        "btn_xvr":     "🎮  XVR Scenario",
        "send_header": "📤  Versturen",
        "send_to":     "Naar (e-mailadres)",
        "send_ph":     "naam@email.be",
        "distrib":     "Voeg distributielijst toe",
        "doc_label":   "Document",
        "doc_opts":    ["Key Concepts PDF", "Visuele tijdlijn", "XVR Scenario"],
        "send_btn":    "Verstuur",
        "no_video":    "Geen video geladen",
        "video_ok":    "Video geladen",
        "welcome":     "Drop een YouTube URL om te beginnen.",
        "welcome_sub": "Stel vragen · Genereer rapporten · Maak XVR scenario's",
        "mode_user":   "Gebruiker",
        "mode_pro":    "Pro",
        "activity":    "Agent activiteit",
        "generating":  "Genereren...",
        "sending":     "Versturen...",
        "clear":       "Wissen",
        "lang_tool":   "dutch",
    },
    "English": {
        "chat_placeholder": "Ask a question or paste a YouTube URL...",
        "btn_pdf":     "📄  Key Concepts PDF",
        "btn_visual":  "📊  Visual Timeline",
        "btn_xvr":     "🎮  XVR Scenario",
        "send_header": "📤  Send",
        "send_to":     "To (email address)",
        "send_ph":     "name@email.com",
        "distrib":     "Add distribution list",
        "doc_label":   "Document",
        "doc_opts":    ["Key Concepts PDF", "Visual Timeline", "XVR Scenario"],
        "send_btn":    "Send",
        "no_video":    "No video loaded",
        "video_ok":    "Video loaded",
        "welcome":     "Drop a YouTube URL to get started.",
        "welcome_sub": "Ask questions · Generate reports · Create XVR scenarios",
        "mode_user":   "User",
        "mode_pro":    "Pro",
        "activity":    "Agent activity",
        "generating":  "Generating...",
        "sending":     "Sending...",
        "clear":       "Clear",
        "lang_tool":   "english",
    },
    "Français": {
        "chat_placeholder": "Posez une question ou collez une URL YouTube...",
        "btn_pdf":     "📄  Concepts clés PDF",
        "btn_visual":  "📊  Chronologie visuelle",
        "btn_xvr":     "🎮  Scénario XVR",
        "send_header": "📤  Envoyer",
        "send_to":     "À (adresse e-mail)",
        "send_ph":     "nom@email.fr",
        "distrib":     "Ajouter liste de diffusion",
        "doc_label":   "Document",
        "doc_opts":    ["Concepts clés PDF", "Chronologie visuelle", "Scénario XVR"],
        "send_btn":    "Envoyer",
        "no_video":    "Aucune vidéo chargée",
        "video_ok":    "Vidéo chargée",
        "welcome":     "Collez une URL YouTube pour commencer.",
        "welcome_sub": "Posez des questions · Générez des rapports · Créez des scénarios XVR",
        "mode_user":   "Utilisateur",
        "mode_pro":    "Pro",
        "activity":    "Activité agent",
        "generating":  "Génération...",
        "sending":     "Envoi...",
        "clear":       "Effacer",
        "lang_tool":   "french",
    },
}

# ── Session state ──────────────────────────────────────────────────────────────
defaults = {
    "messages":      [],
    "thread_id":     f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "video_loaded":  False,
    "video_title":   "",
    "trace_steps":   [],
    "pro_mode":      False,
    "language":      "Nederlands",
    "agent":         None,
    "last_pdf_path": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

L = LABELS[st.session_state.language]

# ── Shared components ──────────────────────────────────────────────────────────
@st.cache_resource
def init_components():
    llm             = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    pc              = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    vectorstore     = PineconeVectorStore(
        index_name="incidentiq",
        embedding=embedding_model,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    )
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return llm, embedding_model, pc, vectorstore, openai_client

llm, embedding_model, pc, vectorstore, openai_client = init_components()

# ── Helpers ────────────────────────────────────────────────────────────────────
def extract_video_id(url):
    if "v=" in url:     return url.split("v=")[1].split("&")[0]
    if "youtu.be/" in url: return url.split("youtu.be/")[1].split("?")[0]
    raise ValueError(f"Cannot extract video ID: {url}")

def clean_transcript(text):
    text = re.sub(r'\[Music\]|\[Applause\]|\[Laughter\]|\[Cheering\]', '', text)
    text = re.sub(r'\[\d{2}:\d{2}\]\s*(?=\[\d{2}:\d{2}\])', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def add_trace(step_type, label, detail="", latency=None, badge=None):
    st.session_state.trace_steps.append({
        "type":    step_type,
        "label":   label,
        "detail":  detail,
        "latency": latency,
        "badge":   badge,
    })

def get_gmail_service():
    SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
    creds  = None
    tp     = Path("token.json")
    cp     = Path("credentials.json")
    if not cp.exists():
        raise FileNotFoundError("credentials.json not found.")
    if tp.exists():
        creds = Credentials.from_authorized_user_file(str(tp), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow  = InstalledAppFlow.from_client_secrets_file(str(cp), SCOPES)
            creds = flow.run_local_server(port=0)
        tp.write_text(creds.to_json())
    return build("gmail", "v1", credentials=creds)

# ── Visual timeline renderer ───────────────────────────────────────────────────
def render_visual_timeline(json_str):
    try:
        data = json.loads(json_str)
    except Exception:
        st.error("Could not render visual timeline.")
        return

    COLOR_MAP = {
        "red":   ("#C0392B", "#FAECE7", "#7A2419"),
        "amber": ("#E67E22", "#FEF3E2", "#7A4A10"),
        "green": ("#1D9E75", "#E1F5EE", "#0A5C3F"),
        "blue":  ("#2980B9", "#E8F4FD", "#1A5276"),
    }

    # Header
    st.markdown(f"""
    <div class="tl-header">
        <div style="font-size:10px;letter-spacing:0.1em;color:#C0392B;background:#C0392B22;
                    padding:3px 10px;border-radius:4px;border:1px solid #C0392B44;
                    display:inline-block;margin-bottom:10px;font-weight:600">
            INCIDENT ANALYSIS
        </div>
        <div style="font-size:19px;font-weight:600;color:white;margin-bottom:4px">
            {data.get('title','')}
        </div>
        <div style="font-size:12px;color:#ffffff66">
            {data.get('subtitle','')} &nbsp;·&nbsp; {data.get('duration','')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    metrics = data.get("metrics", [])
    if metrics:
        cols = st.columns(len(metrics))
        for i, m in enumerate(metrics):
            ch, bh, th = COLOR_MAP.get(m.get("color","blue"), COLOR_MAP["blue"])
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card" style="border-bottom-color:{ch}">
                    <div style="font-size:22px;font-weight:600;color:#1a1a1a;
                                font-family:'DM Mono',monospace;line-height:1">
                        {m.get('value','')}
                        <span style="font-size:12px;color:#aaa;font-weight:400"> {m.get('unit','')}</span>
                    </div>
                    <div style="font-size:11px;color:#777;margin-top:5px">{m.get('label','')}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>INCIDENT TIJDLIJN</div>", unsafe_allow_html=True)

    # Timeline
    for ev in data.get("timeline", []):
        ch, bh, th = COLOR_MAP.get(ev.get("color","blue"), COLOR_MAP["blue"])
        badge = ev.get("badge","")
        quote = ev.get("quote","")
        tags  = ev.get("tags",[])

        tags_html  = "".join([
            f'<span style="font-size:10px;padding:2px 8px;border-radius:4px;'
            f'background:#f5f5f5;color:#888;border:1px solid #eee;margin-right:4px">{t}</span>'
            for t in tags
        ])
        quote_html = (
            f'<div style="border-left:2px solid {ch};padding-left:10px;margin:8px 0;'
            f'font-size:12px;color:#555;font-style:italic;line-height:1.6">{quote}</div>'
        ) if quote else ""

        st.markdown(f"""
        <div style="display:flex;gap:0;margin-bottom:2px">
            <div style="width:50px;flex-shrink:0;padding-top:14px;
                        text-align:right;padding-right:10px">
                <span style="font-size:10px;color:#aaa;font-family:'DM Mono',monospace">
                    {ev.get('timestamp','')}
                </span>
            </div>
            <div style="width:20px;flex-shrink:0;display:flex;flex-direction:column;
                        align-items:center;padding-top:14px">
                <div style="width:10px;height:10px;border-radius:50%;background:{ch};
                            box-shadow:0 0 0 3px {ch}22;flex-shrink:0"></div>
                <div style="width:1px;flex:1;background:#f0f0f0;min-height:20px"></div>
            </div>
            <div style="flex:1;padding:8px 0 16px 12px">
                <div class="tl-card" style="border-left:3px solid {ch}">
                    <div style="display:flex;justify-content:space-between;
                                align-items:flex-start;margin-bottom:8px;gap:8px">
                        <div style="font-size:13px;font-weight:600;color:#1a1a1a">
                            {ev.get('title','')}
                        </div>
                        <span style="font-size:10px;padding:2px 8px;border-radius:4px;
                                     background:{bh};color:{th};font-weight:500;
                                     white-space:nowrap;flex-shrink:0">{badge}</span>
                    </div>
                    <div style="font-size:12px;color:#555;line-height:1.7">
                        {ev.get('text','')}
                    </div>
                    {quote_html}
                    <div style="margin-top:8px">{tags_html}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='sec-label'>KEY LEARNINGS</div>", unsafe_allow_html=True)
    learnings = data.get("learnings", [])
    if learnings:
        cols = st.columns(2)
        for i, l in enumerate(learnings):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="learning-card" style="margin-bottom:8px">
                    <div style="font-size:10px;color:#C0392B;font-family:'DM Mono',monospace;
                                font-weight:600;margin-bottom:6px">{l.get('number','')}</div>
                    <div style="font-size:12px;font-weight:600;color:#1a1a1a;margin-bottom:5px">
                        {l.get('title','')}
                    </div>
                    <div style="font-size:11px;color:#666;line-height:1.6">{l.get('text','')}</div>
                </div>
                """, unsafe_allow_html=True)

# ── PDF generator ──────────────────────────────────────────────────────────────
def generate_pdf_file(context, language="dutch", source_url=""):
    RED    = HexColor("#C0392B")
    DARK   = HexColor("#1C2833")
    ORANGE = HexColor("#E67E22")
    GREEN  = HexColor("#1E8449")
    WHITE  = white

    lang_map = {"dutch": "Dutch", "english": "English", "french": "French"}
    lang = lang_map.get(language, "Dutch")

    prompt = (
        f'Extract structured info for an incident training cheatsheet in {lang}.\n'
        f'Return only this JSON: {{"title":"...","subtitle":"...",'
        f'"keypoints":["..."],"recommendations":["..."]}}\n\n'
        f'Rules: max 12 words per item, no timestamps.\n\nContext:\n{context}\n\nJSON:'
    )
    raw  = re.sub(r'```json|```', '', llm.invoke(prompt).content.strip()).strip()
    data = json.loads(raw)

    filepath = f'/tmp/incidentiq_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    c = rl_canvas.Canvas(filepath, pagesize=A4)
    W, H = A4

    c.setFillColor(RED); c.rect(0, H-3.2*cm, W, 3.2*cm, fill=1, stroke=0)
    c.setFillColor(WHITE); c.circle(1.8*cm, H-1.6*cm, 0.85*cm, fill=1, stroke=0)
    c.setFillColor(RED); c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(1.8*cm, H-1.95*cm, "IQ")
    c.setFillColor(WHITE); c.setFont("Helvetica-Bold", 15)
    c.drawString(3.2*cm, H-1.3*cm, data.get("title", "IncidentIQ"))
    c.setFont("Helvetica", 10)
    c.drawString(3.2*cm, H-1.85*cm, data.get("subtitle", ""))
    c.setFont("Helvetica", 8)
    c.drawRightString(W-1.2*cm, H-1.3*cm, datetime.now().strftime("%d/%m/%Y"))
    c.drawRightString(W-1.2*cm, H-1.75*cm, "Generated by IncidentIQ AI")
    c.setFillColor(ORANGE); c.rect(0, H-3.6*cm, W, 0.4*cm, fill=1, stroke=0)

    y = H - 5.0*cm

    def sh(y, title, color=DARK):
        c.setFillColor(color); c.setFont("Helvetica-Bold", 11)
        c.drawString(1.2*cm, y, title.upper())
        c.setStrokeColor(color); c.setLineWidth(1.5)
        c.line(1.2*cm, y-0.2*cm, W-1.2*cm, y-0.2*cm)
        return y - 0.7*cm

    def bi(y, txt, color=DARK, bc=RED):
        c.setFillColor(bc); c.circle(1.2*cm, y+0.2*cm, 0.1*cm, fill=1, stroke=0)
        c.setFillColor(color); c.setFont("Helvetica", 9.5)
        mw = W - 1.5*cm - 1.2*cm
        words = txt.split(); line, lines = "", []
        for w in words:
            t = line + w + " "
            if c.stringWidth(t, "Helvetica", 9.5) < mw:
                line = t
            else:
                lines.append(line.strip()); line = w + " "
        lines.append(line.strip())
        for i, l in enumerate(lines):
            c.drawString(1.5*cm, y - i*0.45*cm, l)
        return y - len(lines)*0.45*cm - 0.3*cm

    y = sh(y, "Key Points", RED)
    for kp in data.get("keypoints", []): y = bi(y, kp)
    y -= 0.4*cm

    y = sh(y, "AI-generated recommendations based on the video", GREEN)
    c.setFillColor(HexColor("#EAFAF1")); c.setStrokeColor(GREEN); c.setLineWidth(0.8)
    c.roundRect(1.2*cm, y-0.75*cm, W-2.4*cm, 0.65*cm, 3, fill=1, stroke=1)
    c.setFillColor(GREEN); c.setFont("Helvetica-Bold", 8)
    c.drawString(1.55*cm, y-0.3*cm, "AI analysis:")
    c.setFillColor(DARK); c.setFont("Helvetica-Oblique", 8)
    c.drawString(3.0*cm, y-0.3*cm,
        "Automatically extracted from video - not cited from a person.")
    y -= 1.0*cm
    for rec in data.get("recommendations", []): y = bi(y, rec, bc=GREEN)

    c.setFillColor(RED); c.rect(0, 1.2*cm, W, 0.15*cm, fill=1, stroke=0)
    c.setFillColor(DARK); c.rect(0, 0, W, 1.2*cm, fill=1, stroke=0)
    c.setFillColor(WHITE); c.setFont("Helvetica", 7.5)
    c.drawString(1.2*cm, 0.65*cm, "IncidentIQ - AI-powered Incident Intelligence")
    if source_url:
        c.drawCentredString(W/2, 0.65*cm, f"Source: {source_url[:70]}")
    c.drawRightString(W-1.2*cm, 0.65*cm, "Page 1/1")
    c.save()
    return filepath, data.get("title", "Cheatsheet")

# ── Build LangGraph agent ──────────────────────────────────────────────────────
@st.cache_resource
def build_agent(_llm, _vectorstore, _pc):
    DISTRIBUTION_LIST = os.getenv("GMAIL_DISTRIBUTION_LIST", "").split(",")

    @tool
    def fetch_youtube_transcript(youtube_url: str) -> str:
        """
        Fetch the transcript of a YouTube video and store it in Pinecone cloud.
        Use this tool when the user provides a YouTube URL.
        Checks Pinecone cache first - YouTube is only called once per video ever.
        Returns a confirmation with the number of chunks stored.
        """
        try:
            if "v=" in youtube_url:
                video_id = youtube_url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in youtube_url:
                video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
            else:
                raise ValueError(f"Cannot extract video ID: {youtube_url}")

            index = _pc.Index("incidentiq")
            stats = index.describe_index_stats()
            if stats.total_vector_count > 0:
                test = _vectorstore.similarity_search("incident", k=1)
                if test:
                    return (
                        f"Video already in Pinecone.\n"
                        f"Video ID: {video_id}\n"
                        f"Using cached data - no YouTube request needed.\n"
                        f"Ready for Q&A."
                    )
            try:
                entries         = YouTubeTranscriptApi().fetch(video_id, languages=["en","nl","fr"])
                transcript_list = entries.snippets
            except NoTranscriptFound:
                return f"No transcript found for video {video_id}."
            except TranscriptsDisabled:
                return f"Transcripts disabled for video {video_id}."
            except Exception as e:
                return f"Could not fetch transcript. If YouTube blocks your IP wait 30-60 min.\nError: {str(e)}"

            plain       = " ".join(t.text for t in transcript_list)
            timestamped = " ".join(
                f"[{int(t.start//60):02d}:{int(t.start%60):02d}] {t.text}"
                for t in transcript_list
            )
            plain       = clean_transcript(plain)
            timestamped = clean_transcript(timestamped)

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks   = splitter.create_documents(
                texts=[timestamped],
                metadatas=[{"video_id": video_id, "source": youtube_url}]
            )
            _vectorstore.add_documents(chunks)
            return (
                f"Transcript loaded.\n"
                f"Video ID: {video_id}\n"
                f"Chunks stored in Pinecone: {len(chunks)}\n"
                f"Ready for Q&A."
            )
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def search_video_knowledge(query: str) -> str:
        """
        Search the Pinecone knowledge base for information relevant to the query.
        Use this tool to answer questions about the loaded video content.
        Uses query rewriting and multi-query for maximum retrieval quality.
        Automatically translates non-English queries for better results.
        Returns relevant transcript excerpts with timestamp sources.
        """
        try:
            english_query = _llm.invoke(
                f"Translate to English, return only translation: {query}"
            ).content.strip()
            rewritten = _llm.invoke(
                f"Rewrite for incident video search, max 20 words.\n"
                f"Query: {english_query}\nRewritten:"
            ).content.strip()
            try:
                response = _llm.invoke(
                    f"Generate 3 search query variations. Return JSON list.\n"
                    f"Question: {rewritten}\nJSON:"
                ).content.strip()
                queries = json.loads(re.sub(r"```json|```", "", response).strip())
            except Exception:
                queries = [rewritten]
            queries.append(rewritten)

            all_docs = {}
            for q in queries:
                for doc in _vectorstore.similarity_search(q, k=4):
                    key = doc.page_content[:100]
                    if key not in all_docs:
                        all_docs[key] = doc

            if not all_docs:
                return "No relevant information found. Please load a YouTube video first."

            combined  = list(all_docs.values())
            all_ts    = re.findall(r"\[\d{2}:\d{2}\]", " ".join([d.page_content for d in combined]))
            seen, uts = set(), []
            for t in all_ts:
                if t not in seen:
                    seen.add(t); uts.append(t)
            clean = [
                re.sub(r"\[\d{2}:\d{2}\]\s*(?=\[\d{2}:\d{2}\])", "", d.page_content)
                for d in combined
            ]
            return "\n\n".join(clean) + f"\n\nSources: {' | '.join(uts[:5])}"
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def summarize_video(language: str = "english") -> str:
        """
        Generate a structured text summary of the entire loaded video.
        Use this tool when the user asks for a full text summary.
        For visual timeline use generate_visual_summary instead.
        Specify language as english, dutch or french.
        Returns structured summary with introduction, key points, lessons and conclusion.
        """
        try:
            results = _vectorstore.similarity_search(
                "main topic lessons learned conclusions key points", k=12
            )
            if not results:
                return "No video content found. Please load a YouTube video first."
            context = "\n\n".join([
                re.sub(r"\[\d{2}:\d{2}\]\s*(?=\[\d{2}:\d{2}\])", "", r.page_content)
                for r in results
            ])
            lang_map = {
                "english": "English",
                "dutch":   "Dutch - natural direct Belgian incident training language",
                "french":  "French",
            }
            lang   = lang_map.get(language.lower(), "English")
            prompt = (
                f"Write a structured summary in {lang}.\n"
                f"Structure: **Introduction** / **Key Points** / **Lessons Learned** / **Conclusion**\n"
                f"Rules: bullet points only, max 15 words each, strictly from context.\n\n"
                f"Context:\n{context}\n\nSummary:"
            )
            return _llm.invoke(prompt).content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def generate_pdf_cheatsheet(language: str = "dutch", source_url: str = "") -> str:
        """
        Generate a professional 1-page PDF cheatsheet from the loaded video content.
        Use this tool when the user asks for a cheatsheet, key concepts document or PDF.
        Specify language as dutch, english or french.
        Returns the file path of the generated PDF.
        """
        try:
            results = _vectorstore.similarity_search(
                "key points lessons recommendations conclusions", k=10
            )
            if not results:
                return "No video content found. Please load a YouTube video first."
            context  = "\n\n".join([r.page_content for r in results])
            filepath, title = generate_pdf_file(context, language, source_url)
            return f"PDF generated.\nFile path: {filepath}\nTitle: {title}"
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def send_gmail(
        pdf_path:       str = "",
        text_content:   str = "",
        subject_suffix: str = "Document",
        custom_emails:  str = "",
    ) -> str:
        """
        Send any generated document to recipients via Gmail.
        Use after any generation tool has created content.
        pdf_path: file path of generated PDF to attach.
        text_content: text to include in email body.
        subject_suffix: label for subject line e.g. Key Concepts, XVR Scenario.
        custom_emails: comma-separated email addresses to send to.
        Returns confirmation with recipient list.
        """
        try:
            recipients = [e.strip() for e in DISTRIBUTION_LIST if e.strip()]
            if custom_emails:
                recipients.extend([e.strip() for e in custom_emails.split(",") if e.strip()])
            if not recipients:
                return "No recipients provided. Please specify at least one email address."

            service = get_gmail_service()
            subject = f"IncidentIQ - {subject_suffix} - {datetime.now().strftime('%d/%m/%Y')}"
            body    = (
                f"Dear colleague,\n\n"
                f"Please find the AI-generated {subject_suffix} from the latest incident training video.\n\n"
            )
            if text_content:
                body += f"{text_content}\n\n"
            body += (
                "Generated by IncidentIQ AI Agent.\n"
                "Content should be reviewed by a qualified officer before operational use.\n\n"
                "Best regards,\nIncidentIQ AI Agent"
            )

            msg = MIMEMultipart()
            msg["From"]    = "me"
            msg["To"]      = ", ".join(recipients)
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            if pdf_path and Path(pdf_path).exists():
                with open(pdf_path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename={Path(pdf_path).name}"
                )
                msg.attach(part)

            raw     = base64.urlsafe_b64encode(msg.as_bytes()).decode()
            message = {"raw": raw}
            service.users().messages().send(userId="me", body=message).execute()
            return f"Sent successfully.\nRecipients: {', '.join(recipients)}"
        except Exception as e:
            return f"Error sending email: {str(e)}"

    @tool
    def generate_xvr_scenario(language: str = "dutch") -> str:
        """
        Generate a structured XVR simulation scenario brief based on the loaded incident video.
        Use this tool when the user asks to create an XVR scenario.
        Specify language as dutch, english or french.
        Returns formatted scenario brief for XVR operators.
        """
        try:
            results = _vectorstore.similarity_search(
                "location building fire cause complications decisions "
                "resources weather time casualties evacuation", k=12
            )
            if not results:
                return "No video content found. Please load a YouTube video first."
            context = "\n\n".join([
                re.sub(r"\[\d{2}:\d{2}\]\s*(?=\[\d{2}:\d{2}\])", "", r.page_content)
                for r in results
            ])
            lang_map = {
                "dutch":   "Dutch - professional Belgian fire service terminology",
                "english": "English",
                "french":  "French",
            }
            lang   = lang_map.get(language.lower(), "Dutch")
            prompt = (
                f"You are an expert XVR simulation scenario designer for emergency services.\n"
                f"Generate a complete XVR operator scenario brief in {lang}.\n\n"
                f"SCENARIO BRIEF - XVR SIMULATION\n================================\n\n"
                f"INCIDENT TITLE:\n[Short descriptive title]\n\n"
                f"LOCATION & BUILDING:\n"
                f"- Building type: [type]\n- Floors: [number]\n- Construction: [facade, materials]\n\n"
                f"INITIAL SITUATION T+00:00:\n"
                f"- Fire location: [exact location]\n- Visibility: [smoke, flames]\n"
                f"- Known casualties: [number]\n- First resources: [vehicles, personnel]\n\n"
                f"ENVIRONMENTAL CONDITIONS:\n"
                f"- Time of day: [if mentioned]\n- Weather: [if mentioned]\n"
                f"- Special hazards: [materials, access]\n\n"
                f"SCENARIO COMPLICATIONS (inject in order):\n"
                f"- T+[time]: [complication 1]\n- T+[time]: [complication 2]\n"
                f"- T+[time]: [complication 3]\n- T+[time]: [complication 4]\n\n"
                f"CRITICAL DECISION MOMENTS:\n"
                f"1. [Decision moment]\n2. [Decision moment]\n3. [Decision moment]\n\n"
                f"LEARNING OBJECTIVES:\n"
                f"- [Objective 1]\n- [Objective 2]\n- [Objective 3]\n\n"
                f"DEBRIEFING QUESTIONS:\n"
                f"1. [Question based on actual mistakes]\n"
                f"2. [Question based on actual mistakes]\n"
                f"3. [Question based on actual mistakes]\n\n"
                f"XVR OPERATOR NOTES:\n[Notes about key moments to inject]\n\n"
                f"Rules: base on context only, realistic timings, never invent details.\n\n"
                f"Context:\n{context}\n\nScenario brief:"
            )
            return _llm.invoke(prompt).content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def generate_visual_summary(language: str = "dutch") -> str:
        """
        Generate structured JSON data for visual timeline rendering in the app.
        Use this tool when the user asks for a visual summary or timeline view.
        Specify language as dutch, english or french.
        Returns JSON with metrics, timeline events and key learnings.
        """
        try:
            results = _vectorstore.similarity_search(
                "timeline events cause complications lessons learned "
                "mistakes decisions outcome casualties", k=12
            )
            if not results:
                return "No video content found. Please load a YouTube video first."
            context = "\n\n".join([
                re.sub(r"\[\d{2}:\d{2}\]\s*(?=\[\d{2}:\d{2}\])", "", r.page_content)
                for r in results
            ])
            lang_map = {
                "dutch":   "Dutch - natural direct language",
                "english": "English",
                "french":  "French",
            }
            lang = lang_map.get(language.lower(), "Dutch")
            prompt = (
                f'Extract structured information from this incident video context.\n'
                f'Respond in {lang} with this exact JSON and nothing else:\n'
                f'{{"title":"Short incident title","subtitle":"Presenter and event",'
                f'"duration":"duration or unknown",'
                f'"metrics":['
                f'{{"value":"20","unit":"min","label":"Watervertraging","color":"red"}},'
                f'{{"value":"3","unit":"","label":"Kritieke fouten","color":"amber"}},'
                f'{{"value":"16","unit":"","label":"Verdiepingen","color":"blue"}},'
                f'{{"value":"0","unit":"","label":"Slachtoffers","color":"green"}}],'
                f'"timeline":[{{"timestamp":"00:00","title":"Event title",'
                f'"text":"Max 2 sentences.","quote":"","tags":["tag1"],'
                f'"color":"blue","badge":"Context"}}],'
                f'"learnings":[{{"number":"01","title":"Learning title",'
                f'"text":"Max 2 sentences."}}],'
                f'"source_url":""}}\n\n'
                f'Rules: exactly 4 metrics, 4-6 timeline events, 4 learnings, '
                f'colors: red/amber/green/blue, all text in {lang}.\n\n'
                f'Context:\n{context}\n\nJSON:'
            )
            response = _llm.invoke(prompt)
            raw      = re.sub(r"```json|```", "", response.content.strip()).strip()
            json.loads(raw)  # validate
            return raw
        except Exception as e:
            return f"Error: {str(e)}"

    AGENT_TOOLS = [
        fetch_youtube_transcript,
        search_video_knowledge,
        summarize_video,
        generate_pdf_cheatsheet,
        send_gmail,
        generate_xvr_scenario,
        generate_visual_summary,
    ]

    SYSTEM_PROMPT = """You are IncidentIQ, an AI agent specialized in incident training and knowledge extraction.
You help emergency services professionals extract knowledge from incident training videos.
Sector-agnostic: fire services, police, EMS, civil protection or any training context.

Tools:
- fetch_youtube_transcript: load a YouTube video into the knowledge base
- search_video_knowledge: answer questions about the loaded video
- summarize_video: generate a structured text summary
- generate_pdf_cheatsheet: create a 1-page PDF with key concepts
- send_gmail: send any content by email - PDF attachment, text or both
- generate_xvr_scenario: generate a structured XVR simulation scenario brief
- generate_visual_summary: generate visual timeline summary JSON for the app

ROUTING:
- YouTube URL -> fetch_youtube_transcript
- Question about video -> search_video_knowledge
- Summary request -> summarize_video
- PDF or cheatsheet request -> generate_pdf_cheatsheet
- Email request -> send_gmail with correct parameters
- XVR scenario request -> generate_xvr_scenario
- Visual summary or timeline request -> generate_visual_summary
- Multiple actions -> chain tools in correct order

LANGUAGE: Always respond in the same language as the user message.
FORMAT: Bullet points only, max 15 words per bullet, confirm after tool calls.
"""

    llm_with_tools = _llm.bind_tools(AGENT_TOOLS)

    def agent_node(state: MessagesState):
        system   = SystemMessage(content=SYSTEM_PROMPT)
        messages = [system] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(AGENT_TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

# Build agent
if st.session_state.agent is None:
    with st.spinner("Loading IncidentIQ..."):
        st.session_state.agent = build_agent(llm, vectorstore, pc)

# ── Ask function ───────────────────────────────────────────────────────────────
def ask(message: str):
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    inputs = {"messages": [HumanMessage(content=message)]}
    start  = time.time()
    final  = ""
    calls  = []

    for event in st.session_state.agent.stream(inputs, config=config, stream_mode="values"):
        last = event["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            for tc in last.tool_calls:
                calls.append(tc["name"])
        if hasattr(last, "content") and isinstance(last.content, str) and last.content.strip():
            final = last.content.strip()

    latency = time.time() - start
    return final, calls, latency

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Logo ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:20px 16px 12px;margin-bottom:4px">
        <div style="display:flex;align-items:center;gap:10px">
            <div style="width:34px;height:34px;background:#C0392B;border-radius:8px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:13px;font-weight:700;color:white;flex-shrink:0">IQ</div>
            <div>
                <div style="font-size:15px;font-weight:600;color:#1a1a1a;line-height:1.2">IncidentIQ</div>
                <div style="font-size:10px;color:#aaa">AI Incident Intelligence</div>
            </div>
        </div>
    </div>
    <hr style="border:none;border-top:1px solid #f0f0f0;margin:0 0 12px">
    """, unsafe_allow_html=True)

    # ── Language ──────────────────────────────────────────────────────────────
    lang = st.selectbox(
        "🌐",
        ["Nederlands", "English", "Français"],
        index=["Nederlands", "English", "Français"].index(st.session_state.language),
        label_visibility="collapsed",
    )
    if lang != st.session_state.language:
        st.session_state.language = lang
        L = LABELS[lang]
        st.rerun()
    L = LABELS[st.session_state.language]

    # ── Video status ──────────────────────────────────────────────────────────
    st.markdown("<div class='sec-label'>VIDEO</div>", unsafe_allow_html=True)
    if st.session_state.video_loaded:
        title_short = st.session_state.video_title[:35] + "..." if len(st.session_state.video_title) > 35 else st.session_state.video_title
        st.markdown(f"""
        <div style="background:#f0faf5;border-radius:8px;padding:9px 12px;
                    border:1px solid #c8e6c9;margin-bottom:10px">
            <div style="font-size:10px;color:#1D9E75;font-weight:600;margin-bottom:2px">
                ● {L['video_ok']}
            </div>
            <div style="font-size:11px;color:#444">{title_short}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#fafafa;border-radius:8px;padding:9px 12px;
                    border:1px solid #eee;margin-bottom:10px">
            <div style="font-size:11px;color:#bbb">{L['no_video']}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Action buttons ────────────────────────────────────────────────────────
    st.markdown("<div class='sec-label'>TOOLS</div>", unsafe_allow_html=True)

    # PDF Cheatsheet
    if st.button(L["btn_pdf"], use_container_width=True):
        if not st.session_state.video_loaded:
            st.warning("Load a video first.")
        else:
            with st.spinner(L["generating"]):
                t0 = time.time()
                results = vectorstore.similarity_search(
                    "key points lessons recommendations conclusions", k=10
                )
                if results:
                    context  = "\n\n".join([r.page_content for r in results])
                    filepath, title = generate_pdf_file(context, L["lang_tool"])
                    st.session_state.last_pdf_path = filepath
                    lat = time.time() - t0
                    add_trace("done", "generate_pdf_cheatsheet", f"{lat:.1f}s", latency=lat, badge="ok")
                    with open(filepath, "rb") as f:
                        pdf_bytes = f.read()
                    st.download_button(
                        "⬇️  Download PDF",
                        pdf_bytes,
                        file_name="incidentiq_cheatsheet.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": f"PDF cheatsheet aangemaakt: **{title}**",
                        "calls":   ["generate_pdf_cheatsheet"],
                        "visual":  False,
                    })

    # Visual Summary
    if st.button(L["btn_visual"], use_container_width=True):
        if not st.session_state.video_loaded:
            st.warning("Load a video first.")
        else:
            with st.spinner(L["generating"]):
                add_trace("pro", "generate_visual_summary", f"lang: {L['lang_tool']}")
                result, calls, lat = ask(
                    f"Generate a visual timeline summary in {L['lang_tool']}"
                )
                add_trace("done", "generate_visual_summary", f"{lat:.1f}s", latency=lat, badge="ok")
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": result,
                    "calls":   calls,
                    "visual":  True,
                })
                st.rerun()

    # XVR Scenario
    if st.button(L["btn_xvr"], use_container_width=True):
        if not st.session_state.video_loaded:
            st.warning("Load a video first.")
        else:
            with st.spinner(L["generating"]):
                add_trace("pro", "generate_xvr_scenario", f"lang: {L['lang_tool']}")
                result, calls, lat = ask(
                    f"Generate an XVR scenario brief in {L['lang_tool']}"
                )
                add_trace("done", "generate_xvr_scenario", f"{lat:.1f}s", latency=lat, badge="ok")
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": result,
                    "calls":   calls,
                    "visual":  False,
                })
                st.rerun()

    # ── Send section ──────────────────────────────────────────────────────────
    st.markdown(f"<div class='sec-label'>{L['send_header']}</div>", unsafe_allow_html=True)

    email_to = st.text_input(
        L["send_to"],
        placeholder=L["send_ph"],
        key="email_to_input",
    )
    add_distrib = st.checkbox(L["distrib"], value=False, key="add_distrib_cb")

    doc_choice = st.selectbox(
        L["doc_label"],
        L["doc_opts"],
        key="doc_choice_select",
    )

    if st.button(L["send_btn"], use_container_width=True, type="primary"):
        if not email_to and not add_distrib:
            st.warning("Enter at least one email address.")
        else:
            distrib_str = os.getenv("GMAIL_DISTRIBUTION_LIST", "") if add_distrib else ""
            all_emails  = email_to
            if distrib_str:
                all_emails = email_to + "," + distrib_str if email_to else distrib_str

            with st.spinner(L["sending"]):
                add_trace("pro", "send_gmail", f"to: {email_to}")

                if "PDF" in doc_choice:
                    if st.session_state.last_pdf_path:
                        result, calls, lat = ask(
                            f"Send the PDF cheatsheet to {all_emails} with subject_suffix 'Key Concepts Cheatsheet'"
                        )
                    else:
                        result, calls, lat = ask(
                            f"Generate a PDF cheatsheet in {L['lang_tool']} and send to {all_emails}"
                        )
                elif "XVR" in doc_choice or "Scénario" in doc_choice or "Scenario" in doc_choice:
                    result, calls, lat = ask(
                        f"Generate an XVR scenario brief in {L['lang_tool']} and send it to {all_emails} with subject_suffix 'XVR Scenario'"
                    )
                else:
                    result, calls, lat = ask(
                        f"Generate a visual summary in {L['lang_tool']} and send it to {all_emails} with subject_suffix 'Visual Summary'"
                    )

                add_trace("done", "send_gmail", f"to: {email_to} · {lat:.1f}s", latency=lat, badge="ok")
                st.success(result[:120] if result else "Sent!")
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": result,
                    "calls":   calls,
                    "visual":  False,
                })

    # ── Pro / User toggle ─────────────────────────────────────────────────────
    st.markdown(f"<div class='sec-label'>{L['activity'].upper()}</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([3, 1, 3])
    with c1:
        st.markdown(
            f"<div style='font-size:11px;color:#aaa;padding-top:7px;text-align:right'>{L['mode_user']}</div>",
            unsafe_allow_html=True
        )
    with c2:
        tog = st.toggle("", value=st.session_state.pro_mode,
                        key="pro_tog", label_visibility="collapsed")
        if tog != st.session_state.pro_mode:
            st.session_state.pro_mode = tog
            st.rerun()
    with c3:
        st.markdown(
            f"<div style='font-size:11px;color:#aaa;padding-top:7px'>{L['mode_pro']}</div>",
            unsafe_allow_html=True
        )

    # Trace steps
    if st.session_state.trace_steps:
        for step in st.session_state.trace_steps[-8:]:
            if st.session_state.pro_mode:
                lat_str   = f"{step['latency']:.1f}s" if step.get("latency") else ""
                badge_str = f'<span class="badge-ok">{step.get("badge","ok")}</span>' if step["type"] == "done" else ""
                time_str  = f'<span class="badge-time">{lat_str}</span>' if lat_str else ""
                st.markdown(f"""
                <div class="trace-pro">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span><span style="color:#C0392B">tool /</span> {step['label']}</span>
                        <span>{badge_str} {time_str}</span>
                    </div>
                    <div style="color:#bbb;margin-top:2px;font-size:10px">{step.get('detail','')}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                icon    = "✓" if step["type"] == "done" else "›"
                lat_str = f" · {step['latency']:.1f}s" if step.get("latency") else ""
                st.markdown(f"""
                <div class="trace-{'done' if step['type'] == 'done' else 'user'}">
                    {icon} {step['label']}{lat_str}
                </div>
                """, unsafe_allow_html=True)

        if st.button(f"↺ {L['clear']}", use_container_width=True):
            st.session_state.trace_steps = []
            st.rerun()

    # LangSmith link
    if os.getenv("LANGSMITH_API_KEY"):
        st.markdown("""
        <div style="margin-top:12px;padding:8px 10px;background:#f8f8f8;
                    border-radius:8px;border:1px solid #eee">
            <a href="https://smith.langchain.com" target="_blank"
               style="font-size:11px;color:#C0392B;text-decoration:none">
                📊 LangSmith traces →
            </a>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ═══════════════════════════════════════════════════════════════════════════════
L = LABELS[st.session_state.language]

# Header
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:10px 4px 14px;border-bottom:1px solid #f0f0f0;margin-bottom:16px">
    <div style="font-size:14px;color:#999;font-weight:400">
        {st.session_state.video_title if st.session_state.video_loaded else "IncidentIQ"}
    </div>
    <div style="display:flex;gap:8px;align-items:center">
        <span style="font-size:11px;color:#C0392B;background:#FEF0EE;
                     padding:3px 10px;border-radius:20px;border:1px solid #f5c6be">
            gpt-4o-mini
        </span>
        <span style="font-size:11px;color:#2980B9;background:#E8F4FD;
                     padding:3px 10px;border-radius:20px;border:1px solid #b8d4ec">
            Pinecone
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Messages
if not st.session_state.messages:
    st.markdown(f"""
    <div style="text-align:center;padding:80px 20px 40px;color:#bbb">
        <div style="font-size:40px;margin-bottom:16px;opacity:0.25">◈</div>
        <div style="font-size:16px;color:#aaa;margin-bottom:8px;font-weight:500">
            {L['welcome']}
        </div>
        <div style="font-size:13px;color:#ccc">{L['welcome_sub']}</div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="bubble-user">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            content   = msg.get("content", "")
            is_visual = msg.get("visual", False)
            if is_visual:
                try:
                    render_visual_timeline(content)
                except Exception:
                    st.markdown(
                        f'<div class="bubble-agent">{content}</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    f'<div class="bubble-agent">{content}</div>',
                    unsafe_allow_html=True
                )

# Input
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

col_in, col_btn = st.columns([11, 1])
with col_in:
    user_input = st.text_input(
        "",
        placeholder=L["chat_placeholder"],
        label_visibility="collapsed",
        key="main_input",
    )
with col_btn:
    send = st.button("→", use_container_width=True, key="main_send")

# Process input
if (send or user_input) and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    is_url = "youtube.com" in user_input or "youtu.be" in user_input

    with st.spinner(""):
        add_trace("pro", "router", "intent detected")
        if is_url:
            add_trace("pro", "fetch_youtube_transcript", user_input[:50], badge="cached")

        response, tool_calls, latency = ask(user_input)

        for tc in tool_calls:
            add_trace("done", tc, f"{latency:.1f}s", latency=latency, badge="ok")

        # Detect visual summary
        is_visual = False
        try:
            parsed    = json.loads(response)
            is_visual = "timeline" in parsed and "metrics" in parsed
        except Exception:
            pass

        # Update video status
        if is_url and any(w in response.lower() for w in ["loaded","cached","ready","klaar"]):
            st.session_state.video_loaded = True
            try:
                vid = extract_video_id(user_input)
                st.session_state.video_title = f"Video {vid}"
            except Exception:
                st.session_state.video_title = "Video geladen"

        # Store PDF path if generated
        if "File path:" in response:
            pdf_path = response.split("File path: ")[1].split("\n")[0].strip()
            if Path(pdf_path).exists():
                st.session_state.last_pdf_path = pdf_path

        st.session_state.messages.append({
            "role":    "assistant",
            "content": response,
            "calls":   tool_calls,
            "visual":  is_visual,
        })

    st.rerun()
