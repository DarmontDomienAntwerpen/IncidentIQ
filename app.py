"""
IncidentIQ - AI-powered Incident Intelligence
Streamlit application - Light Clean Professional UI
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

# LangChain & LangGraph
from langchain.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pinecone import Pinecone

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# YouTube
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white
from reportlab.pdfgen import canvas as rl_canvas

# Gmail
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IncidentIQ",
    page_icon="🚒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load environment ───────────────────────────────────────────────────────────
load_dotenv()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default Streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #f0f0f0;
    padding: 0;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0;
}

/* Main area */
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Custom scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #fafafa; }
::-webkit-scrollbar-thumb { background: #e0e0e0; border-radius: 2px; }

/* Chat messages */
.chat-msg-user {
    background: #C0392B;
    color: white;
    border-radius: 16px 16px 4px 16px;
    padding: 10px 14px;
    margin: 6px 0 6px 20%;
    font-size: 14px;
    line-height: 1.6;
}
.chat-msg-agent {
    background: #f8f8f8;
    border: 1px solid #f0f0f0;
    color: #1a1a1a;
    border-radius: 4px 16px 16px 16px;
    padding: 10px 14px;
    margin: 6px 20% 6px 0;
    font-size: 14px;
    line-height: 1.6;
}
.chat-msg-sources {
    font-size: 11px;
    color: #C0392B;
    font-family: 'DM Mono', monospace;
    margin-top: 6px;
}

/* Sidebar buttons */
.sidebar-btn {
    width: 100%;
    padding: 10px 14px;
    border-radius: 8px;
    border: 1px solid #f0f0f0;
    background: white;
    color: #333;
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    cursor: pointer;
    text-align: left;
    margin-bottom: 6px;
    transition: all 0.15s;
    display: flex;
    align-items: center;
    gap: 8px;
}
.sidebar-btn:hover {
    background: #fef8f7;
    border-color: #C0392B33;
}
.sidebar-btn.primary {
    background: #C0392B;
    color: white;
    border-color: #C0392B;
}
.sidebar-btn.primary:hover {
    background: #a93226;
}

/* Trace card */
.trace-step-user {
    padding: 8px 12px;
    border-radius: 8px;
    background: #f8f8f8;
    border-left: 3px solid #e0e0e0;
    margin-bottom: 6px;
    font-size: 12px;
    color: #666;
}
.trace-step-done {
    padding: 8px 12px;
    border-radius: 8px;
    background: #f0faf5;
    border-left: 3px solid #1D9E75;
    margin-bottom: 6px;
    font-size: 12px;
    color: #333;
}
.trace-step-pro {
    padding: 8px 12px;
    border-radius: 8px;
    background: #f8f8f8;
    border-left: 3px solid #C0392B;
    margin-bottom: 6px;
    font-size: 11px;
    color: #555;
    font-family: 'DM Mono', monospace;
}
.badge-ok { background: #e1f5ee; color: #0f6e56; padding: 1px 6px; border-radius: 4px; font-size: 10px; }
.badge-cached { background: #e6f1fb; color: #185fa5; padding: 1px 6px; border-radius: 4px; font-size: 10px; }
.badge-time { color: #999; font-size: 10px; }

/* Header bar */
.app-header {
    background: white;
    border-bottom: 1px solid #f0f0f0;
    padding: 14px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
}
</style>
""", unsafe_allow_html=True)

# ── Language strings ───────────────────────────────────────────────────────────
LABELS = {
    "Nederlands": {
        "placeholder": "Stel een vraag of drop een YouTube URL...",
        "btn_pdf": "📄 Key Concepts PDF",
        "btn_visual": "📊 Visuele tijdlijn",
        "btn_xvr": "🎮 XVR Scenario",
        "btn_send": "📤 Verstuur",
        "send_to": "Verstuur naar",
        "send_placeholder": "naam@email.be",
        "add_distrib": "Voeg distributielijst toe",
        "video_loaded": "Video geladen",
        "no_video": "Nog geen video geladen",
        "url_placeholder": "https://youtube.com/watch?v=...",
        "load_btn": "Laden",
        "mode_user": "Gebruiker",
        "mode_pro": "Pro",
        "agent_activity": "Agent activiteit",
        "searching": "Zoeken in de video...",
        "generating": "Genereren...",
        "done": "Klaar",
        "sending": "Versturen...",
        "welcome": "Hallo! Drop een YouTube URL om te beginnen.",
        "doc_type": "Documenttype",
        "doc_options": ["Key Concepts PDF", "Visuele tijdlijn", "XVR Scenario"],
    },
    "English": {
        "placeholder": "Ask a question or drop a YouTube URL...",
        "btn_pdf": "📄 Key Concepts PDF",
        "btn_visual": "📊 Visual Timeline",
        "btn_xvr": "🎮 XVR Scenario",
        "btn_send": "📤 Send",
        "send_to": "Send to",
        "send_placeholder": "name@email.com",
        "add_distrib": "Add distribution list",
        "video_loaded": "Video loaded",
        "no_video": "No video loaded yet",
        "url_placeholder": "https://youtube.com/watch?v=...",
        "load_btn": "Load",
        "mode_user": "User",
        "mode_pro": "Pro",
        "agent_activity": "Agent activity",
        "searching": "Searching the video...",
        "generating": "Generating...",
        "done": "Done",
        "sending": "Sending...",
        "welcome": "Hello! Drop a YouTube URL to get started.",
        "doc_type": "Document type",
        "doc_options": ["Key Concepts PDF", "Visual Timeline", "XVR Scenario"],
    },
    "Français": {
        "placeholder": "Posez une question ou collez une URL YouTube...",
        "btn_pdf": "📄 Concepts clés PDF",
        "btn_visual": "📊 Chronologie visuelle",
        "btn_xvr": "🎮 Scénario XVR",
        "btn_send": "📤 Envoyer",
        "send_to": "Envoyer à",
        "send_placeholder": "nom@email.fr",
        "add_distrib": "Ajouter liste de distribution",
        "video_loaded": "Vidéo chargée",
        "no_video": "Aucune vidéo chargée",
        "url_placeholder": "https://youtube.com/watch?v=...",
        "load_btn": "Charger",
        "mode_user": "Utilisateur",
        "mode_pro": "Pro",
        "agent_activity": "Activité agent",
        "searching": "Recherche dans la vidéo...",
        "generating": "Génération...",
        "done": "Terminé",
        "sending": "Envoi...",
        "welcome": "Bonjour! Collez une URL YouTube pour commencer.",
        "doc_type": "Type de document",
        "doc_options": ["Concepts clés PDF", "Chronologie visuelle", "Scénario XVR"],
    },
}

LANG_TO_TOOL = {
    "Nederlands": "dutch",
    "English": "english",
    "Français": "french",
}

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False
if "video_title" not in st.session_state:
    st.session_state.video_title = ""
if "trace_steps" not in st.session_state:
    st.session_state.trace_steps = []
if "pro_mode" not in st.session_state:
    st.session_state.pro_mode = False
if "language" not in st.session_state:
    st.session_state.language = "Nederlands"
if "agent" not in st.session_state:
    st.session_state.agent = None
if "last_pdf_path" not in st.session_state:
    st.session_state.last_pdf_path = None

# ── Init shared components ─────────────────────────────────────────────────────
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

# ── Helper functions ───────────────────────────────────────────────────────────
def extract_video_id(url):
    if "v=" in url: return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url: return url.split("youtu.be/")[1].split("?")[0]
    raise ValueError(f"Cannot extract video ID: {url}")

def clean_transcript(text):
    text = re.sub(r'\[Music\]|\[Applause\]|\[Laughter\]|\[Cheering\]', '', text)
    text = re.sub(r'\[\d{2}:\d{2}\]\s*(?=\[\d{2}:\d{2}\])', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def add_trace(step_type, label, detail="", latency=None, badge=None):
    st.session_state.trace_steps.append({
        "type": step_type,
        "label": label,
        "detail": detail,
        "latency": latency,
        "badge": badge,
        "time": datetime.now().strftime("%H:%M:%S"),
    })

def get_gmail_service():
    GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
    creds = None
    token_path = Path("token.json")
    creds_path = Path("credentials.json")
    if not creds_path.exists():
        raise FileNotFoundError("credentials.json not found in project root.")
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), GMAIL_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())
    return build("gmail", "v1", credentials=creds)

def render_visual_timeline(json_str):
    """Render the visual timeline from JSON in Streamlit."""
    try:
        data = json.loads(json_str)
    except:
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
    <div style="background:linear-gradient(135deg,#1C2833,#2C3E50);padding:20px 24px;border-radius:12px;margin-bottom:20px;position:relative;overflow:hidden">
        <div style="font-size:10px;letter-spacing:0.1em;color:#C0392B;background:#C0392B18;padding:3px 10px;border-radius:4px;border:1px solid #C0392B33;display:inline-block;margin-bottom:10px">INCIDENT ANALYSIS</div>
        <div style="font-size:20px;font-weight:600;color:white;margin-bottom:4px">{data.get('title','')}</div>
        <div style="font-size:12px;color:#ffffff66">{data.get('subtitle','')} · {data.get('duration','')}</div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics
    metrics = data.get("metrics", [])
    cols = st.columns(len(metrics))
    for i, m in enumerate(metrics):
        col_hex, bg_hex, txt_hex = COLOR_MAP.get(m.get("color","blue"), COLOR_MAP["blue"])
        with cols[i]:
            st.markdown(f"""
            <div style="background:#f8f8f8;border-radius:10px;padding:14px;border-bottom:3px solid {col_hex}">
                <div style="font-size:24px;font-weight:600;color:#1a1a1a;font-family:'DM Mono',monospace">{m.get('value','')}<span style="font-size:12px;color:#999;font-weight:400"> {m.get('unit','')}</span></div>
                <div style="font-size:11px;color:#666;margin-top:4px">{m.get('label','')}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='margin:20px 0 10px;font-size:10px;letter-spacing:0.1em;color:#999;font-weight:500;border-bottom:1px solid #f0f0f0;padding-bottom:8px'>INCIDENT TIJDLIJN</div>", unsafe_allow_html=True)

    # Timeline
    for event in data.get("timeline", []):
        col_hex, bg_hex, txt_hex = COLOR_MAP.get(event.get("color","blue"), COLOR_MAP["blue"])
        badge = event.get("badge","")
        quote = event.get("quote","")
        tags  = event.get("tags",[])
        tags_html = "".join([f'<span style="font-size:10px;padding:2px 8px;border-radius:4px;background:#f5f5f5;color:#888;border:1px solid #eee;margin-right:4px">{t}</span>' for t in tags])
        quote_html = f'<div style="border-left:2px solid {col_hex};padding-left:10px;margin:8px 0;font-size:12px;color:#555;font-style:italic;line-height:1.6">{quote}</div>' if quote else ""

        st.markdown(f"""
        <div style="display:flex;gap:0;margin-bottom:4px">
            <div style="width:52px;flex-shrink:0;padding-top:14px;text-align:right;padding-right:8px">
                <span style="font-size:10px;color:#999;font-family:'DM Mono',monospace">{event.get('timestamp','')}</span>
            </div>
            <div style="width:20px;flex-shrink:0;display:flex;flex-direction:column;align-items:center;padding-top:14px">
                <div style="width:10px;height:10px;border-radius:50%;background:{col_hex};box-shadow:0 0 0 3px {col_hex}22;flex-shrink:0"></div>
                <div style="width:1px;flex:1;background:#f0f0f0;min-height:16px"></div>
            </div>
            <div style="flex:1;padding:8px 0 16px 12px">
                <div style="background:white;border-radius:10px;padding:14px 16px;border:1px solid #f0f0f0;border-left:3px solid {col_hex}">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
                        <div style="font-size:13px;font-weight:600;color:#1a1a1a">{event.get('title','')}</div>
                        <span style="font-size:10px;padding:2px 8px;border-radius:4px;background:{bg_hex};color:{txt_hex};font-weight:500;white-space:nowrap;margin-left:8px">{badge}</span>
                    </div>
                    <div style="font-size:12px;color:#555;line-height:1.7">{event.get('text','')}</div>
                    {quote_html}
                    <div style="margin-top:8px">{tags_html}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Learnings
    st.markdown("<div style='margin:20px 0 10px;font-size:10px;letter-spacing:0.1em;color:#999;font-weight:500;border-bottom:1px solid #f0f0f0;padding-bottom:8px'>KEY LEARNINGS</div>", unsafe_allow_html=True)
    learn_cols = st.columns(2)
    for i, l in enumerate(data.get("learnings",[])):
        with learn_cols[i % 2]:
            st.markdown(f"""
            <div style="background:#f8f8f8;border-radius:10px;padding:14px;margin-bottom:8px">
                <div style="font-size:10px;color:#C0392B;font-family:'DM Mono',monospace;font-weight:600;margin-bottom:6px">{l.get('number','')}</div>
                <div style="font-size:12px;font-weight:600;color:#1a1a1a;margin-bottom:4px">{l.get('title','')}</div>
                <div style="font-size:11px;color:#666;line-height:1.6">{l.get('text','')}</div>
            </div>
            """, unsafe_allow_html=True)

# ── Build agent ────────────────────────────────────────────────────────────────
@st.cache_resource
def build_agent(_llm, _vectorstore, _pc):
    RED    = HexColor("#C0392B")
    DARK   = HexColor("#1C2833")
    ORANGE = HexColor("#E67E22")
    GREEN  = HexColor("#1E8449")
    WHITE  = white
    DISTRIBUTION_LIST = os.getenv("GMAIL_DISTRIBUTION_LIST", "").split(",")

    def extract_kp(context, language="dutch"):
        lang_map = {"dutch":"Dutch","english":"English","french":"French"}
        lang = lang_map.get(language.lower(),"Dutch")
        prompt = (f'Extract structured info for incident cheatsheet in {lang}.\n'
                  f'Return only JSON: {{"title":"...","subtitle":"...","keypoints":["..."],"recommendations":["..."]}}\n\nContext:\n{context}\n\nJSON:')
        raw = re.sub(r'```json|```','',_llm.invoke(prompt).content.strip()).strip()
        return json.loads(raw)

    def gen_pdf(data, source_url=""):
        filepath = f'/tmp/incidentiq_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        c = rl_canvas.Canvas(filepath, pagesize=A4); W,H = A4
        c.setFillColor(RED); c.rect(0,H-3.2*cm,W,3.2*cm,fill=1,stroke=0)
        c.setFillColor(WHITE); c.circle(1.8*cm,H-1.6*cm,0.85*cm,fill=1,stroke=0)
        c.setFillColor(RED); c.setFont("Helvetica-Bold",14)
        c.drawCentredString(1.8*cm,H-1.95*cm,"IQ")
        c.setFillColor(WHITE); c.setFont("Helvetica-Bold",15)
        c.drawString(3.2*cm,H-1.3*cm,data.get("title","IncidentIQ"))
        c.setFont("Helvetica",10); c.drawString(3.2*cm,H-1.85*cm,data.get("subtitle",""))
        c.setFont("Helvetica",8)
        c.drawRightString(W-1.2*cm,H-1.3*cm,datetime.now().strftime("%d/%m/%Y"))
        c.drawRightString(W-1.2*cm,H-1.75*cm,"Generated by IncidentIQ AI")
        c.setFillColor(ORANGE); c.rect(0,H-3.6*cm,W,0.4*cm,fill=1,stroke=0)
        y = H-5.0*cm
        def sh(y,t,col=DARK):
            c.setFillColor(col); c.setFont("Helvetica-Bold",11); c.drawString(1.2*cm,y,t.upper())
            c.setStrokeColor(col); c.setLineWidth(1.5); c.line(1.2*cm,y-0.2*cm,W-1.2*cm,y-0.2*cm)
            return y-0.7*cm
        def bi(y,txt,col=DARK,bc=RED):
            c.setFillColor(bc); c.circle(1.2*cm,y+0.2*cm,0.1*cm,fill=1,stroke=0)
            c.setFillColor(col); c.setFont("Helvetica",9.5); mw=W-1.5*cm-1.2*cm
            words=txt.split(); line,lines="",[]
            for w in words:
                t=line+w+" "
                if c.stringWidth(t,"Helvetica",9.5)<mw: line=t
                else: lines.append(line.strip()); line=w+" "
            lines.append(line.strip())
            for i,l in enumerate(lines): c.drawString(1.5*cm,y-i*0.45*cm,l)
            return y-len(lines)*0.45*cm-0.3*cm
        y=sh(y,"Key Points",RED)
        for kp in data.get("keypoints",[]): y=bi(y,kp)
        y-=0.4*cm
        y=sh(y,"AI-generated recommendations",GREEN)
        c.setFillColor(HexColor("#EAFAF1")); c.setStrokeColor(GREEN); c.setLineWidth(0.8)
        c.roundRect(1.2*cm,y-0.75*cm,W-2.4*cm,0.65*cm,3,fill=1,stroke=1)
        c.setFillColor(GREEN); c.setFont("Helvetica-Bold",8); c.drawString(1.55*cm,y-0.3*cm,"AI analysis:")
        c.setFillColor(DARK); c.setFont("Helvetica-Oblique",8)
        c.drawString(3.0*cm,y-0.3*cm,"Automatically extracted from video - not cited from a person.")
        y-=1.0*cm
        for rec in data.get("recommendations",[]): y=bi(y,rec,bc=GREEN)
        c.setFillColor(RED); c.rect(0,1.2*cm,W,0.15*cm,fill=1,stroke=0)
        c.setFillColor(DARK); c.rect(0,0,W,1.2*cm,fill=1,stroke=0)
        c.setFillColor(WHITE); c.setFont("Helvetica",7.5)
        c.drawString(1.2*cm,0.65*cm,"IncidentIQ - AI-powered Incident Intelligence")
        if source_url: c.drawCentredString(W/2,0.65*cm,f"Source: {source_url[:70]}")
        c.drawRightString(W-1.2*cm,0.65*cm,"Page 1/1"); c.save(); return filepath

    @tool
    def fetch_youtube_transcript(youtube_url: str) -> str:
        """
        Fetch the transcript of a YouTube video and store it in Pinecone cloud.
        Use this tool when the user provides a YouTube URL.
        Checks Pinecone cache first - YouTube is only called once per video ever.
        Returns a confirmation with the number of chunks stored.
        """
        try:
            if "v=" in youtube_url: video_id = youtube_url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in youtube_url: video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
            else: raise ValueError(f"Cannot extract video ID: {youtube_url}")
            index = _pc.Index("incidentiq"); stats = index.describe_index_stats()
            if stats.total_vector_count > 0:
                test = _vectorstore.similarity_search("incident", k=1)
                if test:
                    return f"Video already in Pinecone.\nVideo ID: {video_id}\nUsing cached data.\nReady for Q&A."
            try:
                entries = YouTubeTranscriptApi().fetch(video_id, languages=["en","nl","fr"])
                transcript_list = entries.snippets
            except NoTranscriptFound:
                return f"No transcript found for video {video_id}."
            except TranscriptsDisabled:
                return f"Transcripts disabled for video {video_id}."
            except Exception as e:
                return f"Could not fetch. If YouTube blocks your IP wait 30-60 min.\nError: {str(e)}"
            plain = " ".join(t.text for t in transcript_list)
            timestamped = " ".join(f"[{int(t.start//60):02d}:{int(t.start%60):02d}] {t.text}" for t in transcript_list)
            plain = clean_transcript(plain); timestamped = clean_transcript(timestamped)
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.create_documents(texts=[timestamped], metadatas=[{"video_id":video_id,"source":youtube_url}])
            _vectorstore.add_documents(chunks)
            return f"Transcript loaded.\nVideo ID: {video_id}\nChunks: {len(chunks)}\nReady for Q&A."
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
            english_query = _llm.invoke(f"Translate to English, return only translation: {query}").content.strip()
            rewritten = _llm.invoke(f"Rewrite for incident video search, max 20 words.\nQuery: {english_query}\nRewritten:").content.strip()
            try:
                response = _llm.invoke(f"Generate 3 search query variations. Return JSON list.\nQuestion: {rewritten}\nJSON:").content.strip()
                queries = json.loads(re.sub(r"```json|```","",response).strip())
            except: queries = [rewritten]
            queries.append(rewritten)
            all_docs = {}
            for q in queries:
                for doc in _vectorstore.similarity_search(q, k=4):
                    key = doc.page_content[:100]
                    if key not in all_docs: all_docs[key] = doc
            if not all_docs: return "No relevant information found. Please load a YouTube video first."
            combined = list(all_docs.values())
            all_ts = re.findall(r"\[\d{2}:\d{2}\]"," ".join([d.page_content for d in combined]))
            seen,uts = set(),[]
            for t in all_ts:
                if t not in seen: seen.add(t); uts.append(t)
            clean = [re.sub(r"\[\d{2}:\d{2}\]\s*(?=\[\d{2}:\d{2}\])","",d.page_content) for d in combined]
            return "\n\n".join(clean) + f"\n\nSources: {' | '.join(uts[:5])}"
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def summarize_video(language: str = "english") -> str:
        """
        Generate a structured text summary of the entire loaded video.
        Use this tool when the user asks for a full summary or overview.
        For a visual timeline use generate_visual_summary instead.
        Specify language as english, dutch or french.
        Returns a structured summary with introduction, key points, lessons and conclusion.
        """
        try:
            results = _vectorstore.similarity_search("main topic lessons learned conclusions key points", k=12)
            if not results: return "No video content found."
            context = "\n\n".join([re.sub(r"\[\d{2}:\d{2}\]\s*(?=\[\d{2}:\d{2}\])","",r.page_content) for r in results])
            lang_map = {"english":"English","dutch":"Dutch - natural direct Belgian incident training language","french":"French"}
            lang = lang_map.get(language.lower(),"English")
            prompt = (f"Write a structured summary in {lang}.\n"
                      f"Structure: **Introduction** / **Key Points** / **Lessons Learned** / **Conclusion**\n"
                      f"Rules: bullet points only, max 15 words each.\n\nContext:\n{context}\n\nSummary:")
            return _llm.invoke(prompt).content.strip()
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def generate_pdf_cheatsheet(language: str = "dutch", source_url: str = "") -> str:
        """
        Generate a professional 1-page PDF cheatsheet from the loaded video content.
        Use this tool when the user asks for a cheatsheet, key concepts document or PDF.
        Specify language as dutch, english or french.
        Returns the file path of the generated PDF - pass to send_gmail to email it.
        """
        try:
            results = _vectorstore.similarity_search("key points lessons recommendations conclusions", k=10)
            if not results: return "No video content found."
            context = "\n\n".join([r.page_content for r in results])
            data = extract_kp(context, language)
            filepath = gen_pdf(data, source_url)
            return f"PDF generated.\nFile path: {filepath}\nTitle: {data.get('title','N/A')}"
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def send_gmail(pdf_path: str = "", text_content: str = "", subject_suffix: str = "Document", custom_emails: str = "") -> str:
        """
        Send any generated document to recipients via Gmail.
        Use this tool after any generation tool has created content.
        pdf_path: file path of a generated PDF to attach.
        text_content: text to include in email body.
        subject_suffix: label for subject line.
        custom_emails: comma-separated email addresses to send to.
        Returns a confirmation with the full list of recipients.
        """
        try:
            recipients = [e.strip() for e in DISTRIBUTION_LIST if e.strip()]
            if custom_emails:
                recipients.extend([e.strip() for e in custom_emails.split(",") if e.strip()])
            assert recipients, "No recipients provided."
            service = get_gmail_service()
            subject = f"IncidentIQ - {subject_suffix} - {datetime.now().strftime('%d/%m/%Y')}"
            body = f"Dear colleague,\n\nPlease find the AI-generated {subject_suffix}.\n\n"
            if text_content: body += f"{text_content}\n\n"
            body += "Generated by IncidentIQ AI Agent.\nReview before operational use.\n\nBest regards,\nIncidentIQ AI Agent"
            msg = MIMEMultipart()
            msg["From"]="me"; msg["To"]=", ".join(recipients); msg["Subject"]=subject
            msg.attach(MIMEText(body,"plain"))
            if pdf_path and Path(pdf_path).exists():
                with open(pdf_path,"rb") as f:
                    part = MIMEBase("application","octet-stream"); part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition",f"attachment; filename={Path(pdf_path).name}")
                msg.attach(part)
            message = {"raw": base64.urlsafe_b64encode(msg.as_bytes()).decode()}
            service.users().messages().send(userId="me",body=message).execute()
            return f"Sent.\nSubject: {subject}\nRecipients: {', '.join(recipients)}"
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def generate_xvr_scenario(language: str = "dutch") -> str:
        """
        Generate a structured XVR simulation scenario brief based on the loaded incident video.
        Use this tool when the user asks to create an XVR scenario or clicks Create XVR Scenario.
        Specify language as dutch, english or french.
        Returns a formatted scenario brief for XVR operators.
        """
        try:
            results = _vectorstore.similarity_search(
                "location building fire cause complications decisions resources weather time casualties evacuation", k=12
            )
            if not results: return "No video content found."
            context = "\n\n".join([re.sub(r"\[\d{2}:\d{2}\]\s*(?=\[\d{2}:\d{2}\])","",r.page_content) for r in results])
            lang_map = {"dutch":"Dutch - professional Belgian fire service terminology","english":"English","french":"French"}
            lang = lang_map.get(language.lower(),"Dutch")
            prompt = (
                f"You are an expert XVR simulation scenario designer for emergency services.\n"
                f"Generate a complete XVR operator scenario brief in {lang}.\n\n"
                f"SCENARIO BRIEF - XVR SIMULATION\n================================\n\n"
                f"INCIDENT TITLE:\n[Short descriptive title]\n\n"
                f"LOCATION & BUILDING:\n- Building type: [type]\n- Floors: [number]\n- Construction: [facade, materials]\n\n"
                f"INITIAL SITUATION T+00:00:\n- Fire location: [exact location]\n- Visibility: [smoke, flames]\n- Known casualties: [number]\n- First resources: [vehicles, personnel]\n\n"
                f"ENVIRONMENTAL CONDITIONS:\n- Time of day: [if mentioned]\n- Weather: [if mentioned]\n- Special hazards: [materials, access]\n\n"
                f"SCENARIO COMPLICATIONS (inject in order):\n- T+[time]: [complication 1]\n- T+[time]: [complication 2]\n- T+[time]: [complication 3]\n- T+[time]: [complication 4]\n\n"
                f"CRITICAL DECISION MOMENTS:\n1. [Decision moment]\n2. [Decision moment]\n3. [Decision moment]\n\n"
                f"LEARNING OBJECTIVES:\n- [Objective 1]\n- [Objective 2]\n- [Objective 3]\n\n"
                f"DEBRIEFING QUESTIONS:\n1. [Question based on actual mistakes]\n2. [Question based on actual mistakes]\n3. [Question based on actual mistakes]\n\n"
                f"XVR OPERATOR NOTES:\n[Notes about key moments to inject]\n\n"
                f"Rules: base on context only, realistic timings, never invent details.\n\nContext:\n{context}\n\nScenario brief:"
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
                "timeline events cause complications lessons learned mistakes decisions outcome casualties", k=12
            )
            if not results: return "No video content found."
            context = "\n\n".join([re.sub(r"\[\d{2}:\d{2}\]\s*(?=\[\d{2}:\d{2}\])","",r.page_content) for r in results])
            lang_map = {"dutch":"Dutch - natural direct language","english":"English","french":"French"}
            lang = lang_map.get(language.lower(),"Dutch")
            prompt = (
                f'Extract structured information from this incident video context.\n'
                f'Respond in {lang} with this exact JSON and nothing else:\n'
                f'{{"title":"Short incident title","subtitle":"Presenter and event","duration":"duration or unknown",'
                f'"metrics":[{{"value":"20","unit":"min","label":"Watervertraging","color":"red"}},'
                f'{{"value":"3","unit":"","label":"Kritieke fouten","color":"amber"}},'
                f'{{"value":"16","unit":"","label":"Verdiepingen","color":"blue"}},'
                f'{{"value":"0","unit":"","label":"Slachtoffers","color":"green"}}],'
                f'"timeline":[{{"timestamp":"00:00","title":"Event title","text":"Max 2 sentences.","quote":"","tags":["tag1"],"color":"blue","badge":"Context"}}],'
                f'"learnings":[{{"number":"01","title":"Learning title","text":"Max 2 sentences."}}],'
                f'"source_url":""}}\n\n'
                f'Rules: exactly 4 metrics, 4-6 timeline events, 4 learnings, colors: red/amber/green/blue.\n\nContext:\n{context}\n\nJSON:'
            )
            response = _llm.invoke(prompt)
            raw = re.sub(r"```json|```","",response.content.strip()).strip()
            json.loads(raw)
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
You work for any organization: fire services, police, EMS, civil protection.

Tools available:
- fetch_youtube_transcript: load a YouTube video into the knowledge base
- search_video_knowledge: answer questions about the loaded video
- summarize_video: generate a structured text summary
- generate_pdf_cheatsheet: create a 1-page PDF with key concepts
- send_gmail: send any content by email - PDF, text or both
- generate_xvr_scenario: generate a structured XVR simulation scenario brief
- generate_visual_summary: generate visual timeline summary JSON

HOW TO BEHAVE:
- YouTube URL detected -> call fetch_youtube_transcript
- Question about video -> call search_video_knowledge
- Summary requested -> call summarize_video
- Cheatsheet or PDF requested -> call generate_pdf_cheatsheet
- Send email requested -> call send_gmail with appropriate parameters
- XVR scenario requested -> call generate_xvr_scenario
- Visual summary or timeline requested -> call generate_visual_summary
- Multiple actions -> chain tools in correct order

LANGUAGE RULE:
- Always respond in the same language as the user message
- Dutch -> Dutch | English -> English | French -> French

FORMAT RULE:
- Use bullet points - never long paragraphs
- Max 15 words per bullet
- Confirm clearly what was done after tool calls
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

# ── Build agent ────────────────────────────────────────────────────────────────
if st.session_state.agent is None:
    st.session_state.agent = build_agent(llm, vectorstore, pc)

# ── Ask function ───────────────────────────────────────────────────────────────
def ask(message: str) -> tuple:
    config  = {"configurable": {"thread_id": st.session_state.thread_id}}
    inputs  = {"messages": [HumanMessage(content=message)]}
    start   = time.time()
    final   = ""
    calls   = []
    results = []

    for event in st.session_state.agent.stream(inputs, config=config, stream_mode="values"):
        last = event["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            for tc in last.tool_calls:
                calls.append(tc["name"])
        if hasattr(last, "content") and isinstance(last.content, str) and last.content.strip():
            final = last.content.strip()

    latency = time.time() - start
    return final, calls, latency

# ── Whisper transcription ──────────────────────────────────────────────────────
def transcribe_audio(audio_bytes, lang_code="nl"):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    with open(tmp_path, "rb") as audio_file:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=lang_code,
        )
    Path(tmp_path).unlink(missing_ok=True)
    return transcript.text

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="padding:20px 20px 10px;border-bottom:1px solid #f0f0f0;margin-bottom:16px">
        <div style="display:flex;align-items:center;gap:10px">
            <div style="width:32px;height:32px;background:#C0392B;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;color:white;flex-shrink:0">IQ</div>
            <div>
                <div style="font-size:15px;font-weight:600;color:#1a1a1a">IncidentIQ</div>
                <div style="font-size:10px;color:#999">AI Incident Intelligence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Language selector
    lang = st.selectbox(
        "🌐 Language",
        ["Nederlands", "English", "Français"],
        index=["Nederlands", "English", "Français"].index(st.session_state.language),
        key="lang_select",
    )
    if lang != st.session_state.language:
        st.session_state.language = lang
        st.rerun()

    L = LABELS[st.session_state.language]
    tool_lang = LANG_TO_TOOL[st.session_state.language]

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Video status
    if st.session_state.video_loaded:
        st.markdown(f"""
        <div style="background:#f0faf5;border-radius:8px;padding:10px 12px;margin-bottom:12px;border:1px solid #c8e6c9">
            <div style="font-size:10px;color:#1D9E75;font-weight:500;margin-bottom:3px">● {L['video_loaded']}</div>
            <div style="font-size:11px;color:#333">{st.session_state.video_title[:40]}...</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#f8f8f8;border-radius:8px;padding:10px 12px;margin-bottom:12px;border:1px solid #eee">
            <div style="font-size:11px;color:#999">{L['no_video']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:10px;color:#bbb;margin-bottom:6px;letter-spacing:0.05em'>TOOLS</div>", unsafe_allow_html=True)

    # PDF Cheatsheet
    if st.button(L["btn_pdf"], use_container_width=True):
        if not st.session_state.video_loaded:
            st.warning("Load a video first.")
        else:
            with st.spinner(L["generating"]):
                add_trace("pro", "generate_pdf_cheatsheet", f"lang: {tool_lang}")
                result, calls, lat = ask(f"Maak een key concepts cheatsheet in het {tool_lang}")
                if "File path:" in result:
                    pdf_path = result.split("File path: ")[1].split("\n")[0]
                    st.session_state.last_pdf_path = pdf_path
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download PDF",
                            f,
                            file_name="incidentiq_cheatsheet.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                add_trace("done", L["btn_pdf"], f"{lat:.1f}s", latency=lat, badge="ok")
                st.session_state.messages.append({"role": "assistant", "content": result, "calls": calls})

    # Visual Summary
    if st.button(L["btn_visual"], use_container_width=True):
        if not st.session_state.video_loaded:
            st.warning("Load a video first.")
        else:
            with st.spinner(L["generating"]):
                add_trace("pro", "generate_visual_summary", f"lang: {tool_lang}")
                result, calls, lat = ask(f"Generate a visual timeline summary in {tool_lang}")
                add_trace("done", L["btn_visual"], f"{lat:.1f}s", latency=lat, badge="ok")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result,
                    "calls": calls,
                    "visual": True,
                })
                st.rerun()

    # XVR Scenario
    if st.button(L["btn_xvr"], use_container_width=True):
        if not st.session_state.video_loaded:
            st.warning("Load a video first.")
        else:
            with st.spinner(L["generating"]):
                add_trace("pro", "generate_xvr_scenario", f"lang: {tool_lang}")
                result, calls, lat = ask(f"Generate an XVR scenario brief in {tool_lang}")
                add_trace("done", L["btn_xvr"], f"{lat:.1f}s", latency=lat, badge="ok")
                st.session_state.messages.append({"role": "assistant", "content": result, "calls": calls})
                st.rerun()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:10px;color:#bbb;margin-bottom:6px;letter-spacing:0.05em'>{L['send_to'].upper()}</div>", unsafe_allow_html=True)

    # Email input
    email_input = st.text_input(
        "",
        placeholder=L["send_placeholder"],
        label_visibility="collapsed",
        key="email_input",
    )
    add_distrib = st.checkbox(L["add_distrib"], value=False)

    # Document type selector
    doc_type = st.selectbox(
        L["doc_type"],
        L["doc_options"],
        key="doc_type_select",
        label_visibility="collapsed",
    )

    if st.button(L["btn_send"], use_container_width=True, type="primary"):
        if not email_input and not add_distrib:
            st.warning("Enter at least one email address.")
        else:
            distrib = os.getenv("GMAIL_DISTRIBUTION_LIST","") if add_distrib else ""
            with st.spinner(L["sending"]):
                add_trace("pro", "send_gmail", f"to: {email_input}")
                if "PDF" in doc_type and st.session_state.last_pdf_path:
                    result, calls, lat = ask(
                        f"Send the PDF cheatsheet to {email_input}"
                        + (f" and {distrib}" if distrib else "")
                    )
                elif "XVR" in doc_type or "Scénario" in doc_type or "Scenario" in doc_type:
                    result, calls, lat = ask(
                        f"Generate an XVR scenario and send to {email_input}"
                        + (f" and {distrib}" if distrib else "")
                    )
                else:
                    result, calls, lat = ask(
                        f"Generate a visual summary and send to {email_input}"
                        + (f" and {distrib}" if distrib else "")
                    )
                add_trace("done", L["btn_send"], f"to: {email_input} · {lat:.1f}s", latency=lat, badge="ok")
                st.success(result[:100])

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Pro/User toggle
    st.markdown(f"<div style='font-size:10px;color:#bbb;margin-bottom:6px;letter-spacing:0.05em'>{L['agent_activity'].upper()}</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.markdown(f"<div style='font-size:11px;color:#999;padding-top:6px'>{L['mode_user']}</div>", unsafe_allow_html=True)
    with col2:
        pro_toggle = st.toggle("", value=st.session_state.pro_mode, key="pro_toggle", label_visibility="collapsed")
        if pro_toggle != st.session_state.pro_mode:
            st.session_state.pro_mode = pro_toggle
            st.rerun()
    with col3:
        st.markdown(f"<div style='font-size:11px;color:#999;padding-top:6px'>{L['mode_pro']}</div>", unsafe_allow_html=True)

    # Trace display
    if st.session_state.trace_steps:
        st.markdown("<div style='margin-top:8px'>", unsafe_allow_html=True)
        for step in st.session_state.trace_steps[-6:]:
            if st.session_state.pro_mode:
                badge_html = f'<span class="badge-ok">{step.get("badge","ok")}</span>' if step["type"] == "done" else ""
                time_html  = f'<span class="badge-time">{step.get("latency", 0):.1f}s</span>' if step.get("latency") else ""
                st.markdown(f"""
                <div class="trace-step-pro">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="color:#C0392B">tool /</span> {step['label']} {badge_html} {time_html}
                    </div>
                    <div style="color:#aaa;margin-top:2px">{step.get('detail','')}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                icon = "✓" if step["type"] == "done" else "→"
                st.markdown(f"""
                <div class="trace-step-{'done' if step['type'] == 'done' else 'user'}">
                    {icon} {step['label']}
                    {f'<span style="color:#999;font-size:10px"> · {step.get(\"latency\",0):.1f}s</span>' if step.get('latency') else ''}
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Clear trace
    if st.session_state.trace_steps:
        if st.button("↺ Clear", use_container_width=True):
            st.session_state.trace_steps = []
            st.rerun()

# ── MAIN CHAT AREA ────────────────────────────────────────────────────────────
L = LABELS[st.session_state.language]

# Header
st.markdown(f"""
<div style="background:white;border-bottom:1px solid #f0f0f0;padding:14px 24px;display:flex;align-items:center;justify-content:space-between">
    <div style="font-size:14px;color:#999">
        {st.session_state.video_title if st.session_state.video_loaded else "IncidentIQ"}
    </div>
    <div style="font-size:11px;color:#C0392B;background:#FEF0EE;padding:3px 10px;border-radius:20px;border:1px solid #f5c6be">
        AI-powered · gpt-4o-mini · Pinecone
    </div>
</div>
""", unsafe_allow_html=True)

# Chat messages
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.markdown(f"""
        <div style="text-align:center;padding:60px 20px;color:#bbb">
            <div style="font-size:48px;margin-bottom:16px;opacity:0.3">🚒</div>
            <div style="font-size:15px;color:#999;margin-bottom:8px">{L['welcome']}</div>
            <div style="font-size:12px;color:#bbb">Drop a YouTube URL · Ask questions · Generate reports</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                content = msg["content"]
                is_visual = msg.get("visual", False)
                if is_visual:
                    try:
                        render_visual_timeline(content)
                    except:
                        st.markdown(f'<div class="chat-msg-agent">{content}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-msg-agent">{content}</div>', unsafe_allow_html=True)

# ── INPUT AREA ────────────────────────────────────────────────────────────────
st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

col_input, col_mic, col_send = st.columns([10, 1, 1])

with col_input:
    user_input = st.text_input(
        "",
        placeholder=L["placeholder"],
        label_visibility="collapsed",
        key="chat_input",
    )

with col_mic:
    audio = st.audio_input("", label_visibility="collapsed", key="mic_input")

with col_send:
    send_btn = st.button("→", use_container_width=True, key="send_btn")

# Handle voice input
if audio:
    lang_codes = {"Nederlands": "nl", "English": "en", "Français": "fr"}
    with st.spinner("Transcribing..."):
        transcribed = transcribe_audio(audio.read(), lang_code=lang_codes[st.session_state.language])
    if transcribed:
        user_input = transcribed
        st.info(f"🎤 {transcribed}")

# Handle text or voice submission
if (send_btn or user_input) and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Check if YouTube URL
    is_url = "youtube.com" in user_input or "youtu.be" in user_input

    with st.spinner(""):
        add_trace("pro", "router", f"intent detected")
        if is_url:
            add_trace("pro", "fetch_youtube_transcript", user_input[:40], badge="cached")

        response, tool_calls, latency = ask(user_input)

        for tc in tool_calls:
            add_trace("done", tc, f"{latency:.1f}s", latency=latency, badge="ok")

        # Check if response is visual summary JSON
        is_visual = False
        try:
            parsed = json.loads(response)
            if "timeline" in parsed and "metrics" in parsed:
                is_visual = True
        except:
            pass

        if is_url and ("loaded" in response.lower() or "cached" in response.lower() or "ready" in response.lower()):
            st.session_state.video_loaded = True
            try:
                video_id = extract_video_id(user_input)
                st.session_state.video_title = f"Video {video_id}"
            except:
                pass

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "calls": tool_calls,
            "visual": is_visual,
        })

    st.rerun()
