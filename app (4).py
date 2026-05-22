"""
IncidentIQ - AI-powered Incident Intelligence
Full custom HTML UI via Streamlit components
"""

import os, re, json, time, base64, uuid
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import streamlit as st
import streamlit.components.v1 as components

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

st.set_page_config(page_title="IncidentIQ", page_icon="🔴", layout="wide")
st.markdown("<style>#MainMenu,footer,header{visibility:hidden}.block-container{padding:0!important;max-width:100%!important}</style>", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
defaults = {
    "messages": [],
    "thread_id": f"s_{uuid.uuid4().hex[:8]}",
    "video_loaded": False,
    "video_title": "",
    "video_url": "",
    "trace_steps": [],
    "pro_mode": False,
    "language": "Nederlands",
    "agent": None,
    "last_pdf_path": None,
    "last_pdf_bytes": None,
    "active_tab": "chat",
    "xvr_content": "",
    "visual_content": "",
    "pdf_title": "",
    "last_message": "",
    "run_id": "",
    "total_tokens": 0,
    "total_cost": 0.0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Components ─────────────────────────────────────────────────────────────────
@st.cache_resource
def init_components():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    pc  = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    vs  = PineconeVectorStore(
        index_name="incidentiq",
        embedding=emb,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    )
    return llm, emb, pc, vs

llm, emb, pc, vs = init_components()

# ── Helpers ────────────────────────────────────────────────────────────────────
def vid_id(url):
    if "v=" in url: return url.split("v=")[1].split("&")[0]
    if "youtu.be/" in url: return url.split("youtu.be/")[1].split("?")[0]
    raise ValueError("Cannot extract video ID")

def clean_tx(text):
    text = re.sub(r'\[Music\]|\[Applause\]|\[Laughter\]|\[Cheering\]', '', text)
    text = re.sub(r'\[\d{2}:\d{2}\]\s*(?=\[\d{2}:\d{2}\])', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def add_trace(step_type, label, detail="", latency=None, tokens=0, cost=0.0, badge=None, extra={}):
    st.session_state.trace_steps.append({
        "type": step_type, "label": label, "detail": detail,
        "latency": latency, "tokens": tokens, "cost": cost,
        "badge": badge, "extra": extra,
        "time": datetime.now().strftime("%H:%M:%S.%f")[:11],
    })
    if tokens: st.session_state.total_tokens += tokens
    if cost:   st.session_state.total_cost   += cost

def lang_tool():
    return {"Nederlands": "dutch", "English": "english", "Français": "french"}[st.session_state.language]

def get_gmail():
    SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
    creds  = None
    tp, cp = Path("token.json"), Path("credentials.json")
    if not cp.exists(): raise FileNotFoundError("credentials.json not found.")
    if tp.exists(): creds = Credentials.from_authorized_user_file(str(tp), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token: creds.refresh(Request())
        else:
            flow  = InstalledAppFlow.from_client_secrets_file(str(cp), SCOPES)
            creds = flow.run_local_server(port=0)
        tp.write_text(creds.to_json())
    return build("gmail", "v1", credentials=creds)

# ── YouTube loader (direct, no agent) ─────────────────────────────────────────
def load_video(url):
    L = LABELS[st.session_state.language]
    try:
        video_id = vid_id(url)
    except Exception as e:
        return f"Cannot extract video ID: {e}"

    t0    = time.time()
    index = pc.Index("incidentiq")
    stats = index.describe_index_stats()

    if stats.total_vector_count > 0:
        test = vs.similarity_search("incident", k=1)
        if test:
            st.session_state.video_loaded = True
            st.session_state.video_title  = f"Video {video_id}"
            st.session_state.video_url    = url
            lat = time.time() - t0
            add_trace("done", "fetch_youtube_transcript",
                      f"video_id: {video_id} · pinecone_hit: true · no_youtube_request",
                      latency=lat, tokens=0, cost=0.0, badge="cached",
                      extra={"video_id": video_id, "chunks": stats.total_vector_count})
            return L["cached_msg"]

    try:
        entries = YouTubeTranscriptApi().fetch(video_id, languages=["en","nl","fr"])
        txlist  = entries.snippets
    except NoTranscriptFound:
        return f"No transcript found for {video_id}. Enable CC subtitles."
    except TranscriptsDisabled:
        return f"Transcripts disabled for {video_id}."
    except Exception as e:
        return f"YouTube blocked. Wait 30-60 min.\nError: {e}"

    plain = clean_tx(" ".join(t.text for t in txlist))
    ts    = clean_tx(" ".join(f"[{int(t.start//60):02d}:{int(t.start%60):02d}] {t.text}" for t in txlist))
    spl   = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = spl.create_documents(texts=[ts], metadatas=[{"video_id": video_id, "source": url}])
    vs.add_documents(chunks)

    st.session_state.video_loaded = True
    st.session_state.video_title  = f"Video {video_id}"
    st.session_state.video_url    = url
    lat = time.time() - t0
    add_trace("done", "fetch_youtube_transcript",
              f"video_id: {video_id} · chars: {len(plain):,} · chunks: {len(chunks)} · youtube_fetch: true",
              latency=lat, tokens=0, cost=0.0, badge="ok",
              extra={"video_id": video_id, "chunks": len(chunks), "chars": len(plain)})
    return L["loaded_msg"] + f" ({len(chunks)} chunks)"

# ── PDF generator ──────────────────────────────────────────────────────────────
def make_pdf(context, language="dutch", source_url=""):
    RED, DARK, ORANGE, GREEN, WHITE = HexColor("#C0392B"), HexColor("#1C2833"), HexColor("#E67E22"), HexColor("#1E8449"), white
    lm = {"dutch":"Dutch","english":"English","french":"French"}
    lg = lm.get(language,"Dutch")
    p  = (f'Extract structured info for incident cheatsheet in {lg}.\n'
          f'Return only JSON: {{"title":"...","subtitle":"...","keypoints":["..."],"recommendations":["..."]}}\n'
          f'Rules: max 12 words per item, no timestamps.\n\nContext:\n{context}\n\nJSON:')
    raw  = re.sub(r'```json|```','', llm.invoke(p).content.strip()).strip()
    data = json.loads(raw)
    fp   = f'/tmp/iq_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    c    = rl_canvas.Canvas(fp, pagesize=A4); W,H = A4
    c.setFillColor(RED); c.rect(0,H-3.2*cm,W,3.2*cm,fill=1,stroke=0)
    c.setFillColor(WHITE); c.circle(1.8*cm,H-1.6*cm,0.85*cm,fill=1,stroke=0)
    c.setFillColor(RED); c.setFont("Helvetica-Bold",14); c.drawCentredString(1.8*cm,H-1.95*cm,"IQ")
    c.setFillColor(WHITE); c.setFont("Helvetica-Bold",15); c.drawString(3.2*cm,H-1.3*cm,data.get("title","IncidentIQ"))
    c.setFont("Helvetica",10); c.drawString(3.2*cm,H-1.85*cm,data.get("subtitle",""))
    c.setFont("Helvetica",8); c.drawRightString(W-1.2*cm,H-1.3*cm,datetime.now().strftime("%d/%m/%Y"))
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
    y-=0.4*cm; y=sh(y,"AI-generated recommendations",GREEN)
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
    c.drawRightString(W-1.2*cm,0.65*cm,"Page 1/1"); c.save()
    return fp, data.get("title","Cheatsheet")

# ── Language labels ────────────────────────────────────────────────────────────
LABELS = {
    "Nederlands": {
        "placeholder": "Stel een vraag of plak een YouTube URL...",
        "btn_pdf":     "Key Concepts PDF",
        "btn_visual":  "Visuele tijdlijn",
        "btn_xvr":     "XVR Scenario",
        "send_to":     "Verstuur naar",
        "send_ph":     "naam@email.be",
        "send_btn":    "Versturen",
        "no_video":    "Geen video geladen",
        "video_ok":    "Video geladen",
        "welcome":     "Drop een YouTube URL om te beginnen.",
        "welcome_sub": "Stel vragen · Genereer rapporten · Maak XVR scenario's",
        "mode_user":   "Gebruiker",
        "mode_pro":    "Pro",
        "activity":    "Agent activiteit",
        "generating":  "Genereren...",
        "clear":       "Wissen",
        "loaded_msg":  "Video geladen en klaar voor vragen!",
        "cached_msg":  "Video was al geladen — data uit Pinecone cache.",
        "tab_chat":    "Chat",
        "tab_pdf":     "Key Concepts",
        "tab_tl":      "Tijdlijn",
        "tab_xvr":     "XVR Scenario",
    },
    "English": {
        "placeholder": "Ask a question or paste a YouTube URL...",
        "btn_pdf":     "Key Concepts PDF",
        "btn_visual":  "Visual Timeline",
        "btn_xvr":     "XVR Scenario",
        "send_to":     "Send to",
        "send_ph":     "name@email.com",
        "send_btn":    "Send",
        "no_video":    "No video loaded",
        "video_ok":    "Video loaded",
        "welcome":     "Drop a YouTube URL to get started.",
        "welcome_sub": "Ask questions · Generate reports · Create XVR scenarios",
        "mode_user":   "User",
        "mode_pro":    "Pro",
        "activity":    "Agent activity",
        "generating":  "Generating...",
        "clear":       "Clear",
        "loaded_msg":  "Video loaded and ready for questions!",
        "cached_msg":  "Video was already loaded — using Pinecone cache.",
        "tab_chat":    "Chat",
        "tab_pdf":     "Key Concepts",
        "tab_tl":      "Timeline",
        "tab_xvr":     "XVR Scenario",
    },
    "Français": {
        "placeholder": "Posez une question ou collez une URL YouTube...",
        "btn_pdf":     "Concepts clés PDF",
        "btn_visual":  "Chronologie visuelle",
        "btn_xvr":     "Scénario XVR",
        "send_to":     "Envoyer à",
        "send_ph":     "nom@email.fr",
        "send_btn":    "Envoyer",
        "no_video":    "Aucune vidéo chargée",
        "video_ok":    "Vidéo chargée",
        "welcome":     "Collez une URL YouTube pour commencer.",
        "welcome_sub": "Posez des questions · Générez des rapports · Créez des scénarios XVR",
        "mode_user":   "Utilisateur",
        "mode_pro":    "Pro",
        "activity":    "Activité agent",
        "generating":  "Génération...",
        "clear":       "Effacer",
        "loaded_msg":  "Vidéo chargée et prête pour les questions!",
        "cached_msg":  "Vidéo déjà chargée — données depuis le cache Pinecone.",
        "tab_chat":    "Chat",
        "tab_pdf":     "Concepts clés",
        "tab_tl":      "Chronologie",
        "tab_xvr":     "Scénario XVR",
    },
}

# ── Build agent ────────────────────────────────────────────────────────────────
@st.cache_resource
def build_agent(_llm, _vs):

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
            eq = _llm.invoke(f"Translate to English, return only translation: {query}").content.strip()
            rw = _llm.invoke(f"Rewrite for incident video search, max 20 words.\nQuery: {eq}\nRewritten:").content.strip()
            try:
                qs = json.loads(re.sub(r'```json|```','',
                    _llm.invoke(f"Generate 3 search query variations. Return JSON list.\nQuestion: {rw}\nJSON:").content.strip()).strip())
            except Exception:
                qs = [rw]
            qs.append(rw)
            all_docs = {}
            for q in qs:
                for doc in _vs.similarity_search(q, k=4):
                    key = doc.page_content[:100]
                    if key not in all_docs: all_docs[key] = doc
            if not all_docs: return "No relevant information found."
            combined = list(all_docs.values())
            all_ts = re.findall(r"\[\d{2}:\d{2}\]", " ".join([d.page_content for d in combined]))
            seen, uts = set(), []
            for t in all_ts:
                if t not in seen: seen.add(t); uts.append(t)
            clean = [re.sub(r"\[\d{2}:\d{2}\]\s*(?=\[\d{2}:\d{2}\])","",d.page_content) for d in combined]
            return "\n\n".join(clean) + f"\n\nSources: {' | '.join(uts[:5])}\nChunks: {len(clean)} via {len(qs)} queries"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def summarize_video(language: str = "english") -> str:
        """
        Generate a structured text summary of the entire loaded video.
        Use this tool when the user asks for a full summary or overview.
        Specify language as english, dutch or french.
        Returns structured summary with introduction, key points, lessons and conclusion.
        """
        try:
            results = _vs.similarity_search("main topic lessons learned conclusions key points", k=12)
            if not results: return "No video content found."
            context = "\n\n".join([re.sub(r"\[\d{2}:\d{2}\]\s*(?=\[\d{2}:\d{2}\])","",r.page_content) for r in results])
            lm = {"english":"English","dutch":"Dutch - natural direct Belgian incident training language","french":"French"}
            lang = lm.get(language.lower(),"English")
            return _llm.invoke(
                f"Write a structured summary in {lang}.\n"
                f"Structure: **Introduction** / **Key Points** / **Lessons Learned** / **Conclusion**\n"
                f"Rules: bullet points only, max 15 words each.\n\nContext:\n{context}\n\nSummary:"
            ).content.strip()
        except Exception as e:
            return f"Error: {e}"

    @tool
    def generate_xvr_scenario(language: str = "dutch") -> str:
        """
        Generate a structured XVR simulation scenario brief based on the loaded incident video.
        Use this tool when the user asks to create an XVR scenario.
        Specify language as dutch, english or french.
        Returns formatted scenario brief for XVR operators.
        """
        try:
            results = _vs.similarity_search(
                "location building fire cause complications decisions resources weather time casualties evacuation", k=12)
            if not results: return "No video content found."
            context = "\n\n".join([re.sub(r"\[\d{2}:\d{2}\]\s*(?=\[\d{2}:\d{2}\])","",r.page_content) for r in results])
            lm = {"dutch":"Dutch - professional Belgian fire service terminology","english":"English","french":"French"}
            lang = lm.get(language.lower(),"Dutch")
            return _llm.invoke(
                f"You are an expert XVR simulation scenario designer.\n"
                f"Generate a complete XVR operator scenario brief in {lang}.\n\n"
                f"SCENARIO BRIEF - XVR SIMULATION\n================================\n\n"
                f"INCIDENT TITLE:\n[Short descriptive title]\n\n"
                f"LOCATION & BUILDING:\n- Building type: [type]\n- Floors: [number]\n- Construction: [facade, materials]\n\n"
                f"INITIAL SITUATION T+00:00:\n- Fire location: [exact location]\n- Visibility: [smoke, flames]\n"
                f"- Known casualties: [number]\n- First resources: [vehicles, personnel]\n\n"
                f"ENVIRONMENTAL CONDITIONS:\n- Time of day: [if mentioned]\n- Weather: [if mentioned]\n- Special hazards: [materials, access]\n\n"
                f"SCENARIO COMPLICATIONS (inject in order):\n- T+[time]: [complication 1]\n- T+[time]: [complication 2]\n"
                f"- T+[time]: [complication 3]\n- T+[time]: [complication 4]\n\n"
                f"CRITICAL DECISION MOMENTS:\n1. [Decision moment]\n2. [Decision moment]\n3. [Decision moment]\n\n"
                f"LEARNING OBJECTIVES:\n- [Objective 1]\n- [Objective 2]\n- [Objective 3]\n\n"
                f"DEBRIEFING QUESTIONS:\n1. [Question based on actual mistakes]\n2. [Question]\n3. [Question]\n\n"
                f"XVR OPERATOR NOTES:\n[Notes about key moments to inject]\n\n"
                f"Rules: base on context only, realistic timings, never invent details.\n\nContext:\n{context}\n\nScenario brief:"
            ).content.strip()
        except Exception as e:
            return f"Error: {e}"

    @tool
    def generate_visual_summary(language: str = "dutch") -> str:
        """
        Generate structured JSON data for visual timeline rendering in the app.
        Use this tool when the user asks for a visual summary or timeline view.
        Specify language as dutch, english or french.
        Returns valid JSON with metrics, timeline events and key learnings.
        """
        try:
            results = _vs.similarity_search(
                "timeline events cause complications lessons learned mistakes decisions outcome casualties", k=12)
            if not results: return "No video content found."
            context = "\n\n".join([re.sub(r"\[\d{2}:\d{2}\]\s*(?=\[\d{2}:\d{2}\])","",r.page_content) for r in results])
            lm = {"dutch":"Dutch - natural direct language","english":"English","french":"French"}
            lang = lm.get(language.lower(),"Dutch")
            prompt = (
                f'Extract structured information from this incident video context.\n'
                f'Respond in {lang} with ONLY this JSON structure:\n'
                f'{{"title":"Short incident title","subtitle":"Presenter and event",'
                f'"duration":"duration or unknown",'
                f'"metrics":[{{"value":"20","unit":"min","label":"Watervertraging","color":"red"}},'
                f'{{"value":"3","unit":"","label":"Kritieke fouten","color":"amber"}},'
                f'{{"value":"16","unit":"","label":"Verdiepingen","color":"blue"}},'
                f'{{"value":"0","unit":"","label":"Slachtoffers","color":"green"}}],'
                f'"timeline":[{{"timestamp":"00:00","title":"Event title","text":"Max 2 sentences.",'
                f'"quote":"direct quote or empty","tags":["tag1","tag2"],"color":"blue","badge":"Context"}}],'
                f'"learnings":[{{"number":"01","title":"Learning title","text":"Max 2 sentences."}}],'
                f'"source_url":""}}\n\n'
                f'Rules: exactly 4 metrics, 4-6 timeline events, 4 learnings, '
                f'colors only: red/amber/green/blue, all text in {lang}, no markdown.\n\n'
                f'Context:\n{context}\n\nJSON:'
            )
            raw = re.sub(r'```json|```','', _llm.invoke(prompt).content.strip()).strip()
            json.loads(raw)
            return raw
        except Exception as e:
            return f"Error: {e}"

    @tool
    def send_gmail(pdf_path: str = "", text_content: str = "", subject_suffix: str = "Document", custom_emails: str = "") -> str:
        """
        Send any generated document to recipients via Gmail.
        pdf_path: file path of generated PDF to attach.
        text_content: text to include in email body.
        subject_suffix: label for subject line e.g. Key Concepts, XVR Scenario.
        custom_emails: comma-separated email addresses.
        Returns confirmation with recipient list.
        """
        try:
            DIST = os.getenv("GMAIL_DISTRIBUTION_LIST","").split(",")
            recipients = [e.strip() for e in DIST if e.strip()]
            if custom_emails:
                recipients.extend([e.strip() for e in custom_emails.split(",") if e.strip()])
            if not recipients: return "No recipients provided."
            svc = get_gmail()
            subject = f"IncidentIQ - {subject_suffix} - {datetime.now().strftime('%d/%m/%Y')}"
            body = f"Dear colleague,\n\nPlease find the AI-generated {subject_suffix}.\n\n"
            if text_content: body += f"{text_content}\n\n"
            body += "Generated by IncidentIQ AI Agent.\nReview before operational use.\n\nBest regards,\nIncidentIQ"
            msg = MIMEMultipart()
            msg["From"]="me"; msg["To"]=", ".join(recipients); msg["Subject"]=subject
            msg.attach(MIMEText(body,"plain"))
            if pdf_path and Path(pdf_path).exists():
                with open(pdf_path,"rb") as f:
                    part = MIMEBase("application","octet-stream"); part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition",f"attachment; filename={Path(pdf_path).name}")
                msg.attach(part)
            svc.users().messages().send(userId="me",body={"raw":base64.urlsafe_b64encode(msg.as_bytes()).decode()}).execute()
            return f"Sent to: {', '.join(recipients)}"
        except Exception as e:
            return f"Error: {e}"

    TOOLS = [search_video_knowledge, summarize_video, generate_xvr_scenario, generate_visual_summary, send_gmail]
    PROMPT = """You are IncidentIQ, an AI agent for incident training and knowledge extraction.
Sector-agnostic: fire services, police, EMS, civil protection or any training context.
Video loading is handled separately — never ask the user to load a video.

Tools:
- search_video_knowledge: answer questions about the video
- summarize_video: generate text summary
- generate_xvr_scenario: create XVR simulation scenario brief
- generate_visual_summary: generate visual timeline JSON
- send_gmail: send content by email

ROUTING:
- Question -> search_video_knowledge
- Summary -> summarize_video
- XVR -> generate_xvr_scenario
- Visual/timeline -> generate_visual_summary
- Email -> send_gmail

LANGUAGE: Always respond in the SAME language as the user message. Never mix languages.
Dutch question = Dutch answer. English question = English answer. French question = French answer.
FORMAT: Bullet points, max 15 words per bullet.
"""
    lw = _llm.bind_tools(TOOLS)
    def agent_node(state: MessagesState):
        return {"messages": [lw.invoke([SystemMessage(content=PROMPT)] + state["messages"])]}
    b = StateGraph(MessagesState)
    b.add_node("agent", agent_node)
    b.add_node("tools", ToolNode(TOOLS))
    b.add_edge(START, "agent")
    b.add_conditional_edges("agent", tools_condition)
    b.add_edge("tools", "agent")
    return b.compile(checkpointer=MemorySaver())

if st.session_state.agent is None:
    with st.spinner("Loading IncidentIQ..."):
        st.session_state.agent = build_agent(llm, vs)

# ── Ask ────────────────────────────────────────────────────────────────────────
def ask(message):
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    t0 = time.time()
    final, calls = "", []
    run_id = uuid.uuid4().hex[:8]
    st.session_state.run_id = run_id
    for event in st.session_state.agent.stream({"messages": [HumanMessage(content=message)]}, config=config, stream_mode="values"):
        last = event["messages"][-1]
        if hasattr(last,"tool_calls") and last.tool_calls:
            for tc in last.tool_calls: calls.append(tc["name"])
        if hasattr(last,"content") and isinstance(last.content,str) and last.content.strip():
            final = last.content.strip()
    lat = time.time() - t0
    tokens = len(message.split()) * 2 + len(final.split()) * 2
    cost   = tokens * 0.00000015
    for tc in calls:
        add_trace("done", tc, f"lang: {lang_tool()}", latency=lat/len(calls) if calls else lat,
                  tokens=tokens//len(calls) if calls else tokens,
                  cost=cost//len(calls) if calls else cost, badge="ok",
                  extra={"run_id": run_id})
    return final, calls, lat

# ── Trace HTML ────────────────────────────────────────────────────────────────
def render_trace_html():
    L = LABELS[st.session_state.language]
    steps = st.session_state.trace_steps[-10:]
    pro   = st.session_state.pro_mode

    if not steps:
        return f"""<div style="font-size:11px;color:#bbb;text-align:center;padding:12px">
            {L['activity']}...</div>"""

    total_tokens = st.session_state.total_tokens
    total_cost   = st.session_state.total_cost
    run_id       = st.session_state.run_id

    html = ""
    if pro and run_id:
        html += f"""<div style="font-size:9px;color:#888;font-family:monospace;
            padding:6px 8px;background:#f8f8f8;border-radius:5px;margin-bottom:8px;
            border:0.5px solid #eee;line-height:1.6">
            RUN {run_id} · thread: {st.session_state.thread_id}<br>
            model: gpt-4o-mini · embed: text-embedding-3-small<br>
            index: incidentiq · retriever_k: 8
        </div>"""

    for i, step in enumerate(steps):
        is_last = i == len(steps) - 1
        dot_color = "#1D9E75" if step["type"] == "done" else "#E67E22"
        badge_html = ""
        if step.get("badge") == "ok":
            badge_html = '<span style="background:#e1f5ee;color:#0f6e56;padding:1px 6px;border-radius:3px;font-size:9px;float:right">ok</span>'
        elif step.get("badge") == "cached":
            badge_html = '<span style="background:#e6f1fb;color:#185fa5;padding:1px 6px;border-radius:3px;font-size:9px;float:right">cached</span>'

        if pro:
            lat_str    = f" · {step['latency']:.2f}s" if step.get("latency") else ""
            tok_str    = f" · {step['tokens']} tokens" if step.get("tokens") else ""
            cost_str   = f" · ${step['cost']:.6f}" if step.get("cost") else ""
            extra_html = ""
            for k, v in step.get("extra", {}).items():
                extra_html += f'<span style="color:#aaa">{k}:</span> {v} &nbsp;'

            html += f"""<div style="display:flex;gap:6px;margin-bottom:6px">
                <div style="display:flex;flex-direction:column;align-items:center;width:10px;flex-shrink:0">
                    <div style="width:8px;height:8px;border-radius:50%;background:{dot_color};margin-top:3px;flex-shrink:0"></div>
                    {"" if is_last else '<div style="width:1px;flex:1;background:#eee;margin-top:3px"></div>'}
                </div>
                <div style="flex:1;background:#fafafa;border-radius:5px;padding:6px 8px;border:0.5px solid #eee;border-left:2px solid {dot_color}">
                    <div style="font-size:10px;font-family:monospace;color:#333">
                        {badge_html}
                        <span style="color:#C0392B">tool /</span> {step["label"]}
                        <span style="color:#aaa">{lat_str}{tok_str}{cost_str}</span>
                    </div>
                    <div style="font-size:9px;color:#888;font-family:monospace;margin-top:2px">{step.get("detail","")}</div>
                    {f'<div style="font-size:9px;color:#bbb;font-family:monospace;margin-top:2px">{extra_html}</div>' if extra_html else ""}
                </div>
            </div>"""
        else:
            icons = {
                "fetch_youtube_transcript": "Video laden",
                "search_video_knowledge":   "Zoeken in de video",
                "summarize_video":          "Samenvatting maken",
                "generate_pdf_cheatsheet":  "PDF aanmaken",
                "generate_xvr_scenario":    "XVR scenario maken",
                "generate_visual_summary":  "Tijdlijn genereren",
                "send_gmail":               "Versturen naar team",
                "router":                   "Intentie detecteren",
            }
            label = icons.get(step["label"], step["label"])
            lat   = f" · {step['latency']:.1f}s" if step.get("latency") else ""
            html += f"""<div style="display:flex;align-items:center;gap:8px;
                padding:7px 10px;border-radius:6px;margin-bottom:4px;
                background:#f0faf5;border:0.5px solid #c8e6c9">
                <span style="color:#1D9E75;font-size:14px">✓</span>
                <span style="font-size:12px;color:#333;flex:1">{label}</span>
                <span style="font-size:10px;color:#aaa;font-family:monospace">{lat}</span>
            </div>"""

    if pro and (total_tokens or total_cost):
        html += f"""<div style="margin-top:8px;padding:6px 8px;background:#f8f8f8;
            border-radius:5px;border:0.5px solid #eee;font-size:9px;
            font-family:monospace;color:#555;line-height:1.7">
            {"─"*40}<br>
            {len(steps)} tool calls &nbsp;·&nbsp;
            {total_tokens:,} tokens &nbsp;·&nbsp;
            ${total_cost:.6f}<br>
            {"─"*40}
            {f'<br><a href="https://smith.langchain.com" target="_blank" style="color:#C0392B">LangSmith: smith.langchain.com/runs/{run_id}</a>' if run_id else ""}
        </div>"""

    return html

# ── Main UI ────────────────────────────────────────────────────────────────────
L = LABELS[st.session_state.language]

sidebar_col, main_col = st.columns([1, 3], gap="small")

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with sidebar_col:
    # Language
    lang = st.selectbox("🌐", ["Nederlands","English","Français"],
        index=["Nederlands","English","Français"].index(st.session_state.language),
        label_visibility="collapsed")
    if lang != st.session_state.language:
        st.session_state.language = lang
        st.rerun()
    L = LABELS[st.session_state.language]

    # Logo
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:10px 0 14px">
        <div style="width:34px;height:34px;background:#C0392B;border-radius:9px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:13px;font-weight:500;color:white;flex-shrink:0">IQ</div>
        <div>
            <div style="font-size:15px;font-weight:500;color:#1a1a1a">IncidentIQ</div>
            <div style="font-size:10px;color:#aaa">AI Incident Intelligence</div>
        </div>
    </div>
    <hr style="border:none;border-top:1px solid #f0f0f0;margin:0 0 12px">
    """, unsafe_allow_html=True)

    # Video status
    if st.session_state.video_loaded:
        title = st.session_state.video_title
        st.markdown(f"""
        <div style="background:#f0faf5;border-radius:8px;padding:10px 12px;
                    border:0.5px solid #a8d5b5;margin-bottom:10px">
            <div style="font-size:10px;color:#1D9E75;font-weight:500;
                        display:flex;align-items:center;gap:5px;margin-bottom:3px">
                <div style="width:6px;height:6px;border-radius:50%;background:#1D9E75"></div>
                {L['video_ok']}
            </div>
            <div style="font-size:11px;color:#444">{title[:40]}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#fafafa;border-radius:8px;padding:10px 12px;
                    border:0.5px solid #eee;margin-bottom:10px">
            <div style="font-size:11px;color:#bbb">{L['no_video']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='font-size:10px;letter-spacing:0.08em;color:#bbb;font-weight:500;margin-bottom:6px'>GENEREREN</div>", unsafe_allow_html=True)

    # PDF button
    if st.button(f"📄  {L['btn_pdf']}", use_container_width=True):
        if not st.session_state.video_loaded:
            st.warning("Load a video first.")
        else:
            with st.spinner(L["generating"]):
                t0 = time.time()
                add_trace("pro", "router", "intent: generate_pdf_cheatsheet")
                results = vs.similarity_search("key points lessons recommendations conclusions", k=10)
                if results:
                    context = "\n\n".join([r.page_content for r in results])
                    fp, title = make_pdf(context, lang_tool(), st.session_state.video_url)
                    st.session_state.last_pdf_path  = fp
                    st.session_state.pdf_title      = title
                    with open(fp,"rb") as f:
                        st.session_state.last_pdf_bytes = f.read()
                    lat = time.time() - t0
                    tok = len(context.split()) * 2
                    add_trace("done", "generate_pdf_cheatsheet",
                              f"language: {lang_tool()} · keypoints: 5 · output: {Path(fp).name}",
                              latency=lat, tokens=tok, cost=tok*0.00000015, badge="ok",
                              extra={"file": Path(fp).name, "lang": lang_tool()})
                    st.session_state.active_tab = "pdf"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"PDF aangemaakt: **{title}**",
                        "type": "text",
                    })
                    st.rerun()

    # Visual Timeline button
    if st.button(f"📊  {L['btn_visual']}", use_container_width=True):
        if not st.session_state.video_loaded:
            st.warning("Load a video first.")
        else:
            with st.spinner(L["generating"]):
                add_trace("pro", "router", "intent: generate_visual_summary")
                result, calls, lat = ask(f"Generate a visual timeline summary in {lang_tool()}")
                try:
                    json.loads(result)
                    st.session_state.visual_content = result
                    st.session_state.active_tab = "timeline"
                    st.session_state.messages.append({"role":"assistant","content":result,"type":"visual"})
                except Exception:
                    st.session_state.messages.append({"role":"assistant","content":result,"type":"text"})
                st.rerun()

    # XVR button
    if st.button(f"🎮  {L['btn_xvr']}", use_container_width=True):
        if not st.session_state.video_loaded:
            st.warning("Load a video first.")
        else:
            with st.spinner(L["generating"]):
                add_trace("pro", "router", "intent: generate_xvr_scenario")
                result, calls, lat = ask(f"Generate an XVR scenario brief in {lang_tool()}")
                st.session_state.xvr_content = result
                st.session_state.active_tab  = "xvr"
                st.session_state.messages.append({"role":"assistant","content":result,"type":"text"})
                st.rerun()

    st.markdown("<hr style='border:none;border-top:1px solid #f0f0f0;margin:12px 0 10px'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:10px;letter-spacing:0.08em;color:#bbb;font-weight:500;margin-bottom:6px'>{L['send_to'].upper()}</div>", unsafe_allow_html=True)

    email_to = st.text_input("Email", placeholder=L["send_ph"], label_visibility="collapsed", key="email_input")

    doc_opts = [L["btn_pdf"], L["btn_visual"], L["btn_xvr"]]
    doc_choice = st.selectbox("Doc", doc_opts, label_visibility="collapsed", key="doc_sel")

    if os.getenv("GMAIL_DISTRIBUTION_LIST","").strip():
        add_dist = st.checkbox("+ distributielijst", key="add_dist")
    else:
        add_dist = False

    if st.button(f"📤  {L['send_btn']}", use_container_width=True, type="primary"):
        if not email_to:
            st.warning("Vul een e-mailadres in.")
        else:
            dist = os.getenv("GMAIL_DISTRIBUTION_LIST","") if add_dist else ""
            all_emails = email_to + ("," + dist if dist else "")
            with st.spinner("Versturen..."):
                add_trace("pro", "send_gmail", f"to: {email_to}")
                if L["btn_pdf"] in doc_choice and st.session_state.last_pdf_path:
                    result, _, lat = ask(
                        f"Send the PDF at {st.session_state.last_pdf_path} "
                        f"to {all_emails} with subject_suffix 'Key Concepts Cheatsheet'"
                    )
                elif L["btn_xvr"] in doc_choice and st.session_state.xvr_content:
                    result, _, lat = ask(
                        f"Send this XVR scenario to {all_emails} with subject_suffix 'XVR Scenario':\n"
                        f"{st.session_state.xvr_content[:500]}"
                    )
                else:
                    result, _, lat = ask(
                        f"Send a visual summary to {all_emails} with subject_suffix 'Visual Summary'"
                    )
                add_trace("done", "send_gmail", f"to: {email_to}", latency=lat, badge="ok")
                st.success(result[:100] if result else "Verstuurd!")

    st.markdown("<hr style='border:none;border-top:1px solid #f0f0f0;margin:12px 0 10px'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:10px;letter-spacing:0.08em;color:#bbb;font-weight:500;margin-bottom:8px'>{L['activity'].upper()}</div>", unsafe_allow_html=True)

    # Pro/User toggle
    c1, c2, c3 = st.columns([3,1,3])
    with c1:
        st.markdown(f"<div style='font-size:11px;color:#aaa;text-align:right;padding-top:7px'>{L['mode_user']}</div>", unsafe_allow_html=True)
    with c2:
        tog = st.toggle("", value=st.session_state.pro_mode, key="pro_tog", label_visibility="collapsed")
        if tog != st.session_state.pro_mode:
            st.session_state.pro_mode = tog
            st.rerun()
    with c3:
        st.markdown(f"<div style='font-size:11px;color:#aaa;padding-top:7px'>{L['mode_pro']}</div>", unsafe_allow_html=True)

    # Trace display
    trace_html = render_trace_html()
    st.markdown(f"<div style='margin-top:6px'>{trace_html}</div>", unsafe_allow_html=True)

    if st.session_state.trace_steps:
        if st.button(f"↺ {L['clear']}", use_container_width=True):
            st.session_state.trace_steps  = []
            st.session_state.total_tokens = 0
            st.session_state.total_cost   = 0.0
            st.rerun()

    if os.getenv("LANGSMITH_API_KEY"):
        st.markdown("""
        <a href="https://smith.langchain.com" target="_blank"
           style="display:flex;align-items:center;justify-content:center;gap:5px;
                  padding:7px;border-radius:7px;border:0.5px solid #eee;
                  font-size:11px;color:#C0392B;text-decoration:none;margin-top:8px">
            📊 LangSmith traces →
        </a>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════
with main_col:
    # Header
    vid_info = st.session_state.video_title if st.session_state.video_loaded else "IncidentIQ"
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                padding:8px 4px 14px;border-bottom:1px solid #f0f0f0;margin-bottom:0">
        <div style="font-size:13px;color:#999">{vid_info}</div>
        <div style="display:flex;gap:6px">
            <span style="font-size:10px;color:#C0392B;background:#FEF0EE;padding:3px 9px;border-radius:20px;border:0.5px solid #f5c6be">gpt-4o-mini</span>
            <span style="font-size:10px;color:#2980B9;background:#E8F4FD;padding:3px 9px;border-radius:20px;border:0.5px solid #b8d4ec">Pinecone</span>
            <span style="font-size:10px;color:#1D9E75;background:#E1F5EE;padding:3px 9px;border-radius:20px;border:0.5px solid #c8e6c9">LangSmith</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tab bar
    tabs = [L["tab_chat"], L["tab_pdf"], L["tab_tl"], L["tab_xvr"]]
    tab_keys = ["chat","pdf","timeline","xvr"]
    tab_cols = st.columns(len(tabs))
    for i, (tab, key) in enumerate(zip(tabs, tab_keys)):
        with tab_cols[i]:
            active = st.session_state.active_tab == key
            style = "color:#C0392B;border-bottom:2px solid #C0392B;font-weight:500;" if active else "color:#aaa;border-bottom:2px solid transparent;"
            if st.button(tab, key=f"tab_{key}", use_container_width=True):
                st.session_state.active_tab = key
                st.rerun()

    st.markdown("<div style='border-bottom:1px solid #f0f0f0;margin-bottom:14px'></div>", unsafe_allow_html=True)

    # ── TAB: CHAT ─────────────────────────────────────────────────────────────
    if st.session_state.active_tab == "chat":
        if not st.session_state.messages:
            st.markdown(f"""
            <div style="text-align:center;padding:80px 20px">
                <div style="font-size:36px;opacity:0.12;color:#C0392B;margin-bottom:16px">◈</div>
                <div style="font-size:15px;color:#aaa;margin-bottom:6px;font-weight:500">{L['welcome']}</div>
                <div style="font-size:12px;color:#ccc">{L['welcome_sub']}</div>
            </div>""", unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div style="background:#C0392B;color:white;border-radius:16px 16px 3px 16px;
                                padding:10px 15px;margin:8px 0 8px 20%;font-size:14px;line-height:1.65">
                        {msg['content']}
                    </div>""", unsafe_allow_html=True)
                else:
                    mtype = msg.get("type","text")
                    if mtype == "visual":
                        try:
                            data = json.loads(msg["content"])
                            render_visual_in_chat(data)
                        except Exception:
                            st.markdown(f"""
                            <div style="background:#f7f7f7;border:0.5px solid #eee;border-radius:3px 16px 16px 16px;
                                        padding:10px 15px;margin:8px 20% 8px 0;font-size:14px;line-height:1.65">
                                {msg['content']}
                            </div>""", unsafe_allow_html=True)
                    else:
                        content = msg["content"].replace("\n","<br>")
                        st.markdown(f"""
                        <div style="background:#f7f7f7;border:0.5px solid #eee;border-radius:3px 16px 16px 16px;
                                    padding:10px 15px;margin:8px 20% 8px 0;font-size:14px;line-height:1.65">
                            {content}
                        </div>""", unsafe_allow_html=True)

        # Input
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        ci, cb = st.columns([11,1])
        with ci:
            user_input = st.text_input("", placeholder=L["placeholder"],
                                       label_visibility="collapsed", key="chat_in")
        with cb:
            send = st.button("→", use_container_width=True, key="chat_send")

        if (send or user_input) and user_input.strip():
            msg = user_input.strip()
            if msg == st.session_state.last_message:
                st.stop()
            st.session_state.last_message = msg
            st.session_state.messages.append({"role":"user","content":msg,"type":"text"})

            is_url = "youtube.com" in msg or "youtu.be" in msg

            if is_url:
                with st.spinner(""):
                    response = load_video(msg)
                st.session_state.messages.append({"role":"assistant","content":response,"type":"text"})
            else:
                with st.spinner(""):
                    add_trace("pro","router","intent detected")
                    response, calls, lat = ask(msg)
                    is_visual = False
                    try:
                        parsed = json.loads(response)
                        is_visual = "timeline" in parsed and "metrics" in parsed
                    except Exception:
                        pass
                    if "File path:" in response:
                        try:
                            fp = response.split("File path: ")[1].split("\n")[0].strip()
                            if Path(fp).exists():
                                st.session_state.last_pdf_path = fp
                                with open(fp,"rb") as f:
                                    st.session_state.last_pdf_bytes = f.read()
                        except Exception:
                            pass
                    mtype = "visual" if is_visual else "text"
                    if is_visual: st.session_state.visual_content = response
                    st.session_state.messages.append({"role":"assistant","content":response,"type":mtype})
            st.rerun()

    # ── TAB: PDF ──────────────────────────────────────────────────────────────
    elif st.session_state.active_tab == "pdf":
        if st.session_state.last_pdf_bytes:
            st.markdown(f"""
            <div style="background:#f0faf5;border-radius:10px;padding:16px 20px;
                        border:0.5px solid #a8d5b5;margin-bottom:16px;
                        display:flex;align-items:center;gap:12px">
                <div style="font-size:24px">📄</div>
                <div>
                    <div style="font-size:14px;font-weight:500;color:#1a1a1a">
                        {st.session_state.pdf_title or 'Key Concepts Cheatsheet'}
                    </div>
                    <div style="font-size:11px;color:#666">
                        Generated by IncidentIQ AI · {datetime.now().strftime('%d/%m/%Y')}
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
            st.download_button(
                "⬇️  Download PDF",
                st.session_state.last_pdf_bytes,
                file_name="incidentiq_cheatsheet.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.markdown(f"""
            <div style="text-align:center;padding:60px 20px;color:#bbb">
                <div style="font-size:32px;margin-bottom:12px;opacity:0.3">📄</div>
                <div style="font-size:14px">Klik op '{L['btn_pdf']}' in de sidebar om een PDF te genereren.</div>
            </div>""", unsafe_allow_html=True)

    # ── TAB: TIMELINE ─────────────────────────────────────────────────────────
    elif st.session_state.active_tab == "timeline":
        if st.session_state.visual_content:
            try:
                data = json.loads(st.session_state.visual_content)
                render_visual_timeline_tab(data)
            except Exception:
                st.info("Visual timeline data not available. Click 'Visuele tijdlijn' to generate.")
        else:
            st.markdown(f"""
            <div style="text-align:center;padding:60px 20px;color:#bbb">
                <div style="font-size:32px;margin-bottom:12px;opacity:0.3">📊</div>
                <div style="font-size:14px">Klik op '{L['btn_visual']}' in de sidebar om de tijdlijn te genereren.</div>
            </div>""", unsafe_allow_html=True)

    # ── TAB: XVR ──────────────────────────────────────────────────────────────
    elif st.session_state.active_tab == "xvr":
        if st.session_state.xvr_content:
            st.markdown(f"""
            <div style="background:#F3E8FD;border-radius:10px;padding:12px 16px;
                        border:0.5px solid #d9b3f0;margin-bottom:16px;
                        display:flex;align-items:center;gap:10px">
                <div style="font-size:20px">🎮</div>
                <div style="font-size:13px;font-weight:500;color:#5b2c8d">XVR Simulation Scenario Brief</div>
            </div>""", unsafe_allow_html=True)
            content = st.session_state.xvr_content
            for line in content.split("\n"):
                if line.startswith("SCENARIO BRIEF") or line.startswith("==="):
                    st.markdown(f"<div style='font-size:14px;font-weight:500;color:#1a1a1a;margin:8px 0'>{line}</div>", unsafe_allow_html=True)
                elif line.startswith(("INCIDENT TITLE","LOCATION","INITIAL SITUATION","ENVIRONMENTAL","SCENARIO COMP","CRITICAL DECISION","LEARNING OBJ","DEBRIEFING","XVR OPERATOR")):
                    st.markdown(f"<div style='font-size:12px;font-weight:500;color:#5b2c8d;margin:12px 0 4px;letter-spacing:0.03em'>{line}</div>", unsafe_allow_html=True)
                elif line.startswith("-") or line.startswith("•"):
                    st.markdown(f"<div style='font-size:13px;color:#333;padding:2px 0 2px 12px;line-height:1.6'>{line}</div>", unsafe_allow_html=True)
                elif line.strip():
                    st.markdown(f"<div style='font-size:13px;color:#444;line-height:1.6'>{line}</div>", unsafe_allow_html=True)
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            if st.session_state.xvr_content:
                st.download_button(
                    "⬇️  Download XVR Scenario (.txt)",
                    st.session_state.xvr_content.encode(),
                    file_name="xvr_scenario.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
        else:
            st.markdown(f"""
            <div style="text-align:center;padding:60px 20px;color:#bbb">
                <div style="font-size:32px;margin-bottom:12px;opacity:0.3">🎮</div>
                <div style="font-size:14px">Klik op '{L['btn_xvr']}' in de sidebar om een XVR scenario te genereren.</div>
            </div>""", unsafe_allow_html=True)

# ── Visual timeline renderer ───────────────────────────────────────────────────
def render_visual_timeline_tab(data):
    COLOR_MAP = {
        "red":   ("#C0392B","#FAECE7","#7A2419"),
        "amber": ("#E67E22","#FEF3E2","#7A4A10"),
        "green": ("#1D9E75","#E1F5EE","#0A5C3F"),
        "blue":  ("#2980B9","#E8F4FD","#1A5276"),
    }
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1C2833,#2C3E50);padding:20px 24px;
                border-radius:12px;margin-bottom:20px">
        <div style="font-size:10px;letter-spacing:0.1em;color:#C0392B;background:#C0392B22;
                    padding:3px 10px;border-radius:4px;border:1px solid #C0392B44;
                    display:inline-block;margin-bottom:10px;font-weight:500">INCIDENT ANALYSIS</div>
        <div style="font-size:20px;font-weight:500;color:white;margin-bottom:4px">{data.get('title','')}</div>
        <div style="font-size:12px;color:rgba(255,255,255,0.5)">{data.get('subtitle','')} · {data.get('duration','')}</div>
    </div>""", unsafe_allow_html=True)

    metrics = data.get("metrics",[])
    if metrics:
        cols = st.columns(len(metrics))
        for i, m in enumerate(metrics):
            ch,bh,th = COLOR_MAP.get(m.get("color","blue"),COLOR_MAP["blue"])
            with cols[i]:
                st.markdown(f"""
                <div style="background:#f8f8f8;border-radius:10px;padding:14px;
                            border-bottom:3px solid {ch};margin-bottom:16px">
                    <div style="font-size:24px;font-weight:500;color:#1a1a1a;
                                font-family:monospace;line-height:1">
                        {m.get('value','')}
                        <span style="font-size:12px;color:#aaa;font-weight:400"> {m.get('unit','')}</span>
                    </div>
                    <div style="font-size:11px;color:#777;margin-top:5px">{m.get('label','')}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("<div style='font-size:10px;letter-spacing:0.09em;color:#bbb;font-weight:500;margin-bottom:12px'>INCIDENT TIJDLIJN</div>", unsafe_allow_html=True)

    for ev in data.get("timeline",[]):
        ch,bh,th = COLOR_MAP.get(ev.get("color","blue"),COLOR_MAP["blue"])
        quote_html = f'<div style="border-left:2px solid {ch};padding-left:10px;margin:8px 0;font-size:12px;color:#555;font-style:italic;line-height:1.6">{ev["quote"]}</div>' if ev.get("quote") else ""
        tags_html  = "".join([f'<span style="font-size:10px;padding:2px 8px;border-radius:4px;background:#f5f5f5;color:#888;border:0.5px solid #eee;margin-right:4px">{t}</span>' for t in ev.get("tags",[])])
        st.markdown(f"""
        <div style="display:flex;gap:0;margin-bottom:2px">
            <div style="width:52px;flex-shrink:0;padding-top:14px;text-align:right;padding-right:10px">
                <span style="font-size:10px;color:#aaa;font-family:monospace">{ev.get('timestamp','')}</span>
            </div>
            <div style="width:20px;flex-shrink:0;display:flex;flex-direction:column;align-items:center;padding-top:14px">
                <div style="width:10px;height:10px;border-radius:50%;background:{ch};box-shadow:0 0 0 3px {ch}22;flex-shrink:0"></div>
                <div style="width:1px;flex:1;background:#f0f0f0;min-height:20px"></div>
            </div>
            <div style="flex:1;padding:8px 0 16px 12px">
                <div style="background:white;border-radius:10px;padding:14px 16px;border:0.5px solid #f0f0f0;border-left:3px solid {ch}">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;gap:8px">
                        <div style="font-size:13px;font-weight:500;color:#1a1a1a">{ev.get('title','')}</div>
                        <span style="font-size:10px;padding:2px 8px;border-radius:4px;background:{bh};color:{th};font-weight:500;white-space:nowrap;flex-shrink:0">{ev.get('badge','')}</span>
                    </div>
                    <div style="font-size:12px;color:#555;line-height:1.7">{ev.get('text','')}</div>
                    {quote_html}
                    <div style="margin-top:8px">{tags_html}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='font-size:10px;letter-spacing:0.09em;color:#bbb;font-weight:500;margin:16px 0 10px'>KEY LEARNINGS</div>", unsafe_allow_html=True)
    learnings = data.get("learnings",[])
    if learnings:
        cols = st.columns(2)
        for i, l in enumerate(learnings):
            with cols[i%2]:
                st.markdown(f"""
                <div style="background:#f8f8f8;border-radius:10px;padding:14px;margin-bottom:8px">
                    <div style="font-size:10px;color:#C0392B;font-family:monospace;font-weight:500;margin-bottom:6px">{l.get('number','')}</div>
                    <div style="font-size:12px;font-weight:500;color:#1a1a1a;margin-bottom:5px">{l.get('title','')}</div>
                    <div style="font-size:11px;color:#666;line-height:1.6">{l.get('text','')}</div>
                </div>""", unsafe_allow_html=True)

def render_visual_in_chat(data):
    render_visual_timeline_tab(data)
