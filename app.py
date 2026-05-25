"""
IncidentIQ - AI-powered Incident Intelligence
Gradio 6.14.0 + Python 3.11 — final version
"""

import os, re, json, time, base64, uuid
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import gradio as gr
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

load_dotenv()
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT']    = 'incidentiq-agent'
if os.getenv('LANGSMITH_API_KEY'):
    os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGSMITH_API_KEY')

# ── Init ───────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
emb = OpenAIEmbeddings(model="text-embedding-3-small")
pc  = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
vs  = PineconeVectorStore(
    index_name="incidentiq",
    embedding=emb,
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
)
LANG_TOOL = {"Nederlands":"dutch", "English":"english", "Français":"french"}

# ── Button labels per language ─────────────────────────────────────────────────
BTN = {
    "Nederlands": {
        "pdf":  "📄  Genereer Key Concepts PDF",
        "tl":   "📊  Genereer Visuele Tijdlijn",
        "xvr":  "🎮  Genereer XVR Scenario",
        "send": "📤  Versturen",
        "doc_opts": ["📄 Key Concepts PDF","📊 Visuele tijdlijn","🎮 XVR Scenario"],
    },
    "English": {
        "pdf":  "📄  Generate Key Concepts PDF",
        "tl":   "📊  Generate Visual Timeline",
        "xvr":  "🎮  Generate XVR Scenario",
        "send": "📤  Send",
        "doc_opts": ["📄 Key Concepts PDF","📊 Visual Timeline","🎮 XVR Scenario"],
    },
    "Français": {
        "pdf":  "📄  Générer Concepts Clés PDF",
        "tl":   "📊  Générer Chronologie Visuelle",
        "xvr":  "🎮  Générer Scénario XVR",
        "send": "📤  Envoyer",
        "doc_opts": ["📄 Concepts Clés PDF","📊 Chronologie Visuelle","🎮 Scénario XVR"],
    },
}

# ── State ──────────────────────────────────────────────────────────────────────
S = {
    "video_loaded": False,
    "video_title":  "",
    "video_url":    "",
    "thread_id":    f"s_{uuid.uuid4().hex[:8]}",
    "pdf_path":     None,
    "pdf_data":     None,
    "xvr_content":  "",
    "visual_json":  "",
    "trace":        [],
    "total_tokens": 0,
    "total_cost":   0.0,
    "run_id":       "",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def vid_id(url):
    if "v=" in url: return url.split("v=")[1].split("&")[0]
    if "youtu.be/" in url: return url.split("youtu.be/")[1].split("?")[0]
    raise ValueError("Cannot extract video ID")

def clean_tx(t):
    t = re.sub(r'\[Music\]|\[Applause\]|\[Laughter\]|\[Cheering\]', '', t)
    t = re.sub(r'\[\d{2}:\d{2}\]\s*(?=\[\d{2}:\d{2}\])', '', t)
    return re.sub(r'\s+', ' ', t).strip()

def add_trace(label, detail="", lat=None, tokens=0, cost=0.0):
    S["trace"].append({"label":label,"detail":detail,"lat":lat,"tokens":tokens,"cost":cost,"run_id":S["run_id"]})
    S["total_tokens"] += tokens
    S["total_cost"]   += cost

def video_status_html():
    if S["video_loaded"]:
        return (
            f"<div style='background:#0d1f14;border-radius:8px;padding:10px 12px;"
            f"border:1px solid #1D9E7533;margin:4px 0'>"
            f"<div style='font-size:10px;color:#1D9E75;font-weight:500;margin-bottom:2px'>● Video geladen</div>"
            f"<div style='font-size:11px;color:#aaa'>{S['video_title'][:40]}</div></div>"
        )
    return "<div style='background:#1a1a1a;border-radius:8px;padding:10px 12px;border:1px solid #2a2a2a;margin:4px 0;font-size:11px;color:#555'>Geen video geladen</div>"

def render_trace(pro):
    steps = S["trace"][-10:]
    if not steps:
        return "<div style='color:#555;font-size:11px;text-align:center;padding:12px'>Nog geen activiteit...</div>"
    html = ""
    if pro and S["run_id"]:
        html += (
            f"<div style='font-size:9px;color:#555;font-family:monospace;padding:6px 8px;"
            f"background:#161616;border-radius:6px;margin-bottom:8px;border:1px solid #2a2a2a;line-height:1.7'>"
            f"RUN {S['run_id']} · {S['thread_id'][:16]}<br>"
            f"model: gpt-4o-mini · embed: text-embedding-3-small · index: incidentiq</div>"
        )
    for i, step in enumerate(steps):
        is_last = i == len(steps)-1
        if pro:
            lat_s = f" · {step['lat']:.2f}s" if step.get("lat") else ""
            tok_s = f" · {step['tokens']} tok" if step.get("tokens") else ""
            conn  = "" if is_last else "<div style='width:1px;flex:1;background:#2a2a2a;margin-top:3px;min-height:10px'></div>"
            html += (
                f"<div style='display:flex;gap:6px;margin-bottom:5px'>"
                f"<div style='display:flex;flex-direction:column;align-items:center;width:10px;flex-shrink:0;padding-top:3px'>"
                f"<div style='width:8px;height:8px;border-radius:50%;background:#1D9E75;flex-shrink:0'></div>{conn}</div>"
                f"<div style='flex:1;background:#161616;border-radius:6px;padding:6px 8px;border:1px solid #2a2a2a;border-left:2px solid #1D9E75'>"
                f"<div style='font-size:10px;font-family:monospace;color:#aaa'>"
                f"<span style='background:#0d1f14;color:#1D9E75;padding:1px 5px;border-radius:3px;font-size:9px'>ok</span> "
                f"<span style='color:#C0392B'>tool /</span> {step['label']}"
                f"<span style='color:#555'>{lat_s}{tok_s}</span></div>"
                f"<div style='font-size:9px;color:#555;font-family:monospace;margin-top:2px'>{step.get('detail','')}</div>"
                f"</div></div>"
            )
        else:
            icons = {
                "fetch_youtube_transcript": "Video laden",
                "search_video_knowledge":   "Zoeken in de video",
                "generate_xvr_scenario":    "XVR scenario maken",
                "generate_visual_summary":  "Tijdlijn genereren",
                "generate_pdf_cheatsheet":  "PDF aanmaken",
                "send_gmail_tool":          "Versturen naar team",
            }
            label = icons.get(step["label"], step["label"])
            lat   = f" · {step['lat']:.1f}s" if step.get("lat") else ""
            html += (
                f"<div style='display:flex;align-items:center;gap:8px;padding:7px 10px;"
                f"border-radius:6px;margin-bottom:4px;background:#0d1f14;border:1px solid #1D9E7533'>"
                f"<span style='color:#1D9E75;font-size:13px'>✓</span>"
                f"<span style='font-size:12px;color:#ddd;flex:1'>{label}</span>"
                f"<span style='font-size:10px;color:#555;font-family:monospace'>{lat}</span></div>"
            )
    if pro and S["total_tokens"]:
        ls = f'<br><a href="https://smith.langchain.com" target="_blank" style="color:#C0392B;text-decoration:none">LangSmith: runs/{S["run_id"]}</a>' if S["run_id"] else ""
        html += (
            f"<div style='margin-top:8px;padding:6px 8px;background:#161616;border-radius:5px;"
            f"border:1px solid #2a2a2a;font-size:9px;font-family:monospace;color:#555;line-height:1.8'>"
            f"{'─'*34}<br>{len(steps)} tool calls · {S['total_tokens']:,} tokens · ${S['total_cost']:.6f}{ls}</div>"
        )
    return html

# ── YouTube ────────────────────────────────────────────────────────────────────
def load_video(url, lang="Nederlands"):
    msgs = {
        "Nederlands": {"cached":"✓ Video was al geladen — Pinecone cache. Klaar!","loaded":"✓ Video geladen en klaar voor vragen!"},
        "English":    {"cached":"✓ Already loaded — Pinecone cache. Ready!","loaded":"✓ Video loaded and ready!"},
        "Français":   {"cached":"✓ Déjà chargée — cache Pinecone.","loaded":"✓ Vidéo chargée et prête!"},
    }
    m = msgs.get(lang, msgs["Nederlands"])
    try: video_id = vid_id(url)
    except Exception as e: return f"Cannot extract video ID: {e}"
    t0    = time.time()
    index = pc.Index("incidentiq")
    stats = index.describe_index_stats()
    if stats.total_vector_count > 0:
        test = vs.similarity_search("incident", k=1)
        if test:
            S["video_loaded"]=True; S["video_title"]=f"Video {video_id}"; S["video_url"]=url
            add_trace("fetch_youtube_transcript", f"video_id: {video_id} · pinecone_hit", lat=time.time()-t0)
            return m["cached"] + f"\n\nVideo ID: {video_id}"
    try:
        entries = YouTubeTranscriptApi().fetch(video_id, languages=["en","nl","fr"])
        txlist  = entries.snippets
    except NoTranscriptFound:   return f"No transcript found for {video_id}."
    except TranscriptsDisabled: return f"Transcripts disabled for {video_id}."
    except Exception as e:      return f"YouTube blocked. Wait 30-60 min.\nError: {e}"
    plain  = clean_tx(" ".join(t.text for t in txlist))
    ts     = clean_tx(" ".join(f"[{int(t.start//60):02d}:{int(t.start%60):02d}] {t.text}" for t in txlist))
    spl    = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = spl.create_documents(texts=[ts], metadatas=[{"video_id":video_id,"source":url}])
    vs.add_documents(chunks)
    S["video_loaded"]=True; S["video_title"]=f"Video {video_id}"; S["video_url"]=url
    add_trace("fetch_youtube_transcript", f"video_id: {video_id} · chunks: {len(chunks)}", lat=time.time()-t0)
    return m["loaded"] + f"\n\nVideo ID: {video_id} · {len(chunks)} chunks."

# ── Key Concepts visual renderer ───────────────────────────────────────────────
def render_key_concepts(data):
    title    = data.get("title","")
    subtitle = data.get("subtitle","")
    kp       = data.get("keypoints",[])
    rec      = data.get("recommendations",[])
    tags     = data.get("tags",[])

    tags_html = "".join([
        f"<span style='font-size:10px;padding:2px 10px;border-radius:20px;background:#C0392B22;"
        f"color:#C0392B;border:1px solid #C0392B44;margin-right:5px'>{t}</span>"
        for t in tags
    ])

    kp_html = "".join([
        f"<div style='display:flex;align-items:flex-start;gap:10px;padding:10px 0;border-bottom:1px solid #2a2a2a'>"
        f"<div style='width:6px;height:6px;border-radius:50%;background:#C0392B;flex-shrink:0;margin-top:6px'></div>"
        f"<div style='font-size:13px;color:#ddd;line-height:1.6'>{k}</div>"
        f"</div>"
        for k in kp
    ])

    rec_html = "".join([
        f"<div style='display:flex;align-items:flex-start;gap:10px;padding:10px 0;border-bottom:1px solid #2a2a2a'>"
        f"<div style='width:6px;height:6px;border-radius:50%;background:#1D9E75;flex-shrink:0;margin-top:6px'></div>"
        f"<div style='font-size:13px;color:#ddd;line-height:1.6'>{r}</div>"
        f"</div>"
        for r in rec
    ])

    return (
        f"<div style='background:#161616;border-radius:12px;padding:20px 24px;border:1px solid #2a2a2a'>"
        f"<div style='background:linear-gradient(135deg,#1C2833,#2C3E50);padding:18px 20px;border-radius:10px;margin-bottom:20px'>"
        f"<div style='font-size:9px;letter-spacing:0.12em;color:#C0392B;background:#C0392B22;padding:3px 10px;"
        f"border-radius:4px;border:1px solid #C0392B44;display:inline-block;margin-bottom:10px;font-weight:500'>KEY CONCEPTS</div>"
        f"<div style='font-size:18px;font-weight:500;color:white;margin-bottom:4px'>{title}</div>"
        f"<div style='font-size:12px;color:rgba(255,255,255,0.5);margin-bottom:10px'>{subtitle}</div>"
        f"<div>{tags_html}</div>"
        f"</div>"
        f"<div style='margin-bottom:20px'>"
        f"<div style='font-size:10px;letter-spacing:0.09em;color:#555;font-weight:500;margin-bottom:8px'>KEY POINTS</div>"
        f"{kp_html}</div>"
        f"<div style='background:#0d1f14;border-radius:8px;padding:10px 14px;margin-bottom:12px;border:1px solid #1D9E7533'>"
        f"<div style='font-size:10px;color:#1D9E75;font-weight:500;margin-bottom:2px'>AI analysis</div>"
        f"<div style='font-size:10px;color:#555;font-style:italic'>Automatically extracted from video — not cited from a person.</div>"
        f"</div>"
        f"<div style='font-size:10px;letter-spacing:0.09em;color:#555;font-weight:500;margin-bottom:8px'>AI-GENERATED RECOMMENDATIONS</div>"
        f"{rec_html}"
        f"<div style='margin-top:14px;padding-top:12px;border-top:1px solid #2a2a2a;font-size:10px;color:#555;font-family:monospace'>"
        f"Generated by IncidentIQ AI · {datetime.now().strftime('%d/%m/%Y')}</div>"
        f"</div>"
    )

# ── PDF ────────────────────────────────────────────────────────────────────────
def make_pdf(data, source_url=""):
    RED=HexColor("#C0392B"); DARK=HexColor("#1C2833"); ORANGE=HexColor("#E67E22")
    GREEN=HexColor("#1E8449"); WHITE=white
    fp = f'/tmp/iq_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    c  = rl_canvas.Canvas(fp, pagesize=A4); W,H = A4
    c.setFillColor(RED);   c.rect(0,H-3.2*cm,W,3.2*cm,fill=1,stroke=0)
    c.setFillColor(WHITE); c.circle(1.8*cm,H-1.6*cm,0.85*cm,fill=1,stroke=0)
    c.setFillColor(RED);   c.setFont("Helvetica-Bold",14); c.drawCentredString(1.8*cm,H-1.95*cm,"IQ")
    c.setFillColor(WHITE); c.setFont("Helvetica-Bold",14); c.drawString(3.2*cm,H-1.3*cm,data.get("title","IncidentIQ")[:50])
    c.setFont("Helvetica",10); c.drawString(3.2*cm,H-1.85*cm,data.get("subtitle","")[:60])
    c.setFont("Helvetica",8)
    c.drawRightString(W-1.2*cm,H-1.3*cm,datetime.now().strftime("%d/%m/%Y"))
    c.drawRightString(W-1.2*cm,H-1.75*cm,"Generated by IncidentIQ AI")
    c.setFillColor(ORANGE); c.rect(0,H-3.6*cm,W,0.4*cm,fill=1,stroke=0)
    y = H-5.0*cm
    def sh(y,t,col=DARK):
        c.setFillColor(col); c.setFont("Helvetica-Bold",11); c.drawString(1.2*cm,y,t.upper())
        c.setStrokeColor(col); c.setLineWidth(1.5); c.line(1.2*cm,y-0.2*cm,W-1.2*cm,y-0.2*cm)
        return y-0.8*cm
    def bi(y,txt,col=DARK,bc=RED):
        c.setFillColor(bc); c.circle(1.5*cm,y+0.25*cm,0.1*cm,fill=1,stroke=0)
        c.setFillColor(col); c.setFont("Helvetica",10); mw=W-1.8*cm-1.2*cm
        words=txt.split(); line,lines="",[]
        for w in words:
            t=line+w+" "
            if c.stringWidth(t,"Helvetica",10)<mw: line=t
            else: lines.append(line.strip()); line=w+" "
        lines.append(line.strip())
        for i,l in enumerate(lines): c.drawString(1.8*cm,y-i*0.5*cm,l)
        return y-len(lines)*0.5*cm-0.35*cm
    y=sh(y,"Key Points",RED)
    for kp in data.get("keypoints",[]): y=bi(y,kp)
    y-=0.5*cm; y=sh(y,"AI-generated Recommendations",GREEN)
    c.setFillColor(HexColor("#EAFAF1")); c.setStrokeColor(GREEN); c.setLineWidth(0.8)
    c.roundRect(1.2*cm,y-0.8*cm,W-2.4*cm,0.7*cm,3,fill=1,stroke=1)
    c.setFillColor(GREEN); c.setFont("Helvetica-Bold",8); c.drawString(1.55*cm,y-0.32*cm,"AI analysis:")
    c.setFillColor(DARK);  c.setFont("Helvetica-Oblique",8)
    c.drawString(3.1*cm,y-0.32*cm,"Automatically extracted — not cited from a person.")
    y-=1.1*cm
    for rec in data.get("recommendations",[]): y=bi(y,rec,bc=GREEN)
    c.setFillColor(RED);  c.rect(0,1.2*cm,W,0.15*cm,fill=1,stroke=0)
    c.setFillColor(DARK); c.rect(0,0,W,1.2*cm,fill=1,stroke=0)
    c.setFillColor(WHITE); c.setFont("Helvetica",7.5)
    c.drawString(1.2*cm,0.65*cm,"IncidentIQ — AI-powered Incident Intelligence")
    if source_url: c.drawCentredString(W/2,0.65*cm,f"Source: {source_url[:80]}")
    c.drawRightString(W-1.2*cm,0.65*cm,"Page 1/1"); c.save()
    return fp

# ── Global tools ───────────────────────────────────────────────────────────────
@tool
def search_video_knowledge(query: str) -> str:
    """Search Pinecone for information about the loaded video. Uses query rewriting and multi-query. Translates non-English queries."""
    try:
        eq = llm.invoke(f"Translate to English, return only translation: {query}").content.strip()
        rw = llm.invoke(f"Rewrite for incident video search, max 20 words.\nQuery: {eq}\nRewritten:").content.strip()
        try:
            qs = json.loads(re.sub(r'```json|```','', llm.invoke(
                f"Generate 3 search query variations. Return JSON list.\nQuestion: {rw}\nJSON:"
            ).content.strip()).strip())
        except: qs = [rw]
        qs.append(rw)
        all_docs = {}
        for q in qs:
            for doc in vs.similarity_search(q, k=4):
                key = doc.page_content[:100]
                if key not in all_docs: all_docs[key] = doc
        if not all_docs: return "No relevant information found."
        combined = list(all_docs.values())
        all_ts = re.findall(r"\[\d{2}:\d{2}\]", " ".join([d.page_content for d in combined]))
        seen, uts = set(), []
        for t in all_ts:
            if t not in seen: seen.add(t); uts.append(t)
        clean = [re.sub(r"\[\d{2}:\d{2}\]\s*(?=\[\d{2}:\d{2}\])","",d.page_content) for d in combined]
        return "\n\n".join(clean) + f"\n\nSources: {' | '.join(uts[:5])}"
    except Exception as e: return f"Error: {e}"

@tool
def generate_xvr_scenario(language: str = "dutch") -> str:
    """Generate a complete XVR simulation scenario brief from the loaded incident video."""
    try:
        results = vs.similarity_search(
            "location building fire cause complications decisions resources weather time casualties evacuation", k=12)
        if not results: return "No video content found."
        context = "\n\n".join([re.sub(r"\[\d{2}:\d{2}\]\s*(?=\[\d{2}:\d{2}\])","",r.page_content) for r in results])
        lang = {"dutch":"Dutch - professional Belgian fire service terminology","english":"English","french":"French"}.get(language.lower(),"Dutch")
        return llm.invoke(
            f"Generate a complete XVR operator scenario brief in {lang}.\n\n"
            f"SCENARIO BRIEF - XVR SIMULATION\n================================\n\n"
            f"INCIDENT TITLE:\n[Short title]\n\n"
            f"LOCATION & BUILDING:\n- Building type: [type]\n- Floors: [number]\n- Construction: [materials]\n\n"
            f"INITIAL SITUATION T+00:00:\n- Fire location: [exact]\n- Visibility: [smoke/flames]\n- Casualties: [number]\n- Resources: [vehicles]\n\n"
            f"ENVIRONMENTAL CONDITIONS:\n- Time: [if mentioned]\n- Weather: [if mentioned]\n- Hazards: [materials]\n\n"
            f"SCENARIO COMPLICATIONS:\n- T+[time]: [complication 1]\n- T+[time]: [complication 2]\n- T+[time]: [complication 3]\n- T+[time]: [complication 4]\n\n"
            f"CRITICAL DECISION MOMENTS:\n1. [Decision]\n2. [Decision]\n3. [Decision]\n\n"
            f"LEARNING OBJECTIVES:\n- [Objective 1]\n- [Objective 2]\n- [Objective 3]\n\n"
            f"DEBRIEFING QUESTIONS:\n1. [Question based on actual mistakes]\n2. [Question]\n3. [Question]\n\n"
            f"XVR OPERATOR NOTES:\n[Key moments to inject]\n\n"
            f"STRICT RULES: Base ONLY on context. Never invent details.\n\nContext:\n{context}\n\nBrief:"
        ).content.strip()
    except Exception as e: return f"Error: {e}"

@tool
def generate_visual_summary(language: str = "dutch") -> str:
    """Generate structured JSON for visual timeline using Melding/Aankomst/Problemen/Oplossingen/Einde template."""
    try:
        results = vs.similarity_search(
            "melding aankomst aanleiding oorzaak complicaties problemen acties oplossingen resultaat slachtoffers tijdstip", k=12)
        if not results: return "No video content found."
        context = "\n\n".join([re.sub(r"\[\d{2}:\d{2}\]\s*(?=\[\d{2}:\d{2}\])","",r.page_content) for r in results])
        lang = {"dutch":"Dutch","english":"English","french":"French"}.get(language.lower(),"Dutch")
        raw = re.sub(r'```json|```','', llm.invoke(
            f'Extract ONLY facts explicitly mentioned in the context. Return raw JSON in {lang}:\n'
            f'{{"title":"[incident title from context]","subtitle":"[presenter/source from context]","duration":"[duration if mentioned, else unknown]",'
            f'"metrics":['
            f'{{"value":"[real value from context]","unit":"[unit]","label":"[metric label in {lang}]","color":"blue"}},'
            f'{{"value":"[real value from context]","unit":"[unit]","label":"[metric label in {lang}]","color":"amber"}},'
            f'{{"value":"[real value from context]","unit":"[unit]","label":"[metric label in {lang}]","color":"red"}},'
            f'{{"value":"[real value from context]","unit":"","label":"[metric label in {lang}]","color":"green"}}],'
            f'"timeline":['
            f'{{"timestamp":"[time from context or 00:00]","title":"Melding","text":"[wat was de melding, wanneer en door wie — uit context]","quote":"[directe quote als beschikbaar]","tags":["melding"],"color":"blue","badge":"Melding"}},'
            f'{{"timestamp":"[time]","title":"Aankomst","text":"[situatie bij aankomst — uit context]","quote":"","tags":["aankomst"],"color":"amber","badge":"Aankomst"}},'
            f'{{"timestamp":"[time]","title":"Problemen","text":"[complicaties en problemen tijdens interventie — uit context]","quote":"[directe quote als beschikbaar]","tags":["probleem"],"color":"red","badge":"Complicatie"}},'
            f'{{"timestamp":"[time]","title":"Oplossingen","text":"[acties ondernomen en beslissingen — uit context]","quote":"","tags":["actie"],"color":"amber","badge":"Actie"}},'
            f'{{"timestamp":"[time]","title":"Einde","text":"[resultaat van de interventie — uit context]","quote":"","tags":["resultaat"],"color":"green","badge":"Resultaat"}}],'
            f'"learnings":['
            f'{{"number":"01","title":"[les 1 titel]","text":"[max 2 zinnen — uit context]"}},'
            f'{{"number":"02","title":"[les 2 titel]","text":"[max 2 zinnen — uit context]"}},'
            f'{{"number":"03","title":"[les 3 titel]","text":"[max 2 zinnen — uit context]"}},'
            f'{{"number":"04","title":"[les 4 titel]","text":"[max 2 zinnen — uit context]"}}],'
            f'"source_url":""}}\n\n'
            f'STRIKTE REGELS:\n'
            f'- Gebruik ALLEEN feiten uit de context hieronder\n'
            f'- Verzin NOOIT tijdstempels, getallen of events\n'
            f'- Gebruik exact deze 5 tijdlijn events: Melding, Aankomst, Problemen, Oplossingen, Einde\n'
            f'- Als info niet in context staat: schrijf "Niet vermeld in de video"\n'
            f'- Alle tekst in {lang}\n\n'
            f'Context:\n{context}\n\nJSON:'
        ).content.strip()).strip()
        if "{" in raw and "}" in raw:
            raw = raw[raw.index("{"):raw.rindex("}")+1]
        json.loads(raw)
        return raw
    except Exception as e: return f"Error: {e}"

@tool
def send_gmail_tool(pdf_path: str = "", text_content: str = "", subject_suffix: str = "Document", custom_emails: str = "") -> str:
    """Send generated document via Gmail."""
    try:
        DIST = os.getenv("GMAIL_DISTRIBUTION_LIST","").split(",")
        recipients = [e.strip() for e in DIST if e.strip()]
        if custom_emails: recipients.extend([e.strip() for e in custom_emails.split(",") if e.strip()])
        if not recipients: return "No recipients provided."
        SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
        creds  = None
        tp, cp = Path("token.json"), Path("credentials.json")
        if not cp.exists(): return "❌ credentials.json not found. Gmail requires Google OAuth setup."
        if tp.exists(): creds = Credentials.from_authorized_user_file(str(tp), SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token: creds.refresh(Request())
            else:
                flow  = InstalledAppFlow.from_client_secrets_file(str(cp), SCOPES)
                creds = flow.run_local_server(port=0)
            tp.write_text(creds.to_json())
        svc     = build("gmail","v1",credentials=creds)
        subject = f"IncidentIQ - {subject_suffix} - {datetime.now().strftime('%d/%m/%Y')}"
        body    = f"Dear colleague,\n\nPlease find the AI-generated {subject_suffix}.\n\n"
        if text_content: body += f"{text_content}\n\n"
        body   += "Generated by IncidentIQ AI.\n\nBest regards,\nIncidentIQ"
        msg     = MIMEMultipart()
        msg["From"]="me"; msg["To"]=", ".join(recipients); msg["Subject"]=subject
        msg.attach(MIMEText(body,"plain"))
        if pdf_path and Path(pdf_path).exists():
            with open(pdf_path,"rb") as f:
                part = MIMEBase("application","octet-stream"); part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition",f"attachment; filename={Path(pdf_path).name}")
            msg.attach(part)
        svc.users().messages().send(userId="me",body={"raw":base64.urlsafe_b64encode(msg.as_bytes()).decode()}).execute()
        return f"✓ Verstuurd naar: {', '.join(recipients)}"
    except Exception as e: return f"Error: {e}"

# ── Agent ──────────────────────────────────────────────────────────────────────
def build_agent():
    TOOLS  = [search_video_knowledge, generate_xvr_scenario, generate_visual_summary, send_gmail_tool]
    PROMPT = """You are IncidentIQ, an AI agent for incident training.
Video loading is handled separately — never ask user to load a video.
ROUTING: Question->search_video_knowledge | XVR->generate_xvr_scenario | Visual->generate_visual_summary | Email->send_gmail_tool
LANGUAGE: Always respond in EXACT same language as user. Dutch->Dutch. English->English. French->French.
FORMAT: Bullet points, max 15 words per bullet."""
    lw = llm.bind_tools(TOOLS)
    def agent_node(state: MessagesState):
        return {"messages":[lw.invoke([SystemMessage(content=PROMPT)]+state["messages"])]}
    b = StateGraph(MessagesState)
    b.add_node("agent",agent_node); b.add_node("tools",ToolNode(TOOLS))
    b.add_edge(START,"agent"); b.add_conditional_edges("agent",tools_condition); b.add_edge("tools","agent")
    return b.compile(checkpointer=MemorySaver())

print("Building agent...")
agent = build_agent()
print("Agent ready!")

def ask(message, language="Nederlands"):
    config = {"configurable":{"thread_id":S["thread_id"]}}
    t0=time.time(); final=""; calls=[]; rid=uuid.uuid4().hex[:8]; S["run_id"]=rid
    for event in agent.stream({"messages":[HumanMessage(content=message)]},config=config,stream_mode="values"):
        last=event["messages"][-1]
        if hasattr(last,"tool_calls") and last.tool_calls:
            for tc in last.tool_calls: calls.append(tc["name"])
        if hasattr(last,"content") and isinstance(last.content,str) and last.content.strip():
            final=last.content.strip()
    lat=time.time()-t0; tok=max(len(message.split())*2,len(final.split())*2); cost=tok*0.00000015
    for tc in calls:
        add_trace(tc, f"lang:{LANG_TOOL.get(language,'dutch')}", lat=lat/max(len(calls),1), tokens=tok//max(len(calls),1), cost=cost/max(len(calls),1))
    return final, calls, lat

# ── Timeline renderer ──────────────────────────────────────────────────────────
def render_timeline(json_str):
    try: data = json.loads(json_str)
    except: return "<div style='color:#aaa;padding:20px'>Geen tijdlijn data.</div>"
    CM = {
        "red":   ("#C0392B","#1a0d0d","#ff6b6b"),
        "amber": ("#E67E22","#1a1200","#ffa94d"),
        "green": ("#1D9E75","#0d1a14","#69db7c"),
        "blue":  ("#4a9eff","#0d1220","#74c0fc"),
    }
    html = (
        f"<div style='background:linear-gradient(135deg,#1C2833,#2C3E50);padding:20px 24px;border-radius:12px;margin-bottom:20px'>"
        f"<div style='font-size:9px;letter-spacing:0.12em;color:#C0392B;background:#C0392B22;padding:3px 10px;border-radius:4px;"
        f"border:1px solid #C0392B44;display:inline-block;margin-bottom:10px;font-weight:500'>INCIDENT ANALYSIS</div>"
        f"<div style='font-size:19px;font-weight:500;color:white;margin-bottom:4px'>{data.get('title','')}</div>"
        f"<div style='font-size:12px;color:rgba(255,255,255,0.5)'>{data.get('subtitle','')} · {data.get('duration','')}</div>"
        f"</div>"
    )
    metrics = data.get("metrics",[])
    if metrics:
        html += "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px'>"
        for m in metrics:
            ch,bh,th = CM.get(m.get("color","blue"),CM["blue"])
            html += (
                f"<div style='background:#1a1a1a;border-radius:10px;padding:14px;border-bottom:3px solid {ch}'>"
                f"<div style='font-size:24px;font-weight:500;color:white;font-family:monospace;line-height:1'>"
                f"{m.get('value','')}<span style='font-size:12px;color:#555;font-weight:400'> {m.get('unit','')}</span></div>"
                f"<div style='font-size:11px;color:#777;margin-top:5px'>{m.get('label','')}</div></div>"
            )
        html += "</div>"
    html += "<div style='font-size:10px;letter-spacing:0.09em;color:#555;font-weight:500;margin-bottom:12px'>INCIDENT TIJDLIJN</div>"
    for ev in data.get("timeline",[]):
        ch,bh,th = CM.get(ev.get("color","blue"),CM["blue"])
        qh  = (f"<div style='border-left:2px solid {ch};padding-left:10px;margin:8px 0;font-size:12px;color:#aaa;"
               f"font-style:italic;line-height:1.6'>{ev['quote']}</div>") if ev.get("quote") else ""
        tgs = "".join([f"<span style='font-size:10px;padding:2px 8px;border-radius:4px;background:#2a2a2a;color:#777;margin-right:4px'>{t}</span>" for t in ev.get("tags",[])])
        html += (
            f"<div style='display:flex;gap:0;margin-bottom:2px'>"
            f"<div style='width:52px;flex-shrink:0;padding-top:14px;text-align:right;padding-right:10px'>"
            f"<span style='font-size:10px;color:#555;font-family:monospace'>{ev.get('timestamp','')}</span></div>"
            f"<div style='width:20px;flex-shrink:0;display:flex;flex-direction:column;align-items:center;padding-top:14px'>"
            f"<div style='width:10px;height:10px;border-radius:50%;background:{ch};box-shadow:0 0 0 3px {ch}33;flex-shrink:0'></div>"
            f"<div style='width:1px;flex:1;background:#2a2a2a;min-height:20px'></div></div>"
            f"<div style='flex:1;padding:8px 0 16px 12px'>"
            f"<div style='background:#1a1a1a;border-radius:10px;padding:14px 16px;border:1px solid #2a2a2a;border-left:3px solid {ch}'>"
            f"<div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px;gap:8px'>"
            f"<div style='font-size:13px;font-weight:500;color:white'>{ev.get('title','')}</div>"
            f"<span style='font-size:10px;padding:2px 8px;border-radius:4px;background:{bh};color:{th};font-weight:500;white-space:nowrap;flex-shrink:0'>{ev.get('badge','')}</span></div>"
            f"<div style='font-size:12px;color:#aaa;line-height:1.7'>{ev.get('text','')}</div>"
            f"{qh}<div style='margin-top:8px'>{tgs}</div>"
            f"</div></div></div>"
        )
    html += "<div style='font-size:10px;letter-spacing:0.09em;color:#555;font-weight:500;margin:16px 0 10px'>KEY LEARNINGS</div>"
    html += "<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px'>"
    for l in data.get("learnings",[]):
        html += (
            f"<div style='background:#1a1a1a;border-radius:10px;padding:14px;border:1px solid #2a2a2a'>"
            f"<div style='font-size:10px;color:#C0392B;font-family:monospace;font-weight:500;margin-bottom:6px'>{l.get('number','')}</div>"
            f"<div style='font-size:12px;font-weight:500;color:white;margin-bottom:5px'>{l.get('title','')}</div>"
            f"<div style='font-size:11px;color:#777;line-height:1.6'>{l.get('text','')}</div></div>"
        )
    html += "</div>"
    return html

# ── Handlers ───────────────────────────────────────────────────────────────────
NO_VID = {
    "Nederlands": "Hallo! Ik ben IncidentIQ.\n\nDrop een YouTube URL om te beginnen. Dan kan ik:\n• Vragen beantwoorden over het incident\n• Een professionele cheatsheet genereren\n• Een visuele tijdlijn maken\n• Een XVR simulatie scenario opstellen\n• Documenten versturen naar je team",
    "English":    "Hello! I'm IncidentIQ.\n\nDrop a YouTube URL to get started. Then I can:\n• Answer questions about the incident\n• Generate a professional cheatsheet\n• Create a visual timeline\n• Build an XVR simulation scenario\n• Send documents to your team",
    "Français":   "Bonjour! Je suis IncidentIQ.\n\nCollez une URL YouTube pour commencer:\n• Répondre aux questions\n• Générer une fiche de synthèse\n• Créer une chronologie visuelle\n• Construire un scénario XVR\n• Envoyer des documents",
}

def handle_chat(message, history, language):
    if not message.strip(): return history, render_trace(False), video_status_html(), ""
    history = history or []
    is_url  = "youtube.com" in message or "youtu.be" in message
    if is_url:
        response = load_video(message, language)
        history.append({"role":"user","content":message})
        history.append({"role":"assistant","content":response})
        return history, render_trace(False), video_status_html(), ""
    if not S["video_loaded"]:
        history.append({"role":"user","content":message})
        history.append({"role":"assistant","content":NO_VID.get(language,NO_VID["Nederlands"])})
        return history, render_trace(False), video_status_html(), ""
    response, calls, lat = ask(message, language)
    history.append({"role":"user","content":message})
    history.append({"role":"assistant","content":response})
    return history, render_trace(False), video_status_html(), ""

def handle_pdf(language):
    if not S["video_loaded"]:
        return "<div style='color:#aaa;padding:20px'>⚠️ Laad eerst een video in de chat.</div>", None, render_trace(False)
    lt = LANG_TOOL.get(language,"dutch")
    t0 = time.time()
    lang = {"dutch":"Dutch","english":"English","french":"French"}.get(lt,"Dutch")
    results = vs.similarity_search("key points lessons learned recommendations conclusions", k=12)
    if not results:
        return "<div style='color:#aaa;padding:20px'>❌ Geen video data gevonden.</div>", None, render_trace(False)
    context = "\n\n".join([r.page_content for r in results])
    try:
        raw = re.sub(r'```json|```','', llm.invoke(
            f'Extract structured info for incident cheatsheet in {lang}.\n'
            f'Return ONLY JSON: {{"title":"...","subtitle":"...","tags":["tag1","tag2","tag3"],"keypoints":["..."],"recommendations":["..."]}}\n'
            f'Rules: max 15 words per item, no timestamps, 3-5 tags.\n\nContext:\n{context}\n\nJSON:'
        ).content.strip()).strip()
        data = json.loads(raw)
        S["pdf_data"] = data
        fp = make_pdf(data, S.get("video_url",""))
        S["pdf_path"] = fp
        add_trace("generate_pdf_cheatsheet", f"lang:{lt} · {data.get('title','')}", lat=time.time()-t0, tokens=500, cost=0.000075)
        return render_key_concepts(data), fp, render_trace(False)
    except Exception as e:
        return f"<div style='color:#aaa;padding:20px'>❌ Error: {e}</div>", None, render_trace(False)

def handle_timeline(language):
    if not S["video_loaded"]:
        return "<div style='color:#aaa;padding:20px'>⚠️ Laad eerst een video in de chat.</div>", render_trace(False)
    lt = LANG_TOOL.get(language,"dutch")
    t0 = time.time()
    result  = generate_visual_summary.invoke({"language": lt})
    cleaned = result.strip()
    if "{" in cleaned and "}" in cleaned:
        cleaned = cleaned[cleaned.index("{"):cleaned.rindex("}")+1]
    try:
        json.loads(cleaned)
        S["visual_json"] = cleaned
        add_trace("generate_visual_summary", f"lang:{lt}", lat=time.time()-t0, tokens=600, cost=0.00009)
        return render_timeline(cleaned), render_trace(False)
    except Exception as e:
        return f"<div style='color:#aaa;padding:20px'>❌ Error: {e}<br><br>{result[:300]}</div>", render_trace(False)

def handle_xvr(language):
    if not S["video_loaded"]:
        return "⚠️ Laad eerst een video in de chat.", None, render_trace(False)
    lt     = LANG_TOOL.get(language,"dutch")
    t0     = time.time()
    result = generate_xvr_scenario.invoke({"language": lt})
    if not result or len(result) < 50:
        return f"❌ Lege response: {result}", None, render_trace(False)
    S["xvr_content"] = result
    add_trace("generate_xvr_scenario", f"lang:{lt}", lat=time.time()-t0, tokens=700, cost=0.000105)
    xvr_path = f'/tmp/xvr_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    Path(xvr_path).write_text(result)
    return result, xvr_path, render_trace(False)

def handle_send(email_to, doc_choice, language):
    if not email_to.strip():
        return "⚠️ Vul een e-mailadres in.", render_trace(False)
    if not Path("credentials.json").exists():
        return "⚠️ credentials.json niet gevonden. Zie README voor Gmail setup.", render_trace(False)
    t0 = time.time()
    try:
        if "PDF" in doc_choice and S["pdf_path"] and Path(S["pdf_path"]).exists():
            result = send_gmail_tool.invoke({"pdf_path":S["pdf_path"],"subject_suffix":"Key Concepts Cheatsheet","custom_emails":email_to})
        elif "XVR" in doc_choice and S["xvr_content"]:
            result = send_gmail_tool.invoke({"text_content":S["xvr_content"],"subject_suffix":"XVR Scenario Brief","custom_emails":email_to})
        elif S["visual_json"]:
            result = send_gmail_tool.invoke({"text_content":"Visual Summary bijgevoegd.","subject_suffix":"Visual Summary","custom_emails":email_to})
        else:
            return "⚠️ Genereer eerst een document via de tabs hierboven.", render_trace(False)
        add_trace("send_gmail_tool", f"to:{email_to}", lat=time.time()-t0)
        return result, render_trace(False)
    except Exception as e:
        return f"❌ Gmail error: {e}", render_trace(False)

def update_buttons(lang):
    b = BTN.get(lang, BTN["Nederlands"])
    return (
        gr.update(value=b["pdf"]),
        gr.update(value=b["tl"]),
        gr.update(value=b["xvr"]),
        gr.update(value=b["send"]),
    )

# ── UI ─────────────────────────────────────────────────────────────────────────
CSS = """
body { background: #0f0f0f !important; }
.gradio-container { background: #0f0f0f !important; max-width: 100% !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="IncidentIQ") as demo:

    gr.HTML("""
    <div style='display:flex;align-items:center;justify-content:space-between;
                padding:16px 20px;background:#0f0f0f;border-bottom:1px solid #1e1e1e;margin-bottom:16px'>
        <div style='display:flex;align-items:center;gap:12px'>
            <div style='width:36px;height:36px;background:#C0392B;border-radius:9px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:14px;font-weight:500;color:white'>IQ</div>
            <div>
                <div style='font-size:16px;font-weight:500;color:white'>IncidentIQ</div>
                <div style='font-size:11px;color:#555'>AI Incident Intelligence</div>
            </div>
        </div>
        <div style='display:flex;gap:8px'>
            <span style='font-size:10px;color:#C0392B;background:#C0392B11;padding:3px 10px;border-radius:20px;border:1px solid #C0392B44'>gpt-4o-mini</span>
            <span style='font-size:10px;color:#4a9eff;background:#4a9eff11;padding:3px 10px;border-radius:20px;border:1px solid #4a9eff44'>Pinecone</span>
            <span style='font-size:10px;color:#1D9E75;background:#1D9E7511;padding:3px 10px;border-radius:20px;border:1px solid #1D9E7544'>LangSmith</span>
        </div>
    </div>""")

    with gr.Row():
        # SIDEBAR
        with gr.Column(scale=1, min_width=240):
            language     = gr.Radio(["Nederlands","English","Français"], value="Nederlands", label="🌐 Taal")
            video_status = gr.HTML(video_status_html())
            gr.HTML("<div style='font-size:10px;letter-spacing:0.08em;color:#444;font-weight:500;margin:10px 0 6px'>VERSTUREN</div>")
            email_to    = gr.Textbox(placeholder="naam@email.be", label="Naar")
            doc_choice  = gr.Dropdown(BTN["Nederlands"]["doc_opts"], value=BTN["Nederlands"]["doc_opts"][0], label="Document")
            btn_send    = gr.Button(BTN["Nederlands"]["send"], variant="primary", size="sm")
            send_result = gr.Markdown("")
            gr.HTML("<div style='font-size:10px;letter-spacing:0.08em;color:#444;font-weight:500;margin:10px 0 6px'>AGENT ACTIVITEIT</div>")
            pro_toggle = gr.Checkbox(label="Pro mode", value=False)
            trace_html = gr.HTML(render_trace(False))
            if os.getenv("LANGSMITH_API_KEY"):
                gr.HTML("<a href='https://smith.langchain.com' target='_blank' style='display:block;padding:7px;border-radius:7px;border:1px solid #2a2a2a;font-size:11px;color:#C0392B;text-decoration:none;text-align:center;margin-top:8px'>📊 LangSmith traces →</a>")

        # MAIN
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("💬  Chat"):
                    chatbot = gr.Chatbot(value=[], height=480, label="Chat", autoscroll=True)
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Stel een vraag of drop een YouTube URL...",
                            label="Input", lines=1, scale=10
                        )
                        send_chat = gr.Button("→", scale=1, variant="primary", size="sm")

                with gr.Tab("📄  Key Concepts"):
                    btn_pdf    = gr.Button(BTN["Nederlands"]["pdf"], variant="primary")
                    kc_html    = gr.HTML("<div style='color:#777;text-align:center;padding:40px 20px'>Klik de knop hierboven om key concepts te genereren.</div>")
                    pdf_file   = gr.File(label="Download PDF", visible=True)

                with gr.Tab("📊  Tijdlijn"):
                    btn_tl        = gr.Button(BTN["Nederlands"]["tl"], variant="primary")
                    timeline_html = gr.HTML("<div style='color:#777;text-align:center;padding:40px 20px'>Klik de knop hierboven om de tijdlijn te genereren.</div>")

                with gr.Tab("🎮  XVR Scenario"):
                    btn_xvr      = gr.Button(BTN["Nederlands"]["xvr"], variant="primary")
                    xvr_output   = gr.Textbox(label="XVR Scenario Brief", lines=20, placeholder="Klik de knop hierboven.")
                    xvr_download = gr.File(label="Download scenario", visible=True)

    # Events
    msg_input.submit(handle_chat,  [msg_input,chatbot,language], [chatbot,trace_html,video_status,msg_input])
    send_chat.click(handle_chat,   [msg_input,chatbot,language], [chatbot,trace_html,video_status,msg_input])
    btn_pdf.click(handle_pdf,      [language], [kc_html,pdf_file,trace_html])
    btn_tl.click(handle_timeline,  [language], [timeline_html,trace_html])
    btn_xvr.click(handle_xvr,     [language], [xvr_output,xvr_download,trace_html])
    btn_send.click(handle_send,    [email_to,doc_choice,language], [send_result,trace_html])
    pro_toggle.change(lambda p: render_trace(p), [pro_toggle], [trace_html])
    language.change(update_buttons, [language], [btn_pdf,btn_tl,btn_xvr,btn_send])
    demo.load(update_buttons, [language], [btn_pdf,btn_tl,btn_xvr,btn_send,doc_choice])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=CSS,
        allowed_paths=["/tmp"],
    )
