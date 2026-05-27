"""
IncidentIQ — Flask Web App
Converted from Gradio to Flask + HTML/JS frontend with Pinecone Namespaces
"""

import os, re, json, time, base64, uuid, tempfile
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, send_file

load_dotenv()
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT']    = 'incidentiq-agent'
if os.getenv('LANGSMITH_API_KEY'):
    os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGSMITH_API_KEY')

# ── Lazy imports (only loaded when first needed) ───────────────────────────────
_llm = None
_emb = None
_vs  = None
_agent = None

def get_llm():
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _llm

def get_emb():
    global _emb
    if _emb is None:
        from langchain_openai import OpenAIEmbeddings
        _emb = OpenAIEmbeddings(model="text-embedding-3-small")
    return _emb

def get_vs():
    global _vs
    if _vs is None:
        from langchain_pinecone import PineconeVectorStore
        _vs = PineconeVectorStore(
            index_name="incidentiq",
            embedding=get_emb(),
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        )
    return _vs

def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent

# ── Global state ────────────────────────────────────────────────────────────────
STATE = {
    "video_loaded": False,
    "video_id": "",
    "video_title": "",
    "video_url": "",
    "pdf_path": None,
    "xvr_content": "",
    "visual_json": "",
    "thread_id": f"s_{uuid.uuid4().hex[:8]}",
    "chat_history": [],
    "trace": [],
}

# ── Flask app ───────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Helpers ─────────────────────────────────────────────────────────────────────
def get_vid_id(url):
    if "v=" in url: return url.split("v=")[1].split("&")[0]
    if "youtu.be/" in url: return url.split("youtu.be/")[1].split("?")[0]
    raise ValueError("Cannot extract video ID")

def clean_tx(t):
    t = re.sub(r'\[Music\]|\[Applause\]|\[Laughter\]|\[Cheering\]', '', t)
    t = re.sub(r'\[\d{2}:\d{2}\]\s*(?=\[\d{2}:\d{2}\])', '', t)
    return re.sub(r'\s+', ' ', t).strip()

def add_trace(label, detail="", lat=None):
    STATE["trace"].append({"label": label, "detail": detail, "lat": lat})
    if len(STATE["trace"]) > 10:
        STATE["trace"] = STATE["trace"][-10:]

def search_pinecone(query, k=8):
    try:
        # Haal de huidige video_id op om als namespace te gebruiken
        ns = STATE.get("video_id")
        if not ns:
            return "No video loaded to search context."
            
        results = get_vs().similarity_search(query, k=k, namespace=ns)
        if not results: return "No relevant information found."
        ts_all = re.findall(r"\[\d{2}:\d{2}\]", " ".join(d.page_content for d in results))
        seen, uts = set(), []
        for t in ts_all:
            if t not in seen: seen.add(t); uts.append(t)
        clean = [re.sub(r"\[\d{2}:\d{2}\]\s*(?=\[\d{2}:\d{2}\])","",d.page_content) for d in results]
        sources = f"\n\nSources: {' | '.join(uts[:5])}" if uts else ""
        return "\n\n".join(clean) + sources
    except Exception as e:
        return f"Search error: {e}"

# ── Agent ───────────────────────────────────────────────────────────────────────
def build_agent():
    from langchain.tools import tool
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langgraph.prebuilt import ToolNode, tools_condition
    from langgraph.checkpoint.memory import MemorySaver

    llm = get_llm()

    @tool
    def search_video_knowledge(query: str) -> str:
        """Search the loaded incident video for information."""
        t0 = time.time()
        ns = STATE.get("video_id")
        if not ns:
            return "No active video context found. Please load a video first."
            
        try:
            variations = json.loads(re.sub(r'```json|```','', llm.invoke(
                f'Generate 3 search query variations as JSON array. Query: "{query}"\nJSON:'
            ).content.strip()).strip())
            if not isinstance(variations, list): variations = [query]
        except:
            variations = [query]
        variations.append(query)
        all_docs = {}
        for q in variations:
            try:
                # Zoeken binnen de specifieke video namespace
                for doc in get_vs().similarity_search(q, k=4, namespace=ns):
                    key = doc.page_content[:80]
                    if key not in all_docs: all_docs[key] = doc
            except: pass
        if not all_docs:
            return "No relevant information found in the video."
        combined = list(all_docs.values())
        ts_all = re.findall(r"\[\d{2}:\d{2}\]", " ".join(d.page_content for d in combined))
        seen, uts = set(), []
        for t in ts_all:
            if t not in seen: seen.add(t); uts.append(t)
        clean_docs = [re.sub(r"\[\d{2}:\d{2}\]\s*(?=\[\d{2}:\d{2}\])","",d.page_content) for d in combined]
        add_trace("search", f"q:{query[:30]}", lat=time.time()-t0)
        return "\n\n".join(clean_docs) + (f"\n\nSources: {' | '.join(uts[:5])}" if uts else "")

    @tool
    def generate_xVR_scenario_tool(dummy: str = "") -> str:
        """Generate a complete XVR simulation scenario."""
        t0 = time.time()
        context = search_pinecone("location building fire cause complications decisions resources casualties", k=12)
        result = llm.invoke(
            f"Generate a complete XVR operator scenario brief in English.\n\n"
            f"SCENARIO BRIEF\n==============\n\n"
            f"INCIDENT TITLE: [title]\n\nLOCATION: [building type, floors]\n\n"
            f"INITIAL SITUATION T+00:00:\n- [fire location, casualties, resources]\n\n"
            f"COMPLICATIONS:\n- T+[time]: [complication]\n- T+[time]: [complication]\n\n"
            f"DECISION MOMENTS:\n1. [decision]\n2. [decision]\n\n"
            f"LEARNING OBJECTIVES:\n- [objective]\n- [objective]\n\n"
            f"DEBRIEFING QUESTIONS:\n1. [question]\n2. [question]\n\n"
            f"Base ONLY on context. Never invent.\n\nContext:\n{context}"
        ).content.strip()
        add_trace("xvr", lat=time.time()-t0)
        return result

    @tool
    def generate_timeline_tool(dummy: str = "") -> str:
        """Generate a visual timeline JSON from the loaded incident video."""
        t0 = time.time()
        context = search_pinecone("notification arrival problems complications solutions outcome result", k=12)
        raw = re.sub(r'```json|```','', llm.invoke(
            f'Extract facts from context. Return raw JSON:\n'
            f'{{"title":"[title]","subtitle":"[source]","duration":"[duration or unknown]",'
            f'"metrics":[{{"value":"[val]","unit":"[u]","label":"[l]","color":"blue"}}],'
            f'"timeline":['
            f'{{"timestamp":"[t]","title":"Notification","text":"[what was reported]","quote":"","tags":["notification"],"color":"blue","badge":"Notification"}},'
            f'{{"timestamp":"[t]","title":"Arrival","text":"[situation on arrival]","quote":"","tags":["arrival"],"color":"amber","badge":"Arrival"}},'
            f'{{"timestamp":"[t]","title":"Problems","text":"[complications]","quote":"","tags":["problem"],"color":"red","badge":"Complication"}},'
            f'{{"timestamp":"[t]","title":"Solutions","text":"[actions taken]","quote":"","tags":["action"],"color":"amber","badge":"Action"}},'
            f'{{"timestamp":"[t]","title":"End","text":"[outcome]","quote":"","tags":["outcome"],"color":"green","badge":"Outcome"}}],'
            f'"learnings":['
            f'{{"number":"01","title":"[t]","text":"[2 sentences]"}},'
            f'{{"number":"02","title":"[t]","text":"[2 sentences]"}},'
            f'{{"number":"03","title":"[t]","text":"[2 sentences]"}},'
            f'{{"number":"04","title":"[t]","text":"[2 sentences]"}}],'
            f'"source_url":""}}\n\nFacts only. "Not mentioned" if absent.\n\nContext:\n{context}\n\nJSON:'
        ).content.strip()).strip()
        if "{" in raw and "}" in raw:
            raw = raw[raw.index("{"):raw.rindex("}")+1]
        json.loads(raw)  # validate
        add_trace("timeline", lat=time.time()-t0)
        return raw

    TOOLS = [search_video_knowledge, generate_xVR_scenario_tool, generate_timeline_tool]
    PROMPT = """You are IncidentIQ, an AI agent for incident training.
Always respond in English. Be concise — bullet points, max 15 words per bullet.
Always end answers with Sources: [timestamps] when available.
ROUTING:
- Any question about the video → search_video_knowledge
- XVR scenario request → generate_xVR_scenario_tool
- Timeline/visual request → generate_timeline_tool"""

    lw = llm.bind_tools(TOOLS)
    def agent_node(state: MessagesState):
        from langchain_core.messages import SystemMessage
        return {"messages": [lw.invoke([SystemMessage(content=PROMPT)] + state["messages"])]}

    b = StateGraph(MessagesState)
    b.add_node("agent", agent_node)
    b.add_node("tools", ToolNode(TOOLS))
    b.add_edge(START, "agent")
    b.add_conditional_edges("agent", tools_condition)
    b.add_edge("tools", "agent")
    return b.compile(checkpointer=MemorySaver())

def ask_agent(message):
    from langchain_core.messages import HumanMessage
    config = {"configurable": {"thread_id": STATE["thread_id"]}}
    final = ""
    for event in get_agent().stream(
        {"messages": [HumanMessage(content=message)]},
        config=config, stream_mode="values"
    ):
        last = event["messages"][-1]
        if hasattr(last, "content") and isinstance(last.content, str) and last.content.strip():
            final = last.content.strip()
    return final

# ── PDF generation ───────────────────────────────────────────────────────────────
def make_pdf(data, source_url=""):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.colors import HexColor, white
    from reportlab.pdfgen import canvas as rl_canvas
    RED=HexColor("#C0392B"); DARK=HexColor("#1C2833"); ORANGE=HexColor("#E67E22")
    GREEN=HexColor("#1E8449"); WHITE=white
    fp = f'/tmp/iq_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    c  = rl_canvas.Canvas(fp, pagesize=A4); W, H = A4
    c.setFillColor(RED);   c.rect(0,H-3.2*cm,W,3.2*cm,fill=1,stroke=0)
    c.setFillColor(WHITE); c.circle(1.8*cm,H-1.6*cm,0.85*cm,fill=1,stroke=0)
    c.setFillColor(RED);   c.setFont("Helvetica-Bold",14); c.drawCentredString(1.8*cm,H-1.95*cm,"IQ")
    c.setFillColor(WHITE); c.setFont("Helvetica-Bold",14); c.drawString(3.2*cm,H-1.3*cm,str(data.get("title",""))[:50])
    c.setFont("Helvetica",10); c.drawString(3.2*cm,H-1.85*cm,str(data.get("subtitle",""))[:60])
    c.setFont("Helvetica",8)
    c.drawRightString(W-1.2*cm,H-1.3*cm,datetime.now().strftime("%d/%m/%Y"))
    c.drawRightString(W-1.2*cm,H-1.75*cm,"Generated by IncidentIQ AI")
    c.setFillColor(ORANGE); c.rect(0,H-3.6*cm,W,0.4*cm,fill=1,stroke=0)
    y = H-5.0*cm
    def sh(y, t, col=DARK):
        c.setFillColor(col); c.setFont("Helvetica-Bold",11); c.drawString(1.2*cm,y,t.upper())
        c.setStrokeColor(col); c.setLineWidth(1.5); c.line(1.2*cm,y-0.2*cm,W-1.2*cm,y-0.2*cm)
        return y-0.8*cm
    def bi(y, txt, col=DARK, bc=RED):
        c.setFillColor(bc); c.circle(1.5*cm,y+0.25*cm,0.1*cm,fill=1,stroke=0)
        c.setFillColor(col); c.setFont("Helvetica",10); mw=W-1.8*cm-1.2*cm
        words=str(txt).split(); line, lines="",[]
        for w in words:
            t2=line+w+" "
            if c.stringWidth(t2,"Helvetica",10)<mw: line=t2
            else: lines.append(line.strip()); line=w+" "
        lines.append(line.strip())
        for i, l in enumerate(lines): c.drawString(1.8*cm,y-i*0.5*cm,l)
        return y-len(lines)*0.5*cm-0.35*cm
    y=sh(y,"Key Points",RED)
    for kp in data.get("keypoints",[]): y=bi(y,kp)
    y-=0.5*cm; y=sh(y,"AI Recommendations",GREEN)
    for rec in data.get("recommendations",[]): y=bi(y,rec,bc=GREEN)
    c.setFillColor(RED);  c.rect(0,1.2*cm,W,0.15*cm,fill=1,stroke=0)
    c.setFillColor(DARK); c.rect(0,0,W,1.2*cm,fill=1,stroke=0)
    c.setFillColor(WHITE); c.setFont("Helvetica",7.5)
    c.drawString(1.2*cm,0.65*cm,"IncidentIQ — AI-powered Incident Intelligence")
    if source_url: c.drawCentredString(W/2,0.65*cm,f"Source: {source_url[:80]}")
    c.drawRightString(W-1.2*cm,0.65*cm,"Page 1/1"); c.save()
    return fp

# ── Gmail ────────────────────────────────────────────────────────────────────────
def send_gmail(to_email, subject_suffix, body_text, pdf_path=None):
    import base64 as b64
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders
    SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
    tp, cp = Path("token.json"), Path("credentials.json")
    if not cp.exists():
        return "❌ credentials.json not found"
    creds = None
    if tp.exists():
        creds = Credentials.from_authorized_user_file(str(tp), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow  = InstalledAppFlow.from_client_secrets_file(str(cp), SCOPES)
            creds = flow.run_local_server(port=0)
        tp.write_text(creds.to_json())
    svc = build("gmail","v1",credentials=creds)
    msg = MIMEMultipart()
    msg["From"]    = "me"
    msg["To"]      = to_email
    msg["Subject"] = f"IncidentIQ — {subject_suffix} — {datetime.now().strftime('%d/%m/%Y')}"
    msg.attach(MIMEText(f"Dear colleague,\n\n{body_text}\n\nGenerated by IncidentIQ AI.", "plain"))
    if pdf_path and Path(pdf_path).exists():
        with open(pdf_path,"rb") as f:
            part = MIMEBase("application","octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition","attachment; filename=IncidentIQ.pdf")
        msg.attach(part)
    svc.users().messages().send(
        userId="me",
        body={"raw": b64.urlsafe_b64encode(msg.as_bytes()).decode()}
    ).execute()
    return f"✓ Sent to {to_email}"

# ── Flask routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/state")
def api_state():
    return jsonify({
        "video_loaded": STATE["video_loaded"],
        "video_title":  STATE["video_title"],
        "video_url":    STATE["video_url"],
        "trace":        STATE["trace"],
        "chat_history": STATE["chat_history"],
    })

@app.route("/api/reset", methods=["POST"])
def api_reset():
    STATE["video_loaded"]  = False
    STATE["video_id"]      = ""
    STATE["video_title"]   = ""
    STATE["video_url"]     = ""
    STATE["pdf_path"]      = None
    STATE["xvr_content"]   = ""
    STATE["visual_json"]   = ""
    STATE["trace"]         = []
    STATE["chat_history"]  = []
    STATE["thread_id"]     = f"s_{uuid.uuid4().hex[:8]}"
    return jsonify({"ok": True})

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data    = request.json
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    # URL → load video
    if "youtube.com" in message or "youtu.be" in message:
        result = load_video(message)
        STATE["chat_history"].append({"role":"user","content":message})
        STATE["chat_history"].append({"role":"assistant","content":result})
        return jsonify({"response": result, "trace": STATE["trace"],
                        "video_loaded": STATE["video_loaded"],
                        "video_title":  STATE["video_title"]})

    if not STATE["video_loaded"]:
        response = "Please paste a YouTube URL first to load a video."
        STATE["chat_history"].append({"role":"user","content":message})
        STATE["chat_history"].append({"role":"assistant","content":response})
        return jsonify({"response": response, "trace": STATE["trace"],
                        "video_loaded": False, "video_title": ""})

    response = ask_agent(message)
    STATE["chat_history"].append({"role":"user","content":message})
    STATE["chat_history"].append({"role":"assistant","content":response})
    return jsonify({"response": response, "trace": STATE["trace"],
                    "video_loaded": STATE["video_loaded"],
                    "video_title":  STATE["video_title"]})

@app.route("/api/voice", methods=["POST"])
def api_voice():
    """Receive base64 WAV audio, transcribe with Whisper, return text."""
    try:
        audio_b64 = request.json.get("audio_b64","")
        if not audio_b64:
            return jsonify({"text":""})
        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        with open(tmp_path, "rb") as af:
            result = client.audio.transcriptions.create(model="whisper-1", file=af)
        os.unlink(tmp_path)
        return jsonify({"text": result.text.strip()})
    except Exception as e:
        return jsonify({"text": "", "error": str(e)})

@app.route("/api/pdf", methods=["POST"])
def api_pdf():
    if not STATE["video_loaded"]:
        return jsonify({"error": "Load a video first."}), 400
    t0 = time.time()
    context = search_pinecone("key points lessons learned recommendations conclusions mistakes", k=12)
    try:
        raw = re.sub(r'```json|```','', get_llm().invoke(
            f'Extract cheatsheet info in English. Return ONLY JSON:\n'
            f'{{"title":"...","subtitle":"...","summary":"2-3 sentence overview","tags":["t1","t2","t3"],'
            f'"keypoints":["detailed point max 25 words"],"recommendations":["recommendation max 20 words"]}}\n'
            f'5-7 keypoints, 4-5 recommendations.\n\nContext:\n{context}\n\nJSON:'
        ).content.strip()).strip()
        data = json.loads(raw)
        fp   = make_pdf(data, STATE["video_url"])
        STATE["pdf_path"] = fp
        add_trace("pdf", lat=time.time()-t0)
        return jsonify({"data": data, "trace": STATE["trace"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/pdf/download")
def api_pdf_download():
    fp = STATE.get("pdf_path")
    if not fp or not Path(fp).exists():
        return "PDF not found", 404
    return send_file(fp, as_attachment=True, download_name="IncidentIQ_KeyConcepts.pdf")

@app.route("/api/timeline", methods=["POST"])
def api_timeline():
    if not STATE["video_loaded"]:
        return jsonify({"error": "Load a video first."}), 400
    t0 = time.time()
    context = search_pinecone("notification arrival problems complications solutions outcome", k=12)
    try:
        raw = re.sub(r'```json|```','', get_llm().invoke(
            f'Extract facts from context. Return raw JSON:\n'
            f'{{"title":"[title]","subtitle":"[source]","duration":"[duration or unknown]",'
            f'"metrics":[{{"value":"[v]","unit":"[u]","label":"[l]","color":"blue"}},'
            f'{{"value":"[v]","unit":"[u]","label":"[l]","color":"amber"}},'
            f'{{"value":"[v]","unit":"[u]","label":"[l]","color":"red"}},'
            f'{{"value":"[v]","unit":"","label":"[l]","color":"green"}}],'
            f'"timeline":['
            f'{{"timestamp":"[t]","title":"Notification","text":"[what was reported]","quote":"","tags":["notification"],"color":"blue","badge":"Notification"}},'
            f'{{"timestamp":"[t]","title":"Arrival","text":"[situation on arrival]","quote":"","tags":["arrival"],"color":"amber","badge":"Arrival"}},'
            f'{{"timestamp":"[t]","title":"Problems","text":"[complications]","quote":"","tags":["problem"],"color":"red","badge":"Complication"}},'
            f'{{"timestamp":"[t]","title":"Solutions","text":"[actions taken]","quote":"","tags":["action"],"color":"amber","badge":"Action"}},'
            f'{{"timestamp":"[t]","title":"End","text":"[outcome]","quote":"","tags":["outcome"],"color":"green","badge":"Outcome"}}],'
            f'"learnings":['
            f'{{"number":"01","title":"[t]","text":"[2 sentences]"}},'
            f'{{"number":"02","title":"[t]","text":"[2 sentences]"}},'
            f'{{"number":"03","title":"[t]","text":"[2 sentences]"}},'
            f'{{"number":"04","title":"[t]","text":"[2 sentences]"}}],'
            f'"source_url":""}}\n\nFacts only, never invent, "Not mentioned" if absent.\n\nContext:\n{context}\n\nJSON:'
        ).content.strip()).strip()
        if "{" in raw and "}" in raw:
            raw = raw[raw.index("{"):raw.rindex("}")+1]
        data = json.loads(raw)
        STATE["visual_json"] = raw
        add_trace("timeline", lat=time.time()-t0)
        return jsonify({"data": data, "trace": STATE["trace"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/xvr", methods=["POST"])
def api_xvr():
    if not STATE["video_loaded"]:
        return jsonify({"error": "Load a video first."}), 400
    t0 = time.time()
    context = search_pinecone("location building fire cause complications decisions resources casualties", k=12)
    result = get_llm().invoke(
        f"Generate a complete XVR operator scenario brief in English.\n\n"
        f"SCENARIO BRIEF\n==============\n\n"
        f"INCIDENT TITLE: [title from context]\n\n"
        f"LOCATION & BUILDING:\n- Type: [type]\n- Floors: [number]\n\n"
        f"INITIAL SITUATION T+00:00:\n- Fire: [location]\n- Casualties: [number]\n- Resources: [vehicles]\n\n"
        f"COMPLICATIONS:\n- T+[time]: [complication 1]\n- T+[time]: [complication 2]\n- T+[time]: [complication 3]\n\n"
        f"CRITICAL DECISIONS:\n1. [decision]\n2. [decision]\n\n"
        f"LEARNING OBJECTIVES:\n- [objective 1]\n- [objective 2]\n\n"
        f"DEBRIEFING QUESTIONS:\n1. [question based on actual mistakes]\n2. [question]\n3. [question]\n\n"
        f"XVR OPERATOR NOTES:\n[key moments to inject]\n\n"
        f"Base ONLY on context below.\n\nContext:\n{context}"
    ).content.strip()
    STATE["xvr_content"] = result
    xvr_path = f'/tmp/xvr_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    Path(xvr_path).write_text(result)
    add_trace("xvr", lat=time.time()-t0)
    return jsonify({"content": result, "trace": STATE["trace"]})

@app.route("/api/xvr/download")
def api_xvr_download():
    content = STATE.get("xvr_content","")
    if not content:
        return "XVR not generated yet", 404
    fp = f'/tmp/xvr_download.txt'
    Path(fp).write_text(content)
    return send_file(fp, as_attachment=True, download_name="IncidentIQ_XVR_Scenario.txt")

@app.route("/api/send", methods=["POST"])
def api_send():
    data       = request.json
    email_to   = (data.get("email_to") or "").strip()
    doc_choice = data.get("doc_choice","")
    if not email_to:
        return jsonify({"error": "Enter an email address."}), 400
    t0 = time.time()
    try:
        if "PDF" in doc_choice:
            if not STATE["pdf_path"] or not Path(STATE["pdf_path"]).exists():
                return jsonify({"error": "Generate a PDF first in the Key Concepts tab."}), 400
            result = send_gmail(email_to, "Key Concepts Cheatsheet",
                "Please find the attached Key Concepts cheatsheet.", STATE["pdf_path"])
        elif "XVR" in doc_choice:
            if not STATE["xvr_content"]:
                return jsonify({"error": "Generate an XVR Scenario first."}), 400
            result = send_gmail(email_to, "XVR Scenario Brief", STATE["xvr_content"])
        elif "Timeline" in doc_choice:
            if not STATE["visual_json"]:
                return jsonify({"error": "Generate a Timeline first."}), 400
            try:
                tl = json.loads(STATE["visual_json"])
                tl_text = f"INCIDENT TIMELINE: {tl.get('title','')}\n\n"
                for ev in tl.get("timeline",[]):
                    tl_text += f"[{ev.get('timestamp','')}] {ev.get('title','').upper()}\n{ev.get('text','')}\n\n"
                result = send_gmail(email_to, "Visual Timeline", tl_text)
            except:
                result = send_gmail(email_to, "Visual Timeline", STATE["visual_json"])
        else:
            return jsonify({"error": "Select a document type."}), 400
        add_trace("send", lat=time.time()-t0)
        return jsonify({"result": result, "trace": STATE["trace"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── Load video ───────────────────────────────────────────────────────────────────
def load_video(url):
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    try:
        video_id = get_vid_id(url)
    except Exception as e:
        return f"❌ Cannot extract video ID: {e}"
    t0 = time.time()
    
    # 1. CONTROLEER OF DE VIDEO AL BESTAAT IN ZIJN EIGEN NAMESPACE (Cache check)
    try:
        test = get_vs().similarity_search("incident", k=1, namespace=video_id)
        if test:
            STATE["video_loaded"] = True
            STATE["video_id"]     = video_id
            STATE["video_title"]  = f"Video {video_id}"
            STATE["video_url"]    = url
            add_trace("load_video", f"cache hit · {video_id}", lat=time.time()-t0)
            return f"✓ Video loaded from cache — ready!\n\nID: {video_id} (Namespace: {video_id})"
    except Exception as e:
        pass
        
    # 2. ALS DE VIDEO NIEET BESTAAT, TRANSCRIPT OPHALEN EN OPSLAAN IN DE NAMESPACE
    try:
        entries = YouTubeTranscriptApi().fetch(video_id, languages=["en","nl","fr"])
        txlist  = entries.snippets
    except NoTranscriptFound:   return f"❌ No transcript for {video_id}"
    except TranscriptsDisabled: return f"❌ Transcripts disabled for {video_id}"
    except Exception as e:      return f"❌ YouTube error: {e}"
    
    ts     = clean_tx(" ".join(f"[{int(t.start//60):02d}:{int(t.start%60):02d}] {t.text}" for t in txlist))
    spl    = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = spl.create_documents(texts=[ts], metadatas=[{"video_id":video_id,"source":url}])
    
    # Documenten pushen naar Pinecone binnen de video_id namespace
    get_vs().add_documents(chunks, namespace=video_id)
    
    STATE["video_loaded"] = True
    STATE["video_id"]     = video_id
    STATE["video_title"]  = f"Video {video_id}"
    STATE["video_url"]    = url
    add_trace("load_video", f"new · {video_id} · {len(chunks)} chunks", lat=time.time()-t0)
    return f"✓ Video loaded — {len(chunks)} chunks stored in namespace '{video_id}'.\n\nID: {video_id}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)