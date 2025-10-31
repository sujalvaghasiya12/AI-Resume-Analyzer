"""
AI Resume Analyzer Pro v2 — Enhanced Accuracy & Advanced Features
- Improved fuzzy matching with context-aware scoring
- Better role compatibility with weighted requirements
- Advanced proficiency calculation with multiple factors
- Strength/weakness analysis and detailed recommendations
- Enhanced UI with better visualizations and insights
- Better ATS scoring with modern best practices
"""
import streamlit as st
import json, os, re, base64, io, importlib
from io import BytesIO
import docx
import pdfplumber
import nltk
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from collections import Counter

# ---------- Optional: rapidfuzz (fast fuzzy) with fallback ----------
try:
    from rapidfuzz import fuzz, process
    _FUZZ_BACKEND = "rapidfuzz"
except Exception:
    from difflib import SequenceMatcher
    _FUZZ_BACKEND = "difflib"

    class fuzz:
        @staticmethod
        def token_set_ratio(a, b):
            return int(SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100)

    class process:
        @staticmethod
        def extractOne(query, choices, scorer=None):
            best = None
            best_score = -1
            for choice in choices:
                score = fuzz.token_set_ratio(query, choice)
                if score > best_score:
                    best_score = score
                    best = (choice, best_score, None)
            return best

# Optional imports
_spacy_available = False
_pytesseract_available = False
_pdf2image_available = False
try:
    import spacy
    _spacy_available = True
except Exception:
    pass

try:
    import pytesseract
    _pytesseract_available = True
except Exception:
    pass

try:
    from pdf2image import convert_from_bytes
    _pdf2image_available = True
except Exception:
    pass

# ---------- Safe rerun helper ----------
def safe_rerun():
    if hasattr(st, "experimental_rerun"):
        try:
            st.experimental_rerun()
            return
        except Exception:
            pass
    tried = False
    candidate_modules = [
        "streamlit.runtime.scriptrunner.script_runner",
        "streamlit.runtime.scriptrunner",
        "streamlit.web.server.server",
        "streamlit.web.server",
        "streamlit.report_thread"
    ]
    for mod_path in candidate_modules:
        try:
            mod = importlib.import_module(mod_path)
            RerunException = getattr(mod, "RerunException", None)
            if RerunException:
                tried = True
                raise RerunException()
        except Exception:
            pass
    if not tried:
        st.markdown("<meta http-equiv='refresh' content='0'>", unsafe_allow_html=True)

# ---------- Navigation helper ----------
def navigate(action: str):
    st.session_state._nav_token = st.session_state.get("_nav_token", 0) + 1
    mapping = {
        "home": 1,
        "back": max(1, st.session_state.get("step", 1) - 1),
        "upload": 1,
        "analyze": 2,
        "results": 3,
        "export": 4,
        "reset": 1
    }
    target = mapping.get(action, 1)
    st.session_state.step = target
    st.session_state._last_nav = {"action": action, "token": st.session_state._nav_token}
    safe_rerun()

nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

st.set_page_config(page_title="AI Resume Analyzer Pro", page_icon="🎯", layout="wide")

# ---------- Enhanced CSS ----------
def inject_css():
    st.markdown(
        """
    <style>
    .main-header { font-size:2.5rem; font-weight:800; text-align:center; margin-bottom:0.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .sub-header { color: #6b7280; text-align:center; margin-top:0; font-size:1.1rem; }
    .card { padding:1.5rem; border-radius:12px; background:linear-gradient(180deg,#ffffff 0%, #f9fafb 100%); box-shadow: 0 10px 30px rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.1); }
    .pill { display:inline-block; margin:4px; padding:8px 12px; border-radius:20px; background:#eef2ff; color:#3730a3; border:1px solid #c7d2fe; font-size:0.85rem; font-weight:500; }
    .pill-green { background:#d1fae5; color:#065f46; border-color:#6ee7b7; }
    .pill-red { background:#fee2e2; color:#7f1d1d; border-color:#fca5a5; }
    .pill-yellow { background:#fef3c7; color:#92400e; border-color:#fcd34d; }
    .small-muted { color:#6b7280; font-size:0.9rem; }
    .nav-btn { width: 100%; }
    .score-high { color: #10b981; font-weight:700; }
    .score-mid { color: #f59e0b; font-weight:700; }
    .score-low { color: #ef4444; font-weight:700; }
    .metric-card { padding:1rem; border-radius:8px; background:#f3f4f6; border-left:4px solid #667eea; }
    .insight-box { padding:1rem; border-radius:8px; background:#f0f9ff; border-left:4px solid #0ea5e9; }
    .recommendation { padding:0.75rem; margin:0.5rem 0; border-radius:6px; background:#f5f3ff; border-left:3px solid #a78bfa; }
    .strength { background:#f0fdf4; border-left-color:#22c55e; }
    .weakness { background:#fef2f2; border-left-color:#ef4444; }
    </style>
    """, unsafe_allow_html=True
    )

inject_css()

# ---------- Enhanced skill ontology with levels ----------
@st.cache_data
def load_skill_ontology():
    return {
        "Programming": ["Python", "Java", "JavaScript", "C++", "C#", "Go", "Rust", "TypeScript", "R", "PHP", "Swift", "Kotlin"],
        "Data Science": ["Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Keras", "MLflow", "XGBoost", "Statistics", "A/B Testing", "Tableau", "Power BI"],
        "Cloud & DevOps": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Jenkins", "CI/CD", "GitLab", "GitHub Actions"],
        "Databases": ["SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Cassandra", "DynamoDB", "Elasticsearch"],
        "Web": ["React", "Vue", "Angular", "Node.js", "Django", "Flask", "FastAPI", "Express", "REST", "GraphQL"],
        "Tools": ["Git", "Linux", "Bash", "JIRA", "Confluence", "VS Code", "Docker", "Jupyter", "Postman"],
        "Soft": ["Leadership", "Communication", "Problem Solving", "Teamwork", "Agile", "Scrum", "Project Management", "Mentoring"]
    }

@st.cache_data
def load_roles_framework():
    return {
        "Data Scientist": {
            "required": ["Python", "Machine Learning", "SQL", "Statistics"],
            "important": ["Deep Learning", "AWS", "Data Visualization"],
            "nice_to_have": ["Spark", "Docker", "Cloud"],
            "description": "Extracts insights from data using statistics and ML",
            "keywords": ["model", "algorithm", "predict", "accuracy", "experiment"]
        },
        "ML Engineer": {
            "required": ["Python", "TensorFlow", "MLOps", "Docker"],
            "important": ["Kubernetes", "AWS", "Pipeline"],
            "nice_to_have": ["Monitoring", "Scaling", "Deployment"],
            "description": "Builds and deploys ML systems at scale",
            "keywords": ["deploy", "production", "inference", "latency", "scalable"]
        },
        "Data Engineer": {
            "required": ["SQL", "Python", "ETL", "AWS"],
            "important": ["Spark", "Airflow", "Pipeline"],
            "nice_to_have": ["Kafka", "Docker", "Cloud"],
            "description": "Designs data pipelines and infrastructure",
            "keywords": ["pipeline", "etl", "ingestion", "workflow", "data warehouse"]
        },
        "Software Engineer": {
            "required": ["Python", "JavaScript", "Git", "Algorithms"],
            "important": ["Docker", "AWS", "React", "REST"],
            "nice_to_have": ["Microservices", "Cloud", "Testing"],
            "description": "Develops software systems and applications",
            "keywords": ["built", "developed", "implemented", "design", "architecture"]
        },
        "DevOps Engineer": {
            "required": ["Docker", "Kubernetes", "AWS", "CI/CD"],
            "important": ["Terraform", "Monitoring", "Scripting"],
            "nice_to_have": ["GitLab", "Jenkins", "Cloud"],
            "description": "Manages infrastructure and deployment pipelines",
            "keywords": ["deploy", "infrastructure", "monitoring", "automation", "pipeline"]
        }
    }

SKILL_ONTOLOGY = load_skill_ontology()
ROLES_FRAMEWORK = load_roles_framework()

def read_docx_bytes(bytes_buf: BytesIO) -> str:
    try:
        bytes_buf.seek(0)
        doc = docx.Document(bytes_buf)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception as e:
        return ""

def read_txt_bytes(bytes_buf: BytesIO) -> str:
    try:
        bytes_buf.seek(0)
        text = bytes_buf.read().decode("utf-8", errors="ignore")
        return text
    except Exception:
        return ""

def read_pdf_text(bytes_buf: BytesIO, try_ocr=False) -> str:
    try:
        bytes_buf.seek(0)
        text_pages = []
        with pdfplumber.open(bytes_buf) as pdf:
            for p in pdf.pages:
                page_text = p.extract_text()
                if page_text:
                    text_pages.append(page_text)
        return "\n".join(text_pages)
    except Exception:
        return ""

def extract_text_advanced(file_bytes: BytesIO, filename: str, ocr_enabled: bool=False) -> dict:
    result = {"text": "", "pages": 0, "format": "", "error": None, "ocr_used": False}
    try:
        low = filename.lower()
        file_bytes.seek(0)
        if low.endswith(".txt"):
            result["text"] = read_txt_bytes(file_bytes)
            result["pages"] = 1
            result["format"] = "Text"
        elif low.endswith(".docx"):
            file_bytes.seek(0)
            result["text"] = read_docx_bytes(file_bytes)
            result["format"] = "Word"
            result["pages"] = max(1, result["text"].count("\n") // 50 + 1)
        elif low.endswith(".pdf"):
            file_bytes.seek(0)
            text = read_pdf_text(file_bytes, try_ocr=False)
            result["text"] = text
            result["format"] = "PDF"
            result["pages"] = result["text"].count("\f") + 1 if result["text"] else 1
        if not result["text"].strip():
            result["error"] = "No text extracted from file."
    except Exception as e:
        result["error"] = f"Error extracting: {e}"
    return result

def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_sentences(text: str) -> list:
    """Extract sentences for context analysis"""
    try:
        return nltk.sent_tokenize(text)
    except:
        return text.split(".")

def fuzzy_skill_search(text: str, ontology: dict, threshold: int = 75):
    """Enhanced skill detection with context awareness"""
    text_lower = text.lower()
    sentences = extract_sentences(text)
    results = {}
    flat_skills = [(s, cat) for cat, skills in ontology.items() for s in skills]
    
    for skill, cat in flat_skills:
        skill_lower = skill.lower()
        score = 0
        count = 0
        context_score = 0
        
        # Direct match
        if skill_lower in text_lower:
            count = text_lower.count(skill_lower)
            score = 100
            
            # Check context (action verbs near skill)
            action_verbs = ["developed", "built", "created", "implemented", "designed", "optimized", "deployed", "managed"]
            for sent in sentences:
                if skill_lower in sent.lower():
                    if any(verb in sent.lower() for verb in action_verbs):
                        context_score += 15
        else:
            # Fuzzy match
            try:
                score = fuzz.token_set_ratio(skill_lower, text_lower)
                if score >= threshold:
                    count = 1
            except:
                score = 0
        
        if score >= threshold:
            # Enhanced proficiency: base score + frequency + context
            proficiency = min(100, int(score * 0.5 + min(count, 5) * 8 + context_score * 0.3 + 20))
            results.setdefault(cat, []).append({
                "name": skill,
                "score": int(score),
                "count": count,
                "proficiency": proficiency,
                "context_bonus": context_score
            })
    
    return results

def calculate_role_scores_advanced(text: str, roles: dict, ontology_matches: dict):
    """Multi-factor role scoring: keyword matching, requirement fulfillment, experience level"""
    text_lower = text.lower()
    role_scores = {}
    
    # Build TF-IDF corpus
    role_texts = []
    role_names = []
    for role, data in roles.items():
        role_names.append(role)
        combined = " ".join([
            role,
            " ".join(data.get("required", [])),
            " ".join(data.get("important", [])),
            " ".join(data.get("nice_to_have", [])),
            " ".join(data.get("keywords", []))
        ])
        role_texts.append(combined.lower())
    
    # TF-IDF scoring
    try:
        corpus = role_texts + [text_lower]
        vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        tfidf = vectorizer.fit_transform(corpus)
        cos_sim = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
        tfidf_scores = (cos_sim * 100).tolist()
    except:
        tfidf_scores = [0.0] * len(role_names)
    
    # Keyword matching
    for idx, role in enumerate(role_names):
        data = roles[role]
        
        # Required skills
        req_total = len(data.get("required", []))
        req_matched = sum(1 for r in data.get("required", []) if r.lower() in text_lower)
        req_pct = (req_matched / req_total) if req_total else 0.0
        
        # Important skills (weighted higher)
        imp_total = len(data.get("important", []))
        imp_matched = sum(1 for r in data.get("important", []) if r.lower() in text_lower)
        imp_pct = (imp_matched / imp_total) if imp_total else 0.0
        
        # Nice-to-have
        nice_total = len(data.get("nice_to_have", []))
        nice_matched = sum(1 for r in data.get("nice_to_have", []) if r.lower() in text_lower)
        nice_pct = (nice_matched / nice_total) if nice_total else 0.0
        
        # Keyword frequency (indicators of actual experience)
        keyword_score = 0
        keywords = data.get("keywords", [])
        for kw in keywords:
            keyword_score += text_lower.count(kw.lower()) * 3
        keyword_score = min(30, keyword_score)
        
        # Weighted final score
        rule_score = req_pct * 50 + imp_pct * 30 + nice_pct * 15 + keyword_score
        text_score = tfidf_scores[idx] if idx < len(tfidf_scores) else 0
        final_score = round(0.55 * text_score + 0.45 * rule_score, 1)
        
        role_scores[role] = {
            "score": final_score,
            "text_score": round(text_score, 1),
            "rule_score": round(rule_score, 1),
            "required_matched": req_matched,
            "required_total": req_total,
            "important_matched": imp_matched,
            "important_total": imp_total,
            "nice_matched": nice_matched,
            "nice_total": nice_total,
            "keyword_score": keyword_score,
            "description": data.get("description", "")
        }
    
    return role_scores

def estimate_overall_proficiency(ontology_matches):
    all_skills = [s for cat in ontology_matches.values() for s in cat]
    if not all_skills:
        return 0.0
    avg = sum(s.get("proficiency", 0) for s in all_skills) / len(all_skills)
    return round(avg, 1)

def generate_ats_score(text: str):
    """Better ATS scoring based on modern applicant tracking systems"""
    score = 100
    feedback = []
    t = text.lower()
    
    # Section detection
    sections = ["experience", "education", "skills", "projects", "summary", "contact"]
    found = [s for s in sections if s in t]
    sec_score = (len(found) / len(sections)) * 25
    score -= (25 - sec_score)
    if len(found) < 4:
        feedback.append("📌 Add standard sections: Experience, Education, Skills, Contact")
    
    # Word count
    words = len(text.split())
    if words < 200:
        score -= 15
        feedback.append("📝 Resume too short — add quantified achievements")
    elif words > 1500:
        score -= 8
        feedback.append("📝 Resume long — consider removing outdated experience")
    else:
        score += 5
    
    # Contact info
    if not re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text):
        score -= 10
        feedback.append("📧 Add professional email address")
    
    if not re.search(r"\b\d{10,}\b", text):
        feedback.append("📞 Consider adding contact phone number (optional)")
    
    # Action verbs
    verbs = ["managed", "led", "developed", "created", "implemented", "achieved", "designed", "optimized", "improved", "increased", "reduced", "delivered"]
    found_verbs = sum(1 for v in verbs if v in t)
    if found_verbs < 5:
        score -= 12
        feedback.append("💪 Use more action verbs (managed, developed, implemented, etc.)")
    
    # Numbers/metrics
    if re.search(r"\b\d+%\b|\b\$\d+", t):
        score += 8
    else:
        feedback.append("📊 Add metrics: percentages, dollar amounts, or quantified results")
    
    # Keywords density
    keywords = ["project", "team", "system", "data", "process", "client", "code", "design"]
    keyword_density = sum(t.count(kw) for kw in keywords) / max(words, 1)
    if keyword_density < 0.01:
        score -= 5
        feedback.append("🔑 Add more industry-specific keywords")
    
    # Formatting indicators
    if any(pattern in t for pattern in ["•", "▪", "-", "•"]):
        score += 3
    
    score = max(0, min(100, int(score)))
    return {
        "score": score,
        "feedback": feedback,
        "word_count": words,
        "sections": found,
        "verb_count": found_verbs,
        "has_metrics": bool(re.search(r"\b\d+%\b|\b\$\d+", t))
    }

def analyze_strengths_weaknesses(text: str, skills: dict, roles: dict, ats: dict) -> dict:
    """Identify resume strengths and weaknesses"""
    strengths = []
    weaknesses = []
    recommendations = []
    
    # Strengths
    if ats["verb_count"] >= 8:
        strengths.append("✅ Strong use of action verbs demonstrates impact")
    if ats["has_metrics"]:
        strengths.append("✅ Good use of quantifiable metrics and results")
    if len(skills) >= 4:
        strengths.append(f"✅ Diverse skill set across {len(skills)} categories")
    if ats["word_count"] > 300 and ats["word_count"] < 1200:
        strengths.append("✅ Well-balanced resume length")
    if len(ats["sections"]) >= 4:
        strengths.append("✅ All major sections present")
    
    # Weaknesses
    if ats["verb_count"] < 5:
        weaknesses.append("❌ Limited action verbs — strengthen with power words")
    if not ats["has_metrics"]:
        weaknesses.append("❌ Missing quantifiable results — add percentages, times, or impact")
    if len(skills) < 2:
        weaknesses.append("❌ Limited visible skills — explicitly list technologies used")
    if ats["word_count"] < 250:
        weaknesses.append("❌ Too brief — expand with achievements")
    if len(ats["sections"]) < 3:
        weaknesses.append("❌ Missing key sections")
    
    # Recommendations
    recommendations.append("Add specific project outcomes and business impact")
    if len(skills) < 3:
        recommendations.append("List skills by category (Programming, Cloud, Databases, etc.)")
    recommendations.append("Tailor to job description keywords")
    recommendations.append("Include certifications if applicable")
    
    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommendations": recommendations
    }

# ---------- Visualizations ----------
def role_bar_chart(role_scores: dict):
    names = list(role_scores.keys())
    scores = [role_scores[r]["score"] for r in names]
    colors = ['#10b981' if s >= 70 else '#f59e0b' if s >= 50 else '#ef4444' for s in scores]
    fig = go.Figure(go.Bar(
        x=scores, y=names, orientation='h',
        marker_color=colors,
        text=[f"{s}%" for s in scores],
        textposition='auto',
        hovertemplate="<b>%{y}</b><br>Compatibility: %{x}%<extra></extra>"
    ))
    fig.update_layout(
        height=400,
        xaxis=dict(range=[0,100], title="Compatibility Score (%)"),
        margin=dict(l=150),
        showlegend=False
    )
    return fig

def skills_distribution_chart(skills: dict):
    """Show skills distribution by category"""
    categories = list(skills.keys())
    counts = [len(skills[cat]) for cat in categories]
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', '#fa709a', '#fee140']
    fig = go.Figure(go.Pie(
        labels=categories,
        values=counts,
        marker=dict(colors=colors),
        hovertemplate="<b>%{label}</b><br>Skills: %{value}<extra></extra>"
    ))
    fig.update_layout(height=350, margin=dict(l=0, r=0))
    return fig

def proficiency_gauge(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis':{'range':[0,100]}, 'bar':{'color':'#667eea'}}
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20))
    return fig

def skills_scatter(skills: dict):
    """Scatter plot of skills by proficiency"""
    all_skills = []
    for cat, skill_list in skills.items():
        for s in skill_list:
            all_skills.append({
                'name': s['name'],
                'category': cat,
                'proficiency': s['proficiency'],
                'score': s['score']
            })
    
    if not all_skills:
        return None
    
    df = pd.DataFrame(all_skills)
    categories = df['category'].unique()
    colors_map = {cat: ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', '#fa709a', '#fee140'][i] 
                  for i, cat in enumerate(categories)}
    
    fig = go.Figure()
    for cat in categories:
        cat_data = df[df['category'] == cat]
        fig.add_trace(go.Scatter(
            x=cat_data['score'],
            y=cat_data['proficiency'],
            mode='markers+text',
            name=cat,
            text=cat_data['name'],
            textposition='top center',
            marker=dict(size=12, color=colors_map[cat]),
            hovertemplate="<b>%{text}</b><br>Match: %{x}%<br>Proficiency: %{y}%<extra></extra>"
        ))
    
    fig.update_layout(
        height=400,
        xaxis_title="Skill Match Score (%)",
        yaxis_title="Proficiency Level (%)",
        hovermode='closest',
        margin=dict(t=80)
    )
    return fig

# ---------- Session init ----------
if "step" not in st.session_state:
    st.session_state.step = 1
if "uploaded" not in st.session_state:
    st.session_state.uploaded = None
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "ocr_opt_in" not in st.session_state:
    st.session_state.ocr_opt_in = False
if "_nav_token" not in st.session_state:
    st.session_state._nav_token = 0
if "_last_nav" not in st.session_state:
    st.session_state._last_nav = None

# ---------- UI: Header / Steps ----------
def header_ui():
    st.markdown('<div class="main-header">🎯 AI Resume Analyzer Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced accuracy with enhanced insights, strength analysis & smart recommendations</div>', unsafe_allow_html=True)
    st.markdown("---")

header_ui()

# Step indicator
cols = st.columns([1,1,1,1])
steps = ["Upload", "Analyze", "Results", "Export"]
for i, s in enumerate(steps, start=1):
    label = f"✅ {s}" if st.session_state.step > i else (f"➡️ {s}" if st.session_state.step == i else s)
    cols[i-1].markdown(f"**{label}**")

st.markdown("")

# ---------- Step 1: Upload ----------
if st.session_state.step == 1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📄 Step 1 — Upload or Paste Resume")

    nav1, nav2, nav3 = st.columns([1,1,7])
    with nav1:
        st.button("🏠 Home", disabled=True)
    with nav2:
        st.button("⬅️ Back", disabled=True)

    c1, c2 = st.columns([2,1])
    with c1:
        uploaded_file = st.file_uploader("Upload resume (.pdf / .docx / .txt)", type=["pdf","docx","txt"])
        pasted = st.text_area("Or paste resume text", height=180, placeholder="Paste your resume content here...")
        st.checkbox("Enable OCR fallback for scanned PDFs", value=False, key="ocr_checkbox")

    with c2:
        st.subheader("💡 Quick Tips")
        st.write("✓ Include skills section")
        st.write("✓ Use machine-readable PDFs")
        st.write("✓ Add quantified results")
        st.write("✓ Use action verbs")

    act1, act2, act3 = st.columns([1,1,1])
    if act1.button("📋 Load Sample"):
        sample = """John Doe - Senior Data Scientist
Email: john@example.com | Phone: 555-1234
Experience:
- Developed ML models improving accuracy by 30% using Python, scikit-learn, TensorFlow
- Deployed models on AWS with Docker, reducing inference latency by 45%
- Led team of 3 data scientists on predictive analytics project
Skills: Python, Pandas, NumPy, Scikit-learn, TensorFlow, AWS, Docker, SQL, Spark
Education: MS Data Science
"""
        st.session_state.resume_text = sample
        st.session_state.uploaded = None
        navigate("analyze")

    if act2.button("🗑️ Clear"):
        st.session_state.uploaded = None
        st.session_state.resume_text = ""
        st.session_state.analysis = None
        navigate("home")

    if uploaded_file:
        st.session_state.uploaded = uploaded_file
        st.success(f"✅ File ready: {uploaded_file.name}")

    if pasted and not uploaded_file:
        st.session_state.resume_text = pasted

    if st.button("Next: Analyze ▶️", use_container_width=True):
        if st.session_state.uploaded:
            file = st.session_state.uploaded
            bs = BytesIO(file.read())
            ocr_opt = st.session_state.get("ocr_checkbox", False)
            extraction = extract_text_advanced(bs, file.name, ocr_enabled=ocr_opt)
            if extraction.get("error"):
                st.error(extraction["error"])
            else:
                st.session_state.resume_text = extraction["text"]
                st.session_state.ocr_opt_in = ocr_opt
                navigate("analyze")
        elif st.session_state.resume_text and st.session_state.resume_text.strip():
            navigate("analyze")
        else:
            st.warning("⚠️ Please upload or paste a resume first")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Step 2: Analyze ----------
elif st.session_state.step == 2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("⚙️ Step 2 — Advanced Analysis")

    nav1, nav2, nav3 = st.columns([1,1,7])
    with nav1:
        if st.button("🏠 Home"):
            navigate("home")
    with nav2:
        if st.button("⬅️ Back"):
            navigate("back")

    if not st.session_state.resume_text:
        st.error("❌ No resume text found")
    else:
        txt = normalize_text(st.session_state.resume_text)

        if st.session_state.analysis is not None:
            existing = st.session_state.analysis
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📊 Skills", sum(len(v) for v in existing["skills"].values()))
            col2.metric("🎯 Top Role", max(existing["roles"].items(), key=lambda x: x[1]["score"])[0])
            col3.metric("⭐ ATS Score", f"{existing['ats']['score']}/100")
            col4.metric("💪 Proficiency", f"{existing['proficiency']}%")
            
            rr1, rr2, rr3 = st.columns(3)
            if rr1.button("🔄 Re-run Analysis", use_container_width=True):
                st.session_state.analysis = None
                navigate("analyze")
            if rr2.button("📊 View Results", use_container_width=True):
                navigate("results")
            if rr3.button("📤 Export", use_container_width=True):
                navigate("export")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            prog = st.progress(0)
            status = st.empty()

            status.info("🔍 Extracting skills with context awareness...")
            skill_matches = fuzzy_skill_search(txt, SKILL_ONTOLOGY, threshold=75)
            prog.progress(35)

            status.info("📈 Calculating role compatibility...")
            role_scores = calculate_role_scores_advanced(txt, ROLES_FRAMEWORK, skill_matches)
            prog.progress(60)

            status.info("⭐ Generating ATS score...")
            ats = generate_ats_score(txt)
            prog.progress(80)

            status.info("📊 Analyzing strengths & weaknesses...")
            analysis_sw = analyze_strengths_weaknesses(txt, skill_matches, ROLES_FRAMEWORK, ats)
            prof = estimate_overall_proficiency(skill_matches)
            prog.progress(95)

            st.session_state.analysis = {
                "skills": skill_matches,
                "roles": role_scores,
                "ats": ats,
                "proficiency": prof,
                "strengths_weaknesses": analysis_sw,
                "text": txt,
                "timestamp": datetime.utcnow().isoformat()
            }
            prog.empty()
            status.success("✅ Analysis complete!")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("📊 Skills Found", sum(len(v) for v in skill_matches.values()))
            col2.metric("🎯 Top Role", max(role_scores.items(), key=lambda x: x[1]["score"])[0])
            col3.metric("⭐ ATS Score", f"{ats['score']}/100")
            col4.metric("💪 Avg Proficiency", f"{prof}%")

            if st.button("View Results ▶️", use_container_width=True):
                navigate("results")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Step 3: Results ----------
elif st.session_state.step == 3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📊 Step 3 — Results & Insights")

    nav1, nav2, nav3 = st.columns([1,1,7])
    with nav1:
        if st.button("🏠 Home"):
            navigate("home")
    with nav2:
        if st.button("⬅️ Back"):
            navigate("back")

    analysis = st.session_state.analysis
    if not analysis:
        st.error("❌ No analysis found")
    else:
        # Tab 1: Overview
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "🎯 Roles", "📚 Skills", "💪 Analysis", "🔍 Details"])
        
        with tab1:
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("### ATS Score")
                score_color = "score-high" if analysis["ats"]["score"] >= 70 else "score-mid" if analysis["ats"]["score"] >= 50 else "score-low"
                st.markdown(f'<p style="font-size:2rem;"><span class="{score_color}">{analysis["ats"]["score"]}/100</span></p>', unsafe_allow_html=True)
                st.caption(f"Word count: {analysis['ats']['word_count']}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("### Proficiency")
                st.markdown(f'<p style="font-size:2rem;"><span class="score-high">{analysis["proficiency"]}%</span></p>', unsafe_allow_html=True)
                st.caption(f"Skills: {sum(len(v) for v in analysis['skills'].values())}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("### Top Role")
                best_role = max(analysis["roles"].items(), key=lambda x: x[1]["score"])
                st.markdown(f'<p style="font-size:1.5rem;"><span class="score-high">{best_role[0]}</span></p>', unsafe_allow_html=True)
                st.caption(f"Match: {best_role[1]['score']}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Role compatibility chart
            st.subheader("🎯 Role Compatibility")
            st.plotly_chart(role_bar_chart(analysis["roles"]), use_container_width=True)
        
        with tab2:
            st.subheader("Detailed Role Matching")
            sorted_roles = sorted(analysis["roles"].items(), key=lambda x: x[1]["score"], reverse=True)
            
            for role_name, role_data in sorted_roles:
                with st.expander(f"**{role_name}** — {role_data['score']}%", expanded=(role_name == sorted_roles[0][0])):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Score", f"{role_data['score']}%")
                        st.metric("Text Similarity", f"{role_data['text_score']}%")
                    with col2:
                        st.metric("Required Skills", f"{role_data['required_matched']}/{role_data['required_total']}")
                        st.metric("Important Skills", f"{role_data['important_matched']}/{role_data['important_total']}")
                    with col3:
                        st.metric("Nice-to-have", f"{role_data['nice_matched']}/{role_data['nice_total']}")
                        st.metric("Keyword Match", f"{role_data['keyword_score']}/30")
                    
                    st.write(f"**Description:** {role_data['description']}")
                    
                    # Missing requirements
                    req_framework = ROLES_FRAMEWORK[role_name]
                    missing_req = [r for r in req_framework['required'] if r.lower() not in analysis["text"].lower()]
                    missing_imp = [r for r in req_framework['important'] if r.lower() not in analysis["text"].lower()]
                    
                    if missing_req:
                        st.markdown(f"**🔴 Critical gaps:** {', '.join(missing_req)}")
                    if missing_imp:
                        st.markdown(f"**🟡 Should add:** {', '.join(missing_imp)}")
        
        with tab3:
            st.subheader("Skills Inventory")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.plotly_chart(skills_distribution_chart(analysis["skills"]), use_container_width=True)
            
            with col2:
                for cat, skills in analysis["skills"].items():
                    st.markdown(f"**{cat}**")
                    for s in skills[:5]:  # Top 5 per category
                        level = "🟢" if s["proficiency"] >= 70 else "🟡" if s["proficiency"] >= 50 else "🔴"
                        st.markdown(f"<span class='pill'>{level} {s['name']} • {s['proficiency']}%</span>", unsafe_allow_html=True)
            
            scatter = skills_scatter(analysis["skills"])
            if scatter:
                st.plotly_chart(scatter, use_container_width=True)
        
        with tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 💚 Strengths")
                for strength in analysis["strengths_weaknesses"]["strengths"]:
                    st.markdown(f'<div class="recommendation strength">{strength}</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("### ❌ Weaknesses")
                for weakness in analysis["strengths_weaknesses"]["weaknesses"]:
                    st.markdown(f'<div class="recommendation weakness">{weakness}</div>', unsafe_allow_html=True)
            
            st.markdown("### 💡 Recommendations")
            for i, rec in enumerate(analysis["strengths_weaknesses"]["recommendations"], 1):
                st.markdown(f'<div class="recommendation"><strong>{i}.</strong> {rec}</div>', unsafe_allow_html=True)
            
            st.markdown("### 📋 ATS Feedback")
            if analysis["ats"]["feedback"]:
                for feedback in analysis["ats"]["feedback"]:
                    st.markdown(f'<div class="insight-box">{feedback}</div>', unsafe_allow_html=True)
        
        with tab5:
            st.subheader("📝 Resume Preview")
            st.text_area("Parsed Content", value=analysis["text"][:3000], height=300, disabled=True)
            
            st.download_button("📥 Download Parsed Text", data=analysis["text"], file_name="parsed_resume.txt", mime="text/plain")
            
            st.subheader("Raw JSON Data")
            with st.expander("Expand to view complete analysis JSON"):
                st.json(analysis)
        
        # Action buttons
        b1, b2, b3 = st.columns(3)
        if b1.button("⬅️ Back", use_container_width=True):
            navigate("back")
        if b2.button("📤 Export", use_container_width=True):
            navigate("export")
        if b3.button("📋 New Resume", use_container_width=True):
            st.session_state.uploaded = None
            st.session_state.resume_text = ""
            st.session_state.analysis = None
            navigate("home")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Step 4: Export ----------
elif st.session_state.step == 4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📤 Step 4 — Export & Download")

    nav1, nav2, nav3 = st.columns([1,1,7])
    with nav1:
        if st.button("🏠 Home"):
            navigate("home")
    with nav2:
        if st.button("⬅️ Back"):
            navigate("back")

    if not st.session_state.analysis:
        st.error("❌ No analysis to export")
    else:
        analysis = st.session_state.analysis

        tab1, tab2 = st.tabs(["📊 Downloads", "📋 Summary"])
        
        with tab1:
            st.subheader("Available Formats")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON export
                json_data = json.dumps({
                    "resume": st.session_state.resume_text,
                    "analysis": {
                        "skills": analysis["skills"],
                        "roles": analysis["roles"],
                        "ats": analysis["ats"],
                        "proficiency": analysis["proficiency"],
                        "timestamp": analysis["timestamp"]
                    }
                }, indent=2)
                st.download_button("📋 Download JSON", data=json_data, file_name="resume_analysis.json", mime="application/json", use_container_width=True)
            
            with col2:
                # CSV export
                skills_list = []
                for cat, skills in analysis["skills"].items():
                    for s in skills:
                        skills_list.append({
                            "Category": cat,
                            "Skill": s["name"],
                            "Proficiency": s["proficiency"],
                            "Match Score": s["score"],
                            "Frequency": s["count"]
                        })
                df_skills = pd.DataFrame(skills_list)
                st.download_button("📊 Download CSV (Skills)", data=df_skills.to_csv(index=False), file_name="resume_skills.csv", mime="text/csv", use_container_width=True)
            
            # Role analysis CSV
            roles_list = []
            for role, data in analysis["roles"].items():
                roles_list.append({
                    "Role": role,
                    "Overall Score": data["score"],
                    "Text Match": data["text_score"],
                    "Requirement Match": data["rule_score"],
                    "Required Match": f"{data['required_matched']}/{data['required_total']}",
                    "Important Match": f"{data['important_matched']}/{data['important_total']}"
                })
            df_roles = pd.DataFrame(roles_list)
            st.download_button("🎯 Download CSV (Roles)", data=df_roles.to_csv(index=False), file_name="resume_role_analysis.csv", mime="text/csv", use_container_width=True)
        
        with tab2:
            st.subheader("Analysis Summary")
            st.write(f"**Analysis Date:** {analysis['timestamp']}")
            st.write(f"**Resume Length:** {analysis['ats']['word_count']} words")
            st.write(f"**Skills Found:** {sum(len(v) for v in analysis['skills'].values())}")
            st.write(f"**Overall Proficiency:** {analysis['proficiency']}%")
            st.write(f"**ATS Score:** {analysis['ats']['score']}/100")
            
            best_role = max(analysis["roles"].items(), key=lambda x: x[1]["score"])
            st.write(f"**Best Matching Role:** {best_role[0]} ({best_role[1]['score']}%)")
        
        # New analysis button
        if st.button("🔄 Start New Analysis", use_container_width=True):
            st.session_state.step = 1
            st.session_state.uploaded = None
            st.session_state.resume_text = ""
            st.session_state.analysis = None
            navigate("home")
    
    st.markdown('</div>', unsafe_allow_html=True)
