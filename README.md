# AI-Resume-Analyzer
An interactive AI-powered Resume Analyzer built with Streamlit that evaluates resumes for skills, job role fit, and ATS (Applicant Tracking System) score. It’s a beginner-friendly project designed to showcase practical AI + NLP implementation — and can be easily upgraded with GenAI, LLMs, or Resume Summarization features later.

## 🚀 Features
<pre>
✅ Smart Resume Parsing — Automatically reads uploaded PDF resumes and extracts relevant text.
✅ Fuzzy Skill Matching — Detects key skills using rapidfuzz for accurate comparisons.
✅ Role Fit Scoring — Matches resumes with predefined job roles using NLP + TF-IDF logic.
✅ ATS Scoring — Estimates how well your resume performs in automated HR systems.
✅ Named Entity Recognition (NER) — Highlights names, companies, and dates using spaCy.
✅ Modern Streamlit UI — Clean, responsive interface with step-by-step workflow.
✅ Session Navigation — “Back” and “Home” buttons across all steps for smooth UX.
✅ Upgradeable — Ready for future GenAI integration (e.g., GPT-based Resume Enhancer).
</pre>
## 🧠 How It Works
<pre>
Upload your resume (PDF) or paste raw text.

The app extracts and cleans your resume content.

AI models analyze:

Skills

Role compatibility

ATS score

Key entities (names, organizations, dates)

You get an instant, visual summary of your resume insights.
</pre>
## 🧰 Tech Stack
<pre>
Category	          Tools / Libraries
Framework	       |  Streamlit
NLP Engine  	   |  spaCy, rapidfuzz
Text Processing  |	re, string, nltk (optional)
Visualization	   |  Streamlit UI components
Language	       |  Python 3.10+

</pre>
## ⚙️ Installation Guide
<pre>
1️⃣ Clone the Repository
git clone https://github.com/your-username/ai-resume-analyzer.git
cd ai-resume-analyzer

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # for Windows
# or
source venv/bin/activate  # for macOS/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ (Optional but Recommended) Install spaCy and its model
pip install spacy
python -m spacy download en_core_web_sm

5️⃣ Run the App
streamlit run app.py
</pre>
## 📁 Project Structure
<pre>
resume_analyzer/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # All dependencies
├── README.md              # Documentation
├── /data                  # (Optional) Sample resumes
└── /venv                  # Virtual environment (not pushed to GitHub)

</pre>

## 🔮 Future Enhancements
<pre>

✨ Add ChatGPT / LLM-powered Resume Feedback
✨ Integrate Job Description Matcher
✨ Include Visual Resume Generator (PDF)
✨ Add Multi-language Resume Support
✨ Deploy on Streamlit Cloud or Hugging Face Spaces

</pre>
