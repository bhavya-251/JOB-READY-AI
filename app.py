import os
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

app = FastAPI()


# =========================
# TOOLS
# =========================

@tool
def generate_interview_questions(role: str, skills: str, interview_type: str) -> str:
    """Generate interview questions based on the candidate's role, skills and interview type."""

    return f"""
    Generate suitable {interview_type} interview questions for a
    {role} candidate with the following skills:

    {skills}

    Include questions that test:
    - Core knowledge
    - Practical understanding
    - Problem solving
    - Real-world situations
    """


@tool
def evaluate_answer(answer: str) -> str:
    """Evaluate a candidate's interview answer."""

    return f"""
    Evaluate the following interview answer:

    {answer}

    Give:
    - Score out of 10
    - Strengths
    - Weaknesses
    - What is missing
    - How the answer can be improved
    """


# =========================
# GEMINI MODEL
# =========================

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)


# =========================
# LANGCHAIN AGENT
# =========================

system_prompt = """
You are an AI Interview Coach called InterviewPro.

You are a JOB READY AI AGENT designed to help students and
job seekers prepare for interviews.

Your responsibilities are:

1. Generate technical interview questions.
2. Generate HR interview questions.
3. Generate behavioral questions.
4. Conduct mock interviews.
5. Evaluate candidate answers.
6. Give scores out of 10.
7. Identify strengths and weaknesses.
8. Suggest improved answers.
9. Recommend topics that the candidate should prepare.
10. Provide a final interview performance report.

Rules:

- Ask ONE question at a time during a mock interview.
- Do not give the answer before the candidate responds.
- After the candidate answers, evaluate the answer.
- Give constructive and realistic feedback.
- For technical questions, check conceptual correctness.
- For behavioral questions, check clarity, relevance and structure.
- Use the STAR method when appropriate.

When the user asks to start an interview:
Ask the first appropriate question based on their job role,
experience and skills.

When evaluating an answer, provide:

Score: X/10

Strengths:
...

Weaknesses:
...

What is missing:
...

How to improve:
...

When asked for a final report, provide:

Overall Score
Technical Skills
Communication
Problem Solving
Strengths
Weaknesses
Recommended Preparation Topics
Final Advice
"""


agent = create_agent(
    model=model,
    tools=[
        generate_interview_questions,
        evaluate_answer
    ],
    system_prompt=system_prompt
)


# =========================
# RUN AGENT
# =========================

def run_agent(prompt):

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
    )

    return result["messages"][-1].content


# =========================
# HOME PAGE
# =========================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
    <!DOCTYPE html>

    <html>

    <head>

        <title>AI Interview Coach</title>

        <style>

            body {
                font-family: Arial, sans-serif;
                background: #f3f4f6;
                margin: 0;
                padding: 40px;
            }

            .container {
                max-width: 850px;
                margin: auto;
                background: white;
                padding: 35px;
                border-radius: 15px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            }

            h1 {
                text-align: center;
            }

            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
            }

            label {
                display: block;
                margin-top: 18px;
                font-weight: bold;
            }

            input, select, textarea {
                width: 100%;
                padding: 12px;
                margin-top: 7px;
                border: 1px solid #ccc;
                border-radius: 8px;
                box-sizing: border-box;
                font-size: 15px;
            }

            textarea {
                height: 120px;
            }

            button {
                width: 100%;
                margin-top: 25px;
                padding: 14px;
                border: none;
                border-radius: 8px;
                background: #111827;
                color: white;
                font-size: 16px;
                cursor: pointer;
            }

            button:hover {
                background: #374151;
            }

        </style>

    </head>

    <body>

        <div class="container">

            <h1> AI Interview Coach</h1>

            <p class="subtitle">
                Practice interviews and become job ready.
            </p>

            <form action="/interview" method="post">

                <label>Job Role</label>

                <input
                    type="text"
                    name="role"
                    placeholder="Example: Python Developer"
                    required
                >

                <label>Experience Level</label>

                <select name="experience">

                    <option>Fresher</option>
                    <option>0-1 Years</option>
                    <option>1-3 Years</option>
                    <option>3+ Years</option>

                </select>

                <label>Skills</label>

                <input
                    type="text"
                    name="skills"
                    placeholder="Example: Python, SQL, FastAPI"
                    required
                >

                <label>Interview Type</label>

                <select name="interview_type">

                    <option>Technical</option>
                    <option>HR</option>
                    <option>Technical + HR</option>
                    <option>Behavioral</option>

                </select>

                <label>What do you want to do?</label>

                <textarea
                    name="request"
                    placeholder="Example: Start my mock interview and ask me questions one by one."
                ></textarea>

                <button type="submit">
                    Start Mock Interview 
                </button>

            </form>

        </div>

    </body>

    </html>
    """


# =========================
# INTERVIEW
# =========================

@app.post("/interview", response_class=HTMLResponse)
def interview(
    role: str = Form(...),
    experience: str = Form(...),
    skills: str = Form(...),
    interview_type: str = Form(...),
    request: str = Form("")
):

    prompt = f"""
    Candidate Details:

    Job Role: {role}
    Experience: {experience}
    Skills: {skills}
    Interview Type: {interview_type}

    Candidate Request:
    {request}

    Act as the AI Interview Coach.

    If the candidate wants to start a mock interview,
    ask the first suitable question.

    If the candidate wants interview preparation,
    provide a preparation plan.

    If the candidate provides an interview answer,
    evaluate the answer and provide feedback.
    """

    response = run_agent(prompt)

    return f"""
    <!DOCTYPE html>

    <html>

    <head>

        <title>Interview Result</title>

        <style>

            body {{
                font-family: Arial, sans-serif;
                background: #f3f4f6;
                padding: 40px;
            }}

            .container {{
                max-width: 850px;
                margin: auto;
                background: white;
                padding: 35px;
                border-radius: 15px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            }}

            .response {{
                background: #f9fafb;
                padding: 25px;
                border-radius: 10px;
                white-space: pre-wrap;
                line-height: 1.7;
            }}

            a {{
                display: block;
                margin-top: 25px;
                text-align: center;
                text-decoration: none;
                background: #111827;
                color: white;
                padding: 13px;
                border-radius: 8px;
            }}

        </style>

    </head>

    <body>

        <div class="container">

            <h1> Interview Coach</h1>

            <div class="response">
{response}
            </div>

            <a href="/">
                ← Start Another Interview
            </a>

        </div>

    </body>

    </html>
    """


# =========================
# RENDER
# =========================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
