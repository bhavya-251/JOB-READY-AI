import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI()


# ============================================================
# GOOGLE GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    max_tokens=2000,
    max_retries=2
)


# ============================================================
# JOB ROLE REQUIREMENTS TOOL
# ============================================================

@tool
def get_job_role_requirements(role: str) -> str:
    """
    Provides important skills and interview areas
    for a selected job role.
    """

    roles = {
        "software engineer": """
Skills:
- Programming
- Data Structures and Algorithms
- Object-Oriented Programming
- SQL and Databases
- Git and GitHub
- Problem Solving
- Basic System Design
- Communication
""",

        "web developer": """
Skills:
- HTML
- CSS
- JavaScript
- React
- REST APIs
- Databases
- Git and GitHub
- Debugging
""",

        "data scientist": """
Skills:
- Python
- Statistics
- Machine Learning
- Pandas
- NumPy
- SQL
- Data Visualization
- Problem Solving
""",

        "devops engineer": """
Skills:
- Linux
- Git
- CI/CD
- Docker
- Kubernetes
- Cloud
- Networking
- Monitoring
- Automation
""",

        "ai engineer": """
Skills:
- Python
- Machine Learning
- Deep Learning
- LLMs
- LangChain
- RAG
- APIs
- Vector Databases
- Prompt Engineering
"""
    }

    role_lower = role.lower().strip()

    for job_role, skills in roles.items():
        if job_role in role_lower:
            return skills

    return """
General interview areas:
- Technical knowledge
- Problem solving
- Projects
- Communication
- Behavioral questions
- Role-specific knowledge
"""


# ============================================================
# AGENT INSTRUCTIONS
# ============================================================

SYSTEM_PROMPT = """
You are JobReady AI, a professional job interview agent.

Your purpose is to conduct a realistic mock interview and
evaluate the candidate.

INTERVIEW PROCESS:

1. At the beginning, ask the candidate for their name.

2. Then ask which job role they want to practice for.

3. Once the candidate provides the job role, use the
   get_job_role_requirements tool.

4. Conduct exactly 7 interview questions after the
   candidate's name and role are known.

5. Ask ONLY ONE question at a time.

6. Wait for the candidate's answer before asking the next
   question.

7. Never ask multiple questions in one message.

8. The 7 questions should cover different areas:

Question 1:
Introduction or background.

Question 2:
Technical knowledge.

Question 3:
Problem solving.

Question 4:
Project experience.

Question 5:
Role-specific technical knowledge.

Question 6:
Behavioral or situational question.

Question 7:
A challenging technical or practical question.

9. After every answer:

- Briefly evaluate the answer.
- Mention what was good.
- Mention what could be improved.
- Then ask the next question.

10. Do not give the final score before all 7 questions
have been answered.

11. After the seventh answer, do not ask another question.

12. Give the final interview report.

FINAL REPORT FORMAT:

INTERVIEW COMPLETE

Overall Score: X/10

Technical Knowledge: X/10

Problem Solving: X/10

Communication: X/10

Confidence: X/10

Strengths:
- Give 3 specific strengths.

Weaknesses:
- Give 3 specific weaknesses.

How to Improve:
- Give 3 practical suggestions.

Final Feedback:
Give a short overall assessment of the candidate.

IMPORTANT RULES:

- Ask one question at a time.
- Do not skip questions.
- Do not give the final rating early.
- Base the evaluation only on the candidate's answers.
- Do not invent information about the candidate.
- Keep the interview professional.
- Do not use emojis.
- Do not use decorative symbols.
- Use plain professional text.
"""


# ============================================================
# LANGCHAIN AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=[get_job_role_requirements],
    system_prompt=SYSTEM_PROMPT,
    name="job_ready_agent"
)


# ============================================================
# WEBSITE
# ============================================================

HTML = """
<!DOCTYPE html>
<html>

<head>

    <title>JobReady AI</title>

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f5f7fb;
        }

        .container {
            max-width: 850px;
            margin: 30px auto;
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 5px 25px rgba(0,0,0,0.08);
        }

        h1 {
            text-align: center;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 25px;
        }

        #chat {
            height: 500px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 12px;
            padding: 15px;
            background: #fafafa;
        }

        .message {
            margin: 10px 0;
            padding: 12px 15px;
            border-radius: 12px;
            line-height: 1.5;
            white-space: pre-wrap;
        }

        .user {
            background: #e8f0fe;
            text-align: right;
        }

        .bot {
            background: #eeeeee;
        }

        .input-area {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }

        input {
            flex: 1;
            padding: 14px;
            border: 1px solid #ccc;
            border-radius: 10px;
            font-size: 16px;
        }

        button {
            padding: 14px 20px;
            border: none;
            border-radius: 10px;
            background: #111827;
            color: white;
            cursor: pointer;
            font-size: 15px;
        }

        button:hover {
            opacity: 0.9;
        }

        .start {
            display: block;
            margin: 0 auto 20px auto;
        }

    </style>

</head>


<body>

<div class="container">

    <h1>JobReady AI</h1>

    <div class="subtitle">
        LangChain Powered Interview Agent
    </div>

    <button class="start" onclick="startInterview()">
        Start Interview
    </button>

    <div id="chat"></div>

    <div class="input-area">

        <input
            id="message"
            type="text"
            placeholder="Type your answer..."
            onkeydown="handleEnter(event)"
        >

        <button onclick="sendMessage()">
            Send
        </button>

    </div>

</div>


<script>

let conversation = [];


function addMessage(text, sender) {

    const chat = document.getElementById("chat");

    const message = document.createElement("div");

    message.className = "message " + sender;

    message.textContent = text;

    chat.appendChild(message);

    chat.scrollTop = chat.scrollHeight;
}


async function startInterview() {

    conversation = [];

    document.getElementById("chat").innerHTML = "";

    const startMessage =
        "Start the interview. Ask me for my name.";

    try {

        const response = await fetch("/chat", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                message: startMessage,
                history: []
            })

        });

        const data = await response.json();

        addMessage(data.response, "bot");

        conversation.push({
            role: "user",
            content: startMessage
        });

        conversation.push({
            role: "assistant",
            content: data.response
        });

    } catch (error) {

        addMessage(
            "Unable to connect to the server. Please try again.",
            "bot"
        );

    }
}


async function sendMessage() {

    const input = document.getElementById("message");

    const message = input.value.trim();

    if (!message) {
        return;
    }

    addMessage(message, "user");

    input.value = "";

    conversation.push({
        role: "user",
        content: message
    });


    try {

        const response = await fetch("/chat", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                message: message,
                history: conversation
            })

        });


        const data = await response.json();

        addMessage(data.response, "bot");

        conversation.push({
            role: "assistant",
            content: data.response
        });

    } catch (error) {

        addMessage(
            "Unable to connect to the server. Please try again.",
            "bot"
        );

    }
}


function handleEnter(event) {

    if (event.key === "Enter") {
        sendMessage();
    }

}

</script>

</body>

</html>
"""


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home():

    return HTML


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/chat")
async def chat(request: Request):

    try:

        data = await request.json()

        message = data.get("message", "").strip()

        history = data.get("history", [])

        if not message:

            return JSONResponse({
                "response": "Please enter a message."
            })


        messages = history.copy()

        if (
            not messages
            or messages[-1].get("content") != message
        ):

            messages.append({
                "role": "user",
                "content": message
            })


        result = agent.invoke({
            "messages": messages
        })


        final_message = result["messages"][-1]

        response_text = final_message.content

        if not isinstance(response_text, str):

            if isinstance(response_text, list):

                response_text = " ".join(
                    str(item)
                    for item in response_text
                )

            else:

                response_text = str(response_text)


        return JSONResponse({
            "response": response_text
        })


    except Exception as e:

        print("ERROR:", str(e))

        return JSONResponse(
            {
                "response":
                "Something went wrong. Please try again."
            },
            status_code=500
        )
