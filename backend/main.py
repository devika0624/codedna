from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from hindsight_client import Hindsight
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime
import os, json, asyncio, pathlib

# ── ENV ───────────────────────────────────────────────────────────────────────
env_path = pathlib.Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI(title="CodeDNA API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CLIENTS ───────────────────────────────────────────────────────────────────
hindsight = Hindsight(
    base_url=os.getenv("HINDSIGHT_BASE_URL"),
    api_key=os.getenv("HINDSIGHT_API_KEY"),
)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── SESSION STATE ─────────────────────────────────────────────────────────────
active_sessions: dict[str, dict] = {}

# ── BANK NAMES ────────────────────────────────────────────────────────────────
def bank_mistakes(u): return f"{u}_mistakes"
def bank_behavior(u): return f"{u}_behavior"
def bank_energy(u):   return f"{u}_energy"

def ts(): return datetime.utcnow().strftime("%Y-%m-%dT%H:%M")

# ── HINDSIGHT HELPERS ─────────────────────────────────────────────────────────
async def safe_retain(bank: str, content: str):
    try:
        await asyncio.to_thread(hindsight.retain, bank_id=bank, content=content)
    except Exception as e:
        print(f"[retain error] {e}")

async def safe_recall(bank: str, query: str, chars: int = 600) -> str:
    try:
        result = await asyncio.to_thread(hindsight.recall, bank_id=bank, query=query)
        return str(result)[:chars] if result else ""
    except Exception as e:
        print(f"[recall error] {e}")
        return ""

async def safe_reflect(bank: str, query: str, chars: int = 600) -> str:
    try:
        result = await asyncio.to_thread(hindsight.reflect, bank_id=bank, query=query)
        return str(result)[:chars] if result else ""
    except Exception as e:
        print(f"[reflect error] {e}")
        return ""

# ── GROQ HELPER ───────────────────────────────────────────────────────────────
def _groq_sync(prompt: str, system: str, temp: float) -> str:
    res = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        temperature=temp,
        max_tokens=1024,
    )
    return res.choices[0].message.content.strip()

async def ask_groq(
    prompt: str,
    system: str = "You are an intelligent coding mentor.",
    json_mode: bool = False
) -> str:
    try:
        temp = 0.1 if json_mode else 0.7
        return await asyncio.to_thread(_groq_sync, prompt, system, temp)
    except Exception as e:
        return f"Error: {str(e)}"

# ── SAFE JSON ─────────────────────────────────────────────────────────────────
def safe_json(raw: str, fallback):
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        start = min(
            clean.find("{") if clean.find("{") != -1 else len(clean),
            clean.find("[") if clean.find("[") != -1 else len(clean),
        )
        return json.loads(clean[start:])
    except Exception:
        return fallback

# ── REQUEST MODELS ────────────────────────────────────────────────────────────
class Onboard(BaseModel):
    user_id: str
    language: str
    level: str
    goal: str

class Mood(BaseModel):
    user_id: str
    mood: str

class Submit(BaseModel):
    user_id: str
    code: str
    time_taken: int
    problem_id: str = "unknown"
    problem_topic: str = "unknown"
    problem_title: str = "unknown"

# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":        "live",
        "version":       "4.0",
        "architecture":  "3-bank Hindsight memory",
        "banks":         ["mistakes", "behavior", "energy"],
        "model":         "llama3-70b-8192",
        "memory_system": "Hindsight Cloud",
        "features": [
            "mistake fingerprinting",
            "deja vu detection",
            "pre-failure prediction",
            "fatigue detection",
            "cross-bank reflect",
            "mistake DNA profile",
            "raw memory visibility",
        ]
    }

# ─────────────────────────────────────────────────────────────────────────────
# POST /onboard
# FIX 1: Now returns user_id so frontend can store it
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/onboard")
async def onboard(req: Onboard):
    try:
        await safe_retain(
            bank_behavior(req.user_id),
            f"[{ts()}] User profile: language={req.language}, level={req.level}, goal={req.goal}"
        )

        welcome = await ask_groq(
            f"A {req.level} {req.language} programmer wants to improve for {req.goal}. "
            f"Write a warm specific 2-sentence welcome. Address them directly.",
            system="You are an encouraging coding mentor. Be brief and specific.",
        )

        # FIX 1 — return user_id so frontend stores it correctly
        return {
            "status":  "ok",
            "message": welcome,
            "user_id": req.user_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# POST /mood
# FIX 2: Now returns label and sub fields frontend expects
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/mood")
async def mood(req: Mood):
    try:
        energy_history = await safe_recall(
            bank_energy(req.user_id),
            f"performance when user was {req.mood}",
            chars=400
        )

        raw = await ask_groq(
            f"""
            Student mood today: {req.mood}
            Past mood-performance from Hindsight: {energy_history or "No past data yet."}

            Decide difficulty and write a short encouraging message (1-2 sentences).
            JSON only:
            {{
                "difficulty": "easy|medium|hard",
                "message": "1-2 sentence encouraging message"
            }}
            """,
            system="You are an adaptive coding mentor. JSON only.",
            json_mode=True
        )

        data       = safe_json(raw, {"difficulty": "medium", "message": "Let's have a great session!"})
        difficulty = data.get("difficulty", "medium")
        message    = data.get("message", "Let's have a great session!")

        # Difficulty label map
        label_map = {"easy": "Easy Mode", "medium": "Medium Mode", "hard": "Hard Mode"}
        label = label_map.get(difficulty, "Medium Mode")

        await safe_retain(
            bank_energy(req.user_id),
            f"[{ts()}] Mood: {req.mood}. Difficulty set: {difficulty}."
        )

        # FIX 2 — return label and sub that frontend reads
        return {
            "difficulty": difficulty,
            "message":    message,
            "label":      label,
            "sub":        message,
            "mood":       req.mood
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# GET /session/start
# FIX 3: Problem now returns body + examples[] array frontend expects
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/session/start")
async def session_start(user_id: str, difficulty: str = "medium"):
    try:
        mistakes_ctx, behavior_ctx, energy_ctx = await asyncio.gather(
            safe_recall(bank_mistakes(user_id), "most frequent and recent mistakes, weakest topics", 600),
            safe_recall(bank_behavior(user_id), "language preference, level, goal, learning style",  400),
            safe_recall(bank_energy(user_id),   "recent mood and performance patterns",               300),
        )

        is_first = not mistakes_ctx and not behavior_ctx

        greeting_prompt = (
            "Welcome to CodeDNA! This is your first session — I'll get smarter about "
            "your weaknesses as we go. Let's start building your learning profile!"
            if is_first else
            f"""
            You remember everything about this student.
            Past mistakes: {mistakes_ctx}
            Behavior: {behavior_ctx}
            Energy: {energy_ctx}
            Write a sharp 2-sentence session opening mentioning their specific weak area.
            Address them as "you".
            """
        )

        # FIX 3 — problem JSON now uses body + examples[] to match frontend
        problem_prompt = f"""
            Generate ONE original coding problem for a {difficulty} difficulty session.

            Student memory profile:
            - Weak topics from past mistakes: {mistakes_ctx or "No history — generate a well-rounded problem."}
            - Preferences: {behavior_ctx or "Python, intermediate."}
            - Energy: {energy_ctx or "Standard session."}

            Rules:
            1. Target their WEAKEST topic from mistake history
            2. Match {difficulty} difficulty exactly
            3. Solvable in 10-20 minutes
            4. NOT Two Sum, NOT FizzBuzz, NOT Fibonacci
            5. Include a tricky edge case matching their known blind spots

            Respond in this EXACT JSON format:
            {{
                "id": "p_topic_uniqueid",
                "title": "Problem title",
                "topic": "arrays|recursion|binary_search|stacks|strings|dynamic_programming|graphs|sorting|trees",
                "difficulty": "{difficulty}",
                "body": "Clear problem statement. What is the input? What should be returned? Write 2-3 sentences.",
                "examples": [
                    {{"input": "example input here", "output": "example output here", "note": "brief explanation"}},
                    {{"input": "another input", "output": "another output", "note": "edge case explanation"}}
                ],
                "constraints": ["constraint 1 e.g. 1 <= n <= 10^4", "constraint 2"],
                "hidden_trap": "the tricky edge case this user is likely to miss based on their history"
            }}
            JSON only. No extra text.
        """

        reason_prompt = f"""
            In ONE sentence explain why this problem was chosen for this student.
            Their weak areas: {mistakes_ctx[:200] or "general practice"}
            Be specific — mention the actual weak topic.
        """

        greeting, raw_problem, reason = await asyncio.gather(
            ask_groq(greeting_prompt, system="You are a sharp coding mentor. Be brief."),
            ask_groq(problem_prompt,  system="You are a coding problem generator. JSON only.", json_mode=True),
            ask_groq(reason_prompt,   system="You are a coding mentor. One sentence only."),
        )

        # FIX 3 — fallback also uses body + examples[]
        problem = safe_json(raw_problem, {
            "id":          "p_arrays_fallback",
            "title":       "Find Maximum Subarray Sum",
            "topic":       "arrays",
            "difficulty":  difficulty,
            "body":        "Given an integer array, find the contiguous subarray with the largest sum and return that sum. A subarray is a contiguous part of the array.",
            "examples": [
                {"input": "nums = [-2,1,-3,4,-1,2,1,-5,4]", "output": "6", "note": "subarray [4,-1,2,1] has the largest sum"},
                {"input": "nums = [1]",                      "output": "1", "note": "single element array"},
            ],
            "constraints": ["1 <= nums.length <= 10^5", "-10^4 <= nums[i] <= 10^4"],
            "hidden_trap": "All-negative arrays — the answer is the least negative number, not 0"
        })

        active_sessions[user_id] = {
            "problem":    problem,
            "difficulty": difficulty,
            "start_time": ts(),
        }

        await safe_retain(
            bank_behavior(user_id),
            f"[{ts()}] Session started. Difficulty: {difficulty}. "
            f"Problem: {problem.get('title')} (topic: {problem.get('topic')})"
        )

        return {
            "greeting":         greeting,
            "problem":          problem,
            "reason":           reason,
            "difficulty":       difficulty,
            "is_first_session": is_first,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# GET /predict — Pre-failure prediction
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/predict")
async def predict_mistake(user_id: str):
    try:
        session = active_sessions.get(user_id)
        if not session:
            return {
                "predicted_mistake": "none",
                "confidence":        "low",
                "warning":           "Start a session first.",
                "watch_out_for":     ""
            }

        problem  = session.get("problem", {})
        patterns = await safe_recall(
            bank_mistakes(user_id),
            "most recurring mistake types and topics",
            chars=500
        )

        if not patterns:
            return {
                "predicted_mistake": "none",
                "confidence":        "low",
                "warning":           "No history yet — I'll learn your patterns as we go!",
                "watch_out_for":     "Read the problem carefully before coding."
            }

        raw = await ask_groq(
            f"""
            Student's recurring mistake patterns from Hindsight memory:
            {patterns}

            Current problem topic: {problem.get('topic', 'unknown')}
            Problem hidden trap:   {problem.get('hidden_trap', 'edge cases')}

            Based ONLY on their Hindsight memory, predict what mistake they will likely make.
            JSON only:
            {{
                "predicted_mistake": "specific mistake type",
                "confidence":        "high|medium|low",
                "warning":           "one specific warning sentence to show before they code",
                "watch_out_for":     "exactly what to double-check in their solution"
            }}
            """,
            system="You are a predictive AI mentor. Base predictions only on memory data. JSON only.",
            json_mode=True
        )

        prediction = safe_json(raw, {
            "predicted_mistake": "edge_case",
            "confidence":        "medium",
            "warning":           "Watch out for edge cases — your most common blind spot.",
            "watch_out_for":     "Empty inputs and boundary conditions."
        })

        await safe_retain(
            bank_behavior(user_id),
            f"[{ts()}] Pre-failure prediction shown: "
            f"{prediction.get('predicted_mistake')} on {problem.get('topic')}"
        )

        return prediction

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# POST /submit
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/submit")
async def submit(req: Submit):
    try:
        session = active_sessions.get(req.user_id)
        if not session:
            raise HTTPException(
                status_code=400,
                detail="No active session. Call /session/start first."
            )

        problem = session.get("problem", {})
        title   = req.problem_title or problem.get("title",  "unknown")
        topic   = req.problem_topic or problem.get("topic",  "unknown")

        past_mistakes = await safe_recall(
            bank_mistakes(req.user_id),
            f"mistakes on {topic} problems",
            chars=500
        )

        # Evaluate the ACTUAL submitted code
        raw_eval = await ask_groq(
            f"""
            Problem: "{title}" (topic: {topic})
            Student's past mistakes on this topic: {past_mistakes or "No history yet."}

            Student's submitted code:
            ```
            {req.code}
            ```

            Evaluate carefully. JSON only:
            {{
                "is_correct":         true or false,
                "mistake_type":       "none|off_by_one|wrong_data_structure|missing_edge_case|time_complexity|logic_error|syntax_error",
                "explanation":        "specific explanation of what is wrong (empty string if correct)",
                "hint":               "helpful nudge without giving away solution (empty if correct)",
                "correct_approach":   "brief description of the optimal approach",
                "time_complexity":    "submitted code complexity e.g. O(n²)",
                "optimal_complexity": "optimal complexity e.g. O(n)"
            }}
            JSON only. No extra text.
            """,
            system="You are a precise code evaluator. Always respond with valid JSON only.",
            json_mode=True
        )

        evaluation = safe_json(raw_eval, {
            "is_correct":         False,
            "mistake_type":       "logic_error",
            "explanation":        "Could not parse submission automatically.",
            "hint":               "Review your logic carefully.",
            "correct_approach":   "Think through edge cases.",
            "time_complexity":    "Unknown",
            "optimal_complexity": "Unknown"
        })

        is_correct   = evaluation.get("is_correct", False)
        mistake_type = evaluation.get("mistake_type", "none")

        # ── DEJA VU DETECTOR ──────────────────────────────────────────────────
        deja_vu         = False
        deja_vu_message = ""
        deja_vu_count   = 0

        if not is_correct and mistake_type != "none":
            if past_mistakes and mistake_type in past_mistakes:
                deja_raw = await ask_groq(
                    f"""
                    Past mistake history from Hindsight: {past_mistakes}
                    Current mistake: {mistake_type} on topic: {topic}

                    Has this exact mistake type appeared before?
                    JSON only:
                    {{
                        "is_recurring":   true or false,
                        "times_seen":     number,
                        "pattern_summary":"one sentence describing the recurring pattern"
                    }}
                    """,
                    system="You are a pattern detector. JSON only.",
                    json_mode=True
                )
                deja_data = safe_json(
                    deja_raw,
                    {"is_recurring": False, "times_seen": 0, "pattern_summary": ""}
                )

                if deja_data.get("is_recurring"):
                    deja_vu       = True
                    deja_vu_count = deja_data.get("times_seen", 2)
                    pattern       = deja_data.get("pattern_summary", "")
                    deja_vu_message = (
                        f"Recurring blind spot! You've made this "
                        f"'{mistake_type.replace('_', ' ')}' error "
                        f"{deja_vu_count} time(s) before. {pattern}"
                    )

        # ── FATIGUE DETECTION ─────────────────────────────────────────────────
        fatigue_signals = []
        if req.time_taken > 120:   fatigue_signals.append("slow solving time")
        if deja_vu:                fatigue_signals.append("repeating same mistake")
        if deja_vu_count >= 3:     fatigue_signals.append("persistent blind spot 3+ sessions")
        fatigue        = bool(fatigue_signals)
        fatigue_reason = ", ".join(fatigue_signals)

        # ── CONFIDENCE SCORE ──────────────────────────────────────────────────
        if is_correct:
            confidence = round(0.6 + 0.4 * max(0, 1 - req.time_taken / 300), 2)
        else:
            confidence = max(0, round(0.5 - req.time_taken / 600, 2))

        # ── AI ENCOURAGEMENT ──────────────────────────────────────────────────
        ai_msg = await ask_groq(
            f"Result: {'correct!' if is_correct else f'wrong — {mistake_type}'}. "
            f"Fatigue: {fatigue_reason or 'none'}. Deja vu: {deja_vu}. "
            f"Give ONE motivating line max 15 words.",
            system="You are an encouraging mentor. One sentence, max 15 words."
        )

        # ── RETAIN TO HINDSIGHT ───────────────────────────────────────────────
        if is_correct:
            await asyncio.gather(
                safe_retain(
                    bank_mistakes(req.user_id),
                    f"[{ts()}] Correct: '{title}' ({topic}). "
                    f"Time:{req.time_taken}s. Complexity:{evaluation.get('time_complexity','?')}."
                ),
                safe_retain(
                    bank_behavior(req.user_id),
                    f"[{ts()}] Solved '{title}' correctly in {req.time_taken}s."
                )
            )
        else:
            await asyncio.gather(
                safe_retain(
                    bank_mistakes(req.user_id),
                    f"[{ts()}] Mistake:{mistake_type} on '{title}' ({topic}). "
                    f"{evaluation.get('explanation', '')}"
                ),
                safe_retain(
                    bank_behavior(req.user_id),
                    f"[{ts()}] Wrong on {topic}. Mistake:{mistake_type}. "
                    f"Time:{req.time_taken}s. Fatigue:{fatigue_reason or 'none'}."
                )
            )

        return {
            "is_correct":         is_correct,
            "mistake_type":       mistake_type,
            "explanation":        evaluation.get("explanation", ""),
            "hint":               evaluation.get("hint", ""),
            "correct_approach":   evaluation.get("correct_approach", ""),
            "time_complexity":    evaluation.get("time_complexity", ""),
            "optimal_complexity": evaluation.get("optimal_complexity", ""),
            "deja_vu":            deja_vu,
            "deja_vu_message":    deja_vu_message,
            "deja_vu_count":      deja_vu_count,
            "fatigue":            fatigue,
            "fatigue_reason":     fatigue_reason,
            "confidence":         confidence,
            "ai_message":         ai_msg,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# POST /hint
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/hint")
async def hint(user_id: str):
    try:
        session = active_sessions.get(user_id)
        if not session:
            return {"hint": "Start a session first!"}

        problem       = session.get("problem", {})
        past_mistakes = await safe_recall(
            bank_mistakes(user_id),
            f"mistakes on {problem.get('topic', 'unknown')} problems",
            chars=400
        )

        hint_text = await ask_groq(
            f"""
            Problem: {problem.get('title')} (topic: {problem.get('topic')})
            Description: {problem.get('body', '')}
            Past mistakes on this topic: {past_mistakes or "No history yet."}

            Give a specific targeted hint.
            Do NOT give the solution.
            Reference their known weak spots if relevant.
            2-3 sentences max.
            """,
            system="You are a Socratic coding mentor. Guide, don't give away answers."
        )

        return {"hint": hint_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# GET /session/end
# FIX 4: Now returns score and streak
# FIX 5: Insights now have both body and text fields
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/session/end")
async def session_end(user_id: str):
    try:
        mistakes_data, behavior_data, energy_data = await asyncio.gather(
            safe_recall(bank_mistakes(user_id), "all mistakes today and recurring patterns", 600),
            safe_recall(bank_behavior(user_id), "behavior today — pace, give-up signals",   400),
            safe_recall(bank_energy(user_id),   "mood today and effect on performance",      300),
        )

        # Cross-bank reflect — the 3-bank differentiator
        root_cause = await safe_reflect(
            bank_mistakes(user_id),
            f"""
            Behavior context: {behavior_data}
            Energy context:   {energy_data}
            What is the ROOT CAUSE of this user's mistakes?
            Not WHAT they got wrong — WHY do they keep getting it wrong?
            What is the single most important thing to work on next session?
            3 sentences max. Be specific.
            """,
            chars=500
        )

        # Generate summary + insights + score in parallel
        summary_prompt = f"""
            Session data:
            Mistakes:   {mistakes_data}
            Behavior:   {behavior_data}
            Energy:     {energy_data}
            Root cause: {root_cause}

            Write a personalised 3-4 sentence session summary.
            Cover: what went well, what needs work, one specific next action.
            Be honest and encouraging. Address the user as "you" directly.
        """

        insights_prompt = f"""
            Memory data:
            Mistakes:   {mistakes_data[:300]}
            Behavior:   {behavior_data[:200]}
            Root cause: {root_cause[:200]}

            Generate exactly 3 specific insight cards grounded in the memory data.
            NOT generic advice — real insights from what Hindsight observed.
            JSON array only:
            [
                {{"title": "short title", "body": "one specific actionable sentence", "type": "mistake|behavior|energy"}},
                {{"title": "short title", "body": "one specific actionable sentence", "type": "mistake|behavior|energy"}},
                {{"title": "short title", "body": "one specific actionable sentence", "type": "mistake|behavior|energy"}}
            ]
        """

        # FIX 4 — score calculation from Hindsight memory
        score_prompt = f"""
            Based on this session: {mistakes_data[:300]}
            Estimate a performance score from 0-100.
            Consider: problems attempted, mistakes made, improvement shown.
            Reply with a single integer only. Nothing else.
        """

        summary, raw_insights, score_raw = await asyncio.gather(
            ask_groq(summary_prompt, system="You are an honest encouraging coding mentor."),
            ask_groq(insights_prompt, system="You are a learning analyst. JSON array only.", json_mode=True),
            ask_groq(score_prompt, system="Reply with a single integer 0-100 only. Nothing else.")
        )

        # Parse score safely
        try:
            score = max(0, min(100, int(''.join(filter(str.isdigit, score_raw[:5])))))
        except Exception:
            score = 65

        # FIX 5 — insights have BOTH body and text fields
        raw_list = safe_json(raw_insights, [])
        if not isinstance(raw_list, list) or len(raw_list) == 0:
            raw_list = [
                {"title": "Keep practicing",   "body": "Consistency beats intensity every time.",      "type": "behavior"},
                {"title": "Target weak spots", "body": "Focus on your recurring mistake types.",       "type": "mistake"},
                {"title": "Great effort",      "body": "CodeDNA gets smarter every session you do.",  "type": "energy"},
            ]

        # Add text as alias for body so frontend works regardless of which field it reads
        insights = []
        for ins in raw_list:
            body = ins.get("body", ins.get("text", "Keep practicing."))
            insights.append({
                "title": ins.get("title", "Insight"),
                "body":  body,
                "text":  body,   # alias — frontend reads ins.text
                "type":  ins.get("type", "behavior"),
            })

        await safe_retain(
            bank_energy(user_id),
            f"[{ts()}] Session completed. Score:{score}. Root cause: {root_cause[:200]}"
        )

        active_sessions.pop(user_id, None)

        return {
            "summary":    summary,
            "insights":   insights,
            "root_cause": root_cause,
            "score":      score,       # FIX 4
            "streak":     3,           # FIX 4 — static for now, good enough for demo
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# GET /dna — Mistake DNA Profile
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/dna")
async def get_dna(user_id: str):
    try:
        mistakes_data, behavior_data, energy_data = await asyncio.gather(
            safe_recall(bank_mistakes(user_id), "all mistake patterns frequencies recurring errors", 700),
            safe_recall(bank_behavior(user_id), "learning style preferences pace give-up patterns", 500),
            safe_recall(bank_energy(user_id),   "mood vs performance correlation all sessions",      400),
        )

        if not mistakes_data and not behavior_data:
            return {
                "ready":   False,
                "message": "Complete at least 2 sessions to unlock your Mistake DNA profile.",
                "dna":     None
            }

        deep_insight = await safe_reflect(
            bank_mistakes(user_id),
            f"""
            Behavior: {behavior_data}
            Energy:   {energy_data}
            What are the deepest patterns in this user's learning?
            What does their mistake history reveal about how they think?
            """,
            chars=600
        )

        raw_dna = await ask_groq(
            f"""
            Complete Hindsight memory:
            Mistakes:     {mistakes_data}
            Behavior:     {behavior_data}
            Energy:       {energy_data}
            Deep insight: {deep_insight}

            Generate a complete Mistake DNA profile. JSON only:
            {{
                "primary_weakness":    "their #1 recurring mistake with specific detail",
                "secondary_weakness":  "their #2 issue",
                "learning_style":      "how they learn best from behavior data",
                "best_condition":      "when they perform best (mood, time, topic)",
                "worst_condition":     "when they perform worst",
                "predicted_next_fail": "what they will likely get wrong next session",
                "recommended_focus":   "single most important thing to practice",
                "strength":            "one genuine strength observed in their history",
                "dna_summary":         "one punchy sentence capturing this coder's unique pattern",
                "sessions_analysed":   "estimated number of sessions in memory"
            }}
            JSON only. No extra text.
            """,
            system="You are a learning scientist generating precise DNA profiles. JSON only.",
            json_mode=True
        )

        dna = safe_json(raw_dna, {
            "primary_weakness":  "Still building your profile...",
            "dna_summary":       "Complete more sessions to unlock your full DNA profile.",
            "sessions_analysed": "1"
        })

        return {
            "ready":      True,
            "dna":        dna,
            "powered_by": "3-bank Hindsight cross-reflect"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# GET /memory — Raw Hindsight memory visibility
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/memory")
async def show_memory(user_id: str):
    try:
        mistakes_mem, behavior_mem, energy_mem = await asyncio.gather(
            safe_recall(bank_mistakes(user_id), "everything stored", 800),
            safe_recall(bank_behavior(user_id), "everything stored", 600),
            safe_recall(bank_energy(user_id),   "everything stored", 400),
        )

        return {
            "user_id":         user_id,
            "has_memory":      bool(mistakes_mem or behavior_mem or energy_mem),
            "banks_active":    3,
            "mistakes_memory": mistakes_mem  or "No mistake data yet",
            "behavior_memory": behavior_mem  or "No behavior data yet",
            "energy_memory":   energy_mem    or "No energy data yet",
            "message":         "This is what Hindsight remembers about you across all sessions"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
