"""
capstone_streamlit.py — Python Coding Assistant Agent
Run: streamlit run capstone_streamlit.py
"""
import streamlit as st
import uuid
import os
import chromadb
from dotenv import load_dotenv
from typing import TypedDict, List
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

st.set_page_config(page_title="Python Coding Assistant", page_icon="🐍", layout="centered")
st.title("🐍 Python Coding Assistant")
st.caption("Ask me anything about Python — lists, dicts, functions, OOP, decorators, file I/O, debugging, and more.")

# ── Load models and KB (cached) ───────────────────────────
@st.cache_resource
def load_agent():
    llm      = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client()
    try: client.delete_collection("capstone_kb")
    except: pass
    collection = client.create_collection("capstone_kb")

    DOCUMENTS = [
        {"id":"doc_001","topic":"Python Lists","text":"""A list in Python is an ordered, mutable collection that can hold items of any type. You create a list with square brackets: my_list = [1, 2, 3]. Common operations: Append: my_list.append(4). Insert: my_list.insert(0, 0). Remove: my_list.remove(2). Pop: my_list.pop(). Slicing: my_list[1:3]. List comprehension: squares = [x**2 for x in range(10)]. Lists are zero-indexed. Negative indices count from the end: my_list[-1] is the last element. len(my_list) returns the number of elements. Sorting: my_list.sort() sorts in-place; sorted(my_list) returns a new sorted list."""},
        {"id":"doc_002","topic":"Python Dictionaries","text":"""A dictionary (dict) is an unordered collection of key-value pairs. Keys must be hashable. Creation: person = {"name": "Alice", "age": 30}. Access: person["name"]. Safe access: person.get("city", "Unknown"). Adding: person["email"] = "alice@example.com". Deleting: del person["age"]. Iterating: for key, value in person.items(). Dict comprehension: squares = {x: x**2 for x in range(5)}. Merging (Python 3.9+): merged = dict1 | dict2. Python 3.7+ dicts maintain insertion order."""},
        {"id":"doc_003","topic":"Python Functions and Scope","text":"""Functions are defined with def. def greet(name, greeting="Hello"): return f"{greeting}, {name}!". *args collects extra positional args. **kwargs collects extra keyword args. Never use mutable defaults like []. Use None as default and set inside the body. Scope rules (LEGB): Local, Enclosing, Global, Built-in. global x modifies module-level variable. nonlocal x modifies enclosing variable. Lambda: square = lambda x: x**2."""},
        {"id":"doc_004","topic":"Python Classes and OOP","text":"""Classes define blueprints. class Animal: def __init__(self, name, sound): self.name = name; self.sound = sound. Inheritance: class Dog(Animal). super().__init__(name, "Woof"). Key dunders: __str__, __repr__, __len__, __eq__, __add__. Class variables shared across instances. @property makes read-only attribute. @classmethod receives cls. @staticmethod is a namespaced function."""},
        {"id":"doc_005","topic":"Exception Handling","text":"""try/except blocks handle errors. try: result = 10/divisor. except ZeroDivisionError: print("Cannot divide by zero"). else: runs only if no exception. finally: always runs. Raise: raise ValueError("bad"). Custom: class InsufficientFundsError(Exception): pass. Context managers: with open("file.txt") as f: data = f.read()."""},
        {"id":"doc_006","topic":"Python List Comprehensions and Generators","text":"""List comprehensions: evens = [x for x in range(20) if x % 2 == 0]. Dict comprehension: {x: x**2 for x in range(5)}. Set comprehension: {c for c in "hello"}. Generator expressions are lazy: gen = (x**2 for x in range(1000000)). Generator functions use yield. Use itertools for chain, islice, product, combinations."""},
        {"id":"doc_007","topic":"File I/O","text":"""Read: with open("data.txt","r",encoding="utf-8") as f: content = f.read(). Line by line: for line in f: process(line.rstrip()). Write: with open("output.txt","w") as f: f.write("Hello"). Append: mode "a". Modes: r, w, a, b, r+. Always use with. pathlib.Path: from pathlib import Path; p = Path("data")/"file.txt"; p.read_text(); p.write_text("hello")."""},
        {"id":"doc_008","topic":"Python Decorators","text":"""Decorators wrap functions. def timer(func): def wrapper(*args,**kwargs): start=time.time(); result=func(*args,**kwargs); print(time.time()-start); return result; return wrapper. @timer is shorthand for slow_function = timer(slow_function). Use @functools.wraps(func) to preserve __name__ and __doc__. Decorators with arguments need an extra layer."""},
        {"id":"doc_009","topic":"Virtual Environments and pip","text":"""python -m venv venv creates a virtual environment. source venv/bin/activate on macOS/Linux. venv\Scripts\activate on Windows. deactivate to leave. pip install requests. pip install requests==2.31.0 for specific version. pip freeze > requirements.txt to save. pip install -r requirements.txt to restore. Modern tools: pipenv, poetry, uv. Add venv/ to .gitignore."""},
        {"id":"doc_010","topic":"Debugging and Testing","text":"""Debugging: print(), pdb.set_trace(), breakpoint() (3.7+). pdb commands: n, s, c, q, p var. pytest: def test_add(): assert add(2,3)==5. Run: pytest test_file.py -v. Fixtures: @pytest.fixture. Coverage: pytest --cov=mymodule. unittest is the built-in alternative."""},
        {"id":"doc_011","topic":"Python Type Hints","text":"""def greet(name: str, age: int) -> str. Optional[str] means str or None. Union[int,str] means either. Python 3.10+: int | str. Type hints not enforced at runtime. Use mypy for static checking. TypeVar for generics. @dataclass uses type hints to auto-generate __init__, __repr__, __eq__."""},
        {"id":"doc_012","topic":"Common Python Pitfalls","text":"""Mutable default argument bug: def f(to=[]). Fix: use to=None then create inside. Late binding in closures: use lambda i=i: i. Do not use is for value comparison — use ==. Circular imports cause ImportError. Forgetting to copy: use .copy() or list(). Use copy.deepcopy() for nested objects."""},
    ]

    texts = [d["text"] for d in DOCUMENTS]
    collection.add(
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        ids=[d["id"] for d in DOCUMENTS],
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS]
    )

    # ── State ─────────────────────────────────────────────
    class CapstoneState(TypedDict):
        question:       str
        messages:       List[dict]
        route:          str
        retrieved:      str
        sources:        List[str]
        tool_result:    str
        answer:         str
        faithfulness:   float
        eval_retries:   int
        search_results: str

    # ── Nodes ─────────────────────────────────────────────
    def memory_node(state):
        msgs = state.get("messages", [])
        msgs = msgs + [{"role": "user", "content": state["question"]}]
        if len(msgs) > 6: msgs = msgs[-6:]
        return {"messages": msgs}

    def router_node(state):
        question = state["question"]
        messages = state.get("messages", [])
        recent = "; ".join(f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]) or "none"
        prompt = f"""You are a router for a Python programming assistant.
Options:
- retrieve: Python concepts, syntax, OOP, decorators, testing, etc.
- memory_only: answer from conversation history
- tool: latest library version, recent release notes, PyPI info

Recent: {recent}
Question: {question}
Reply with ONLY one word: retrieve / memory_only / tool"""
        decision = llm.invoke(prompt).content.strip().lower()
        if "memory" in decision: decision = "memory_only"
        elif "tool" in decision: decision = "tool"
        else: decision = "retrieve"
        return {"route": decision}

    def retrieval_node(state):
        q_emb = embedder.encode([state["question"]]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=3)
        chunks = results["documents"][0]
        topics = [m["topic"] for m in results["metadatas"][0]]
        context = "\n\n---\n\n".join(f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks)))
        return {"retrieved": context, "sources": topics}

    def skip_retrieval_node(state):
        return {"retrieved": "", "sources": []}

    def tool_node(state):
        question = state["question"]
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(question + " Python", max_results=4))
            tool_result = "\n\n".join(
                f"Source: {r['href']}\nTitle: {r['title']}\nSnippet: {r['body'][:300]}"
                for r in results
            ) if results else "No web results found."
        except Exception as e:
            tool_result = f"Web search unavailable: {e}"
        return {"tool_result": tool_result}

    def answer_node(state):
        question = state["question"]
        retrieved = state.get("retrieved", "")
        tool_result = state.get("tool_result", "")
        messages = state.get("messages", [])
        eval_retries = state.get("eval_retries", 0)
        context_parts = []
        if retrieved: context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
        if tool_result: context_parts.append(f"TOOL RESULT:\n{tool_result}")
        context = "\n\n".join(context_parts)
        if context:
            system_content = f"""You are a friendly Python programming assistant.
Answer using ONLY the information provided in the context below.
Use concrete code examples whenever possible.
If the answer is not in the context, say: "I don't have that in my knowledge base — try asking about lists, dicts, functions, classes, exceptions, comprehensions, file I/O, decorators, environments, debugging, type hints, or common pitfalls."
Do NOT fabricate Python behaviour not stated in the context.

{context}"""
        else:
            system_content = "You are a friendly Python programming assistant. Answer based on the conversation history."
        if eval_retries > 0:
            system_content += "\n\nIMPORTANT: Answer using ONLY information explicitly stated in the context."
        lc_msgs = [SystemMessage(content=system_content)]
        for msg in messages[:-1]:
            lc_msgs.append(HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]))
        lc_msgs.append(HumanMessage(content=question))
        return {"answer": llm.invoke(lc_msgs).content}

    FAITHFULNESS_THRESHOLD = 0.7
    MAX_EVAL_RETRIES = 2

    def eval_node(state):
        answer = state.get("answer", "")
        context = state.get("retrieved", "")[:500]
        retries = state.get("eval_retries", 0)
        if not context:
            return {"faithfulness": 1.0, "eval_retries": retries + 1}
        prompt = f"Rate faithfulness 0.0-1.0. Reply with only a number.\nContext: {context}\nAnswer: {answer[:300]}"
        try:
            score = float(llm.invoke(prompt).content.strip().split()[0].replace(",", "."))
            score = max(0.0, min(1.0, score))
        except:
            score = 0.5
        return {"faithfulness": score, "eval_retries": retries + 1}

    def save_node(state):
        messages = state.get("messages", [])
        return {"messages": messages + [{"role": "assistant", "content": state["answer"]}]}

    # ── Graph ─────────────────────────────────────────────
    def route_decision(state):
        route = state.get("route", "retrieve")
        if route == "tool": return "tool"
        if route == "memory_only": return "skip"
        return "retrieve"

    def eval_decision(state):
        score = state.get("faithfulness", 1.0)
        retries = state.get("eval_retries", 0)
        if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES: return "save"
        return "answer"

    g = StateGraph(CapstoneState)
    g.add_node("memory",   memory_node)
    g.add_node("router",   router_node)
    g.add_node("retrieve", retrieval_node)
    g.add_node("skip",     skip_retrieval_node)
    g.add_node("tool",     tool_node)
    g.add_node("answer",   answer_node)
    g.add_node("eval",     eval_node)
    g.add_node("save",     save_node)
    g.set_entry_point("memory")
    g.add_edge("memory", "router")
    g.add_conditional_edges("router", route_decision, {"retrieve":"retrieve","skip":"skip","tool":"tool"})
    g.add_edge("retrieve", "answer")
    g.add_edge("skip",     "answer")
    g.add_edge("tool",     "answer")
    g.add_edge("answer",   "eval")
    g.add_conditional_edges("eval", eval_decision, {"answer":"answer","save":"save"})
    g.add_edge("save", END)
    agent_app = g.compile(checkpointer=MemorySaver())

    return agent_app, embedder, collection

# ── App setup ─────────────────────────────────────────────
try:
    agent_app, embedder, collection = load_agent()
    st.success(f"✅ Knowledge base loaded — {collection.count()} documents")
except Exception as e:
    st.error(f"Failed to load agent: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.write("Ask me anything about Python — lists, dicts, functions, OOP, decorators, file I/O, debugging, and more.")
    st.write(f"Session: {st.session_state.thread_id}")
    st.divider()
    st.write("**Topics covered:**")
    topics = ["Python Lists","Python Dictionaries","Python Functions and Scope",
              "Python Classes and OOP","Exception Handling",
              "Python List Comprehensions and Generators","File I/O",
              "Python Decorators","Virtual Environments and pip",
              "Debugging and Testing","Python Type Hints","Common Python Pitfalls"]
    for t in topics:
        st.write(f"• {t}")
    if st.button("🗑️ New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

# ── Chat history ──────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Chat input ────────────────────────────────────────────
if prompt := st.chat_input("Ask a Python question..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = agent_app.invoke({"question": prompt}, config=config)
            answer = result.get("answer", "Sorry, I could not generate an answer.")
        st.write(answer)
        faith = result.get("faithfulness", 0.0)
        if faith > 0:
            st.caption(f"Faithfulness: {faith:.2f} | Sources: {result.get('sources', [])}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
