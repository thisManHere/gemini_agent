import sys
import uuid
 
from graph import build_agent
from rag import get_vectorstore

def main():
    get_vectorstore(rebuild="--rebuild" in sys.argv)  # builds the index on first run
    agent = build_agent()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}, "recursion_limit": 25}
    print("Agent ready. Type 'exit' to quit.\n")
 
    while True:
        user = input("you > ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        if not user:
            continue

        for update in agent.stream({"messages": [("user", user)]}, config, stream_mode="updates"):
            for node, out in update.items():
                if node == "planner":
                    print(f"\n[plan]\n{out['plan']}\n")
                elif node == "agent":
                    msg = out["messages"][-1]
                    for call in msg.tool_calls:
                        print(f"[tool call] {call['name']}({call['args']})")
                    if not msg.tool_calls:
                        print(f"\nagent > {msg.content}\n")
                        
                elif node == "tools":
                    for m in out["messages"]:
                        preview = str(m.content)[:200].replace("\n", " ")
                        print(f"[tool result] {preview}")
 
 
if __name__ == "__main__":
    main()