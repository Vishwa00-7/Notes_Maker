import sys
from Graph import app
from Nodes import save_state
from State import State


def print_banner():
    banner = """
================================================================================
                    NOTES MAKER - AI CURRICULUM GENERATOR                       
        Multi-Agent Knowledge Graph & Structured Study Material Pipeline        
================================================================================
"""
    print(banner)


def main():
    print_banner()

    # Initial state template
    initial_state: State = {
        "topic": "",
        "methodology": "",
        "prequeist_knowledge": "",
        "depth": "",
        "level": "",
        "teaching_style": "",
        "selected_model": "chatgpt",
        "show_preview": True,
        "question": "",
        "roadmap": [],
        "length": 0,
        "progress": 0,
        "last_completed": "",
        "no_of_attempts_local": 0,
        "no_of_attempts_global": 0,
        "status": "running",
        "failed_at": "",
        "meta_prompt": "",
        "prompts": "",
        "filename": "",
        "answers": "",
        "summary": [],
    }

    try:
        final_state = app.invoke(initial_state)
        print("\n" + "=" * 80)
        print("[DONE] Workflow execution finished.")
        if final_state.get("topic"):
            print(f"Topic: {final_state.get('topic')}")
            print(f"Model used: {final_state.get('selected_model', 'chatgpt')}")
            print(f"Modules completed: {final_state.get('progress', 0)} / {final_state.get('length', 0)}")
            print(f"Last node: {final_state.get('last_completed')}")
        print("=" * 80)
    except KeyboardInterrupt:
        print("\n\n[WARN] Execution interrupted by user (Ctrl+C).")
        if initial_state.get("topic"):
            print(f"[SAVE] Preserving current state for '{initial_state.get('topic')}'...")
            save_state(initial_state)
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Uncaught exception during graph execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
