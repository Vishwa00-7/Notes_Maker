import json
import re
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import questionary
from Models import chatgpt, meta, qwen
from Prompts import master_prompt, meta_prompt_genrator
from State import State

# -----------------------------------------------------------------------------
# Configuration Constants & Options
# -----------------------------------------------------------------------------

depths = [
    "Like ChatGPT",
    "Primer / Crash Course",
    "Standard Foundation",
    "Comprehensive Review",
    "Advanced / Specialized",
    "Academic / Theoretical"
]

methodologies = [
    "Like ChatGPT",
    "Standard Textbook Chapter",
    "Cheat Sheet / Quick Reference",
    "Case-Study Driven",
    "Visual / Structural Notes",
    "Flashcard / Q&A Style"
]

verbosity_levels = [
    "Like ChatGPT",
    "Brief / Essential: Highly concise, bare minimum elaboration, strict focus on brevity (Target: ~300-500 words).",
    "Standard / Moderate: Balanced explanation with brief examples, getting straight to the point (Target: ~500-1000 words).",
    "Detailed / Thorough: Well-explained concepts with multiple examples, nuances, and careful elaboration (Target: ~1000-2000 words).",
    "Comprehensive / Very Detailed: Deep exploration, extensively elaborated paragraphs, and heavy contextualization (Target: ~2000-3000 words).",
    "Exhaustive / Maximum Verbosity: Pushes the context limit. Leaves absolutely no stone unturned, maximizing detail and word count (Target: 3000+ words)."
]

teaching_styles = [
    "Like ChatGPT",
    "Direct & Authoritative: Straightforward, factual, and strictly professional. No fluff, just the information.",
    "Socratic Method: Guides the reader by asking thought-provoking questions, encouraging them to deduce the answers before explicitly revealing them.",
    "Enthusiastic & Conversational: Warm, encouraging, and highly approachable. Speaks directly to the reader like a supportive, high-energy mentor.",
    "Narrative / Storytelling: Weaves the concepts into a cohesive story, historical journey, or continuous metaphor to make abstract ideas tangible.",
    "Humorous & Witty: Lighthearted and engaging. Uses analogies, pop-culture references, and gentle wit to keep the reader entertained while learning."
]

dictionary_for_chatGPT = {
    "teaching_methodology": "Concept -> Terminology -> Fundamentals -> Mathematical Foundation -> Step-by-Step Working -> Examples -> Implementation",
    "explanation_level": "Expert Instructor -> Beginner-Friendly -> Intermediate -> Advanced",
    "depth_of_explanation": "Very High / Comprehensive",
    "teaching_style": "Structured, systematic, one topic at a time, no information dumping, theory first and code at the end"
}

# -----------------------------------------------------------------------------
# File & Directory Helper Functions with Exception Handling
# -----------------------------------------------------------------------------

def sanitize_name(name: str) -> str:
    """Sanitize directory and file names to be safe across operating systems."""
    return re.sub(r'[\\/*?:"<>|]', "_", str(name)).strip()


def create_folder(name: str) -> bool:
    """Safely create a folder with retry and error handling."""
    safe_name = sanitize_name(name)
    for attempt in range(1, 4):
        try:
            Path(safe_name).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"[WARN] [create_folder] Attempt {attempt}/3 failed for '{safe_name}': {e}")
            if attempt < 3:
                time.sleep(1)
            else:
                print(f"[ERROR] [create_folder] Failed to create directory '{safe_name}' after 3 attempts.")
                return False
    return False


def save_state(state: State) -> bool:
    """Safely save state to <topic>/state.json with retry and error handling."""
    topic = sanitize_name(state.get("topic", "default_topic"))
    create_folder(topic)
    filename = Path(topic) / "state.json"

    for attempt in range(1, 4):
        try:
            # Convert non-serializable objects using default=str
            state_dict = dict(state)
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(state_dict, f, indent=4, default=str)
            print(f"[SAVE] State successfully saved to '{filename}'.")
            return True
        except Exception as e:
            print(f"[WARN] [save_state] Attempt {attempt}/3 failed: {e}")
            if attempt < 3:
                time.sleep(1)
            else:
                print(f"[ERROR] [save_state] Could not save state to '{filename}' after 3 attempts.")
                return False
    return False


def load_state(topic: str) -> Optional[State]:
    """Safely load state from <topic>/state.json with error handling."""
    safe_topic = sanitize_name(topic)
    filename = Path(safe_topic) / "state.json"

    for attempt in range(1, 4):
        try:
            if not filename.exists():
                print(f"[WARN] State file not found at '{filename}'.")
                return None
            with open(filename, "r", encoding="utf-8") as f:
                loaded_state = json.load(f)
            print(f"[LOAD] State successfully loaded from '{filename}'.")
            return loaded_state
        except json.JSONDecodeError as e:
            print(f"[ERROR] [load_state] Corrupted JSON in '{filename}': {e}")
            return None
        except Exception as e:
            print(f"[WARN] [load_state] Attempt {attempt}/3 failed: {e}")
            if attempt < 3:
                time.sleep(1)
            else:
                print(f"[ERROR] [load_state] Failed to load state after 3 attempts: {e}")
                return None
    return None


def extract_json_payload(raw_content: str) -> dict:
    """Robustly extract and parse JSON from LLM output, handling markdown blocks."""
    content = raw_content.strip()
    # Strip markdown code fences if present
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0].strip()

    # If pure json.loads works
    try:
        return json.loads(content)
    except Exception:
        pass

    # Regex search for outermost JSON object { ... }
    match = re.search(r"(\{.*\})", content, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    raise ValueError(f"Could not parse valid JSON from response: {raw_content[:200]}...")


# -----------------------------------------------------------------------------
# Human Intervention for Errors (Triggered after 3 consecutive failures)
# -----------------------------------------------------------------------------

def error_human_intervention(
    state: State,
    failed_node: str = "unknown_operation",
    error_message: str = ""
) -> Dict[str, Any]:
    """
    Function triggered when 3 consecutive errors occur in any node or operation.
    Prompts the user with clear options:
    1. Save state and stop
    2. Retry the failed operation
    3. Continue on (proceed / skip)
    """
    print("\n" + "=" * 65)
    print(f"[CRITICAL] [HUMAN INTERVENTION REQUIRED]")
    print(f"The operation '{failed_node}' failed 3 consecutive times.")
    if error_message:
        print(f"Error details: {error_message}")
    print("=" * 65)

    choices = [
        "1. Save state and stop (Safely persist progress and exit)",
        "2. Retry the failed operation (Try 3 more times)",
        "3. Stop without saving (Discard recent changes and exit immediately)"
    ]

    try:
        selection = questionary.select(
            "Three consecutive errors occurred. How would you like to proceed?",
            choices=choices
        ).ask()
    except Exception as e:
        print(f"[WARN] Prompt input failed ({e}). Defaulting to 'Save state and stop'.")
        selection = choices[0]

    if not selection or "1. Save" in selection:
        print(f"\n[SAVE] [Intervention] Saving state for topic '{state.get('topic', 'unnamed')}'...")
        save_state(state)
        print("Process halted safely. You can resume later.")
        return {
            "action": "stop",
            "status": "stopped",
            "save_on_stop": True,
            "failed_at": failed_node,
            "no_of_attempts_local": state.get("no_of_attempts_local", 3),
            "no_of_attempts_global": state.get("no_of_attempts_global", 3),
            "last_completed": "error_human_intervention"
        }
    elif "2. Retry" in selection:
        print(f"\n[RETRY] [Intervention] Resetting local attempt counter and retrying '{failed_node}'...")
        return {
            "action": "retry",
            "no_of_attempts_local": 0,
            "failed_at": ""
        }
    else:
        print(f"\n[STOPPED] [Intervention] Stopping process without saving state...")
        return {
            "action": "stop",
            "status": "stopped",
            "save_on_stop": False,
            "failed_at": failed_node,
            "no_of_attempts_local": state.get("no_of_attempts_local", 3),
            "no_of_attempts_global": state.get("no_of_attempts_global", 3),
            "last_completed": "error_human_intervention_discarded"
        }


# -----------------------------------------------------------------------------
# Retry Wrapper with Human Intervention Loop
# -----------------------------------------------------------------------------

def execute_with_retry(
    operation: Callable[[], Dict[str, Any]],
    state: State,
    node_name: str,
    max_retries: int = 3,
    fallback_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Executes a callable operation up to `max_retries` times.
    Tracks state attempts (no_of_attempts_local, no_of_attempts_global, failed_at).
    If it fails 3 times, leads to `error_human_intervention`.
    If the user chooses 'Retry', resets counter and restarts the retry loop.
    """
    while True:
        try_count = 0
        last_error: Optional[Exception] = None
        while try_count < max_retries:
            try_count += 1
            try:
                print(f"[RUNNING] [{node_name}] Attempt {try_count}/{max_retries}...")
                result = operation()
                # On success, clear local failure tracker
                state["no_of_attempts_local"] = 0
                state["failed_at"] = ""
                if isinstance(result, dict):
                    result["no_of_attempts_local"] = 0
                    result["no_of_attempts_global"] = state.get("no_of_attempts_global", 0)
                    result["failed_at"] = ""
                return result

            except Exception as e:
                last_error = e
                state["no_of_attempts_local"] = state.get("no_of_attempts_local", 0) + 1
                state["no_of_attempts_global"] = state.get("no_of_attempts_global", 0) + 1
                state["failed_at"] = node_name
                print(f"[WARN] [{node_name}] Error on attempt {try_count}/{max_retries}: {e}")
                if try_count < max_retries:
                    time.sleep(1.5 * try_count)

        # Reached 3 failures -> lead to human intervention
        state["failed_at"] = node_name
        intervention = error_human_intervention(
            state=state,
            failed_node=node_name,
            error_message=str(last_error)
        )

        action = intervention.get("action")
        if action == "retry":
            state["no_of_attempts_local"] = 0
            continue  # Restart attempt loop
        else:
            # Stopped (either saved or stopped without saving)
            return intervention


# -----------------------------------------------------------------------------
# Graph Nodes with Exceptional Handling
# -----------------------------------------------------------------------------

def starting_node(state: State) -> str:
    """Determine starting node: fresh input or continue from existing state."""
    try:
        question = questionary.select(
            "Choose the Starting Node : ",
            choices=["getInput", "continue_from_last_state"],
            default="getInput"
        ).ask()
        if question == "continue_from_last_state":
            return "continue_from_last_state"
        return "getInput"
    except (KeyboardInterrupt, Exception) as e:
        print(f"[WARN] [starting_node] Selection interrupted or failed ({e}). Defaulting to 'getInput'.")
        return "getInput"


def continue_from_last_state(state: State) -> Dict[str, Any]:
    """Load existing state from disk with input retry and error handling."""
    max_input_tries = 3
    for attempt in range(1, max_input_tries + 1):
        try:
            topic = input("Enter the Topic to resume: ").strip()
            if not topic:
                print("[WARN] Topic cannot be empty.")
                continue

            loaded_state = load_state(topic)
            if loaded_state:
                return {
                    "topic": loaded_state.get("topic", topic),
                    "prequeist_knowledge": loaded_state.get("prequeist_knowledge", ""),
                    "methodology": loaded_state.get("methodology", "Standard Textbook Chapter"),
                    "depth": loaded_state.get("depth", "Standard Foundation"),
                    "teaching_style": loaded_state.get("teaching_style", "Direct & Authoritative"),
                    "level": loaded_state.get("level", "Standard / Moderate"),
                    "progress": loaded_state.get("progress", 0),
                    "last_completed": loaded_state.get("last_completed", "continue_from_last_state"),
                    "roadmap": loaded_state.get("roadmap", []),
                    "length": loaded_state.get("length", len(loaded_state.get("roadmap", []))),
                    "prompts": loaded_state.get("prompts", ""),
                    "filename": loaded_state.get("filename", ""),
                    "answers": loaded_state.get("answers", ""),
                    "no_of_attempts_local": 0,
                    "no_of_attempts_global": loaded_state.get("no_of_attempts_global", 0),
                    "failed_at": ""
                }
            else:
                print(f"[WARN] Could not load state for '{topic}'. (Attempt {attempt}/{max_input_tries})")
        except (KeyboardInterrupt, Exception) as e:
            print(f"[WARN] [continue_from_last_state] Error: {e}")

    print("[ERROR] Failed to find state. Falling back to human intervention.")
    return error_human_intervention(
        state=state,
        failed_node="continue_from_last_state",
        error_message="Could not load specified topic state after 3 attempts."
    )


def getInput(state: State) -> Dict[str, Any]:
    """Interactively collect curriculum settings with error handling."""
    try:
        topic = input("Enter the Topic : ").strip()
        if not topic:
            topic = "Untitled Topic"

        pk = input("Enter the Prerequisite Knowledge you have : ").strip()
        if not pk:
            pk = "None (Beginner)"

        methodology = questionary.select("Choose the Methodology : ", choices=methodologies).ask()
        if not methodology or methodology == "Like ChatGPT":
            methodology = dictionary_for_chatGPT["teaching_methodology"]

        depth = questionary.select("Choose the Depth : ", choices=depths).ask()
        if not depth or depth == "Like ChatGPT":
            depth = dictionary_for_chatGPT["depth_of_explanation"]

        level = questionary.select("Choose the Verbosity Level : ", choices=verbosity_levels).ask()
        if not level or level == "Like ChatGPT":
            level = dictionary_for_chatGPT["explanation_level"]

        teaching_style = questionary.select("Choose the Teaching Style : ", choices=teaching_styles).ask()
        if not teaching_style or teaching_style == "Like ChatGPT":
            teaching_style = dictionary_for_chatGPT["teaching_style"]

        return {
            "topic": topic,
            "prequeist_knowledge": pk,
            "methodology": methodology,
            "depth": depth,
            "teaching_style": teaching_style,
            "level": level,
            "progress": 0,
            "last_completed": "getInput",
            "no_of_attempts_local": 0,
            "no_of_attempts_global": 0,
            "failed_at": ""
        }

    except (KeyboardInterrupt, Exception) as e:
        print(f"[WARN] [getInput] Input interrupted or failed: {e}")
        return {
            "topic": state.get("topic", "Default Study"),
            "prequeist_knowledge": state.get("prequeist_knowledge", "None"),
            "methodology": state.get("methodology", dictionary_for_chatGPT["teaching_methodology"]),
            "depth": state.get("depth", dictionary_for_chatGPT["depth_of_explanation"]),
            "teaching_style": state.get("teaching_style", dictionary_for_chatGPT["teaching_style"]),
            "level": state.get("level", dictionary_for_chatGPT["explanation_level"]),
            "progress": 0,
            "last_completed": "getInput_interrupted",
            "no_of_attempts_local": 0,
            "no_of_attempts_global": 1,
            "failed_at": "getInput"
        }


def generate_roadmap(state: State) -> Dict[str, Any]:
    """Generate structured curriculum roadmap via ChatGPT with 3-attempt retry."""
    def _op():
        prompt = master_prompt.invoke({
            "topic": state.get("topic", ""),
            "prerequisites": state.get("prequeist_knowledge", ""),
            "methodology": state.get("methodology", ""),
            "depth": state.get("depth", ""),
            "teaching_style": state.get("teaching_style", ""),
            "verbosity_level": state.get("level", ""),
            "last_completed": "generate_roadmap"
        })

        result = chatgpt.invoke(prompt)
        raw_content = result.content if hasattr(result, "content") else str(result)
        if not raw_content or not raw_content.strip():
            raise ValueError("Curriculum planner returned empty response. Check token limits.")

        parsed = extract_json_payload(raw_content)

        

        curriculum = parsed.get("curriculum", [])
        if not curriculum:
            raise ValueError("Curriculum is empty in LLM response.")

        print(f"[SUCCESS] Roadmap successfully generated with {len(curriculum)} modules.")

        # Create a folder with the name of the topic
        create_folder(state.get("topic", "default_topic"))

        return {
            "roadmap": curriculum,
            "length": len(curriculum),
            "last_completed": "generate_roadmap"
        }

    return execute_with_retry(
        operation=_op,
        state=state,
        node_name="generate_roadmap",
        max_retries=3,
        fallback_data={"roadmap": [], "length": 0}
    )


def generate_meta_prompt(state: State) -> Dict[str, Any]:
    """Generate module-specific prompt via Meta LLM with 3-attempt retry."""
    def _op():
        index = state.get("progress", 0)
        roadmap = state.get("roadmap", [])
        if not roadmap or index >= len(roadmap):
            raise IndexError(f"Module index {index} out of range for roadmap of length {len(roadmap)}.")

        topic_info = roadmap[index]
        title = topic_info.get("title", f"Module {index + 1}")
        scope_boundary = topic_info.get("scope_boundary", "Standard module overview.")
        related_files = topic_info.get("related_files", [])
        file_name = topic_info.get("file_slug", f"topic_{index + 1}.md")

        prompt = meta_prompt_genrator.invoke({
            "module_title": title,
            "teaching_style": state.get("teaching_style", ""),
            "verbosity_level": state.get("level", ""),
            "module_scope_boundary": scope_boundary,
            "module_related_files": related_files,
            "methodology": state.get("methodology", ""),
            "depth": state.get("depth", ""),
            "last_completed": "generate_meta_prompt"
        })

        # Call meta LLM to generate the specialized downstream prompt
        res = meta.invoke(prompt)
        meta_prompt_text = res.content if hasattr(res, "content") else str(res)

        print(f"[SUCCESS] Meta-prompt generated for: '{title}'.")
        return {
            "meta_prompt": meta_prompt_text,
            "filename": file_name,
            "last_completed": "generate_meta_prompt"
        }

    return execute_with_retry(
        operation=_op,
        state=state,
        node_name="generate_meta_prompt",
        max_retries=3,
        fallback_data={"meta_prompt": "", "filename": f"module_{state.get('progress', 0)}.md"}
    )


def generate_content(state: State) -> Dict[str, Any]:
    """Generate Markdown content via Qwen LLM with 3-attempt retry."""
    def _op():
        prompt = state.get("meta_prompt")
        if not prompt:
            raise ValueError("No meta_prompt found in state to generate content.")

        result = qwen.invoke(prompt)
        content = result.content if hasattr(result, "content") else str(result)
        if not content or not content.strip():
            raise ValueError("Generated content from model was empty.")

        print("[SUCCESS] Module content successfully generated by Qwen.")
        return {
            "answers": content,
            "last_completed": "generate_content"
        }

    return execute_with_retry(
        operation=_op,
        state=state,
        node_name="generate_content",
        max_retries=3,
        fallback_data={"answers": "# Content generation skipped\n"}
    )


def createFile(state: State) -> Dict[str, Any]:
    """Write generated module content to file with 3-attempt retry and error handling."""
    def _op():
        raw_filename = state.get("filename", f"module_{state.get('progress', 0)}")
        content = state.get("answers", "")
        topic = sanitize_name(state.get("topic", "default_topic"))

        create_folder(topic)

        # Normalize filename and ensure .md extension without double .md
        clean_name = sanitize_name(Path(raw_filename).stem)
        final_filename = f"{clean_name}.md"
        file_path = Path(topic) / final_filename

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"[FILE] Successfully created note: '{file_path}'")
        return {
            "progress": state.get("progress", 0) + 1,
            "last_completed": "createFile"
        }

    return execute_with_retry(
        operation=_op,
        state=state,
        node_name="createFile",
        max_retries=3,
        fallback_data={"progress": state.get("progress", 0) + 1}
    )


def humanIntervention(state: State) -> str:
    """Interactively ask user whether to continue generation of the next topic."""
    try:
        progress = state.get("progress", 0)
        total_length = state.get("length", len(state.get("roadmap", [])))

        if progress >= total_length:
            return "process_completed"

        choice = questionary.confirm(
            f"Module {progress}/{total_length} completed. Do you want to continue to the next module?",
            default=True
        ).ask()

        if choice:
            return "generate_meta_prompt"
        else:
            return "stop_the_process"
    except (KeyboardInterrupt, Exception) as e:
        print(f"[WARN] [humanIntervention] Prompt interrupted ({e}). Stopping gracefully.")
        return "stop_the_process"


def stop_the_process(state: State) -> Dict[str, Any]:
    """Safely halt the pipeline."""
    print("\n[STOPPED] Process Stopped by user or intervention.")
    if state.get("save_on_stop", True):
        save_state(state)
    else:
        print("[INFO] State was not saved as requested.")
    return {
        "last_completed": "stop_the_process"
    }


def process_completed(state: State) -> Dict[str, Any]:
    """Finalize successful pipeline execution and persist state to disk."""
    print("\n[DONE] Process Completed! All modules have been successfully generated.")
    save_state(state)
    return {
        "last_completed": "process_completed"
    }


# Backward-compatibility alias if referenced in older graph definitions
askAI = generate_content
