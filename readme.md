# 📚 Notes Maker — AI-Powered Knowledge Graph & Curriculum Generator

An intelligent, multi-stage study material and curriculum generation pipeline powered by **LangGraph**, **LangChain**, and high-performance LLM orchestration via **Groq** (`openai/gpt-oss-120b`). 

Instead of generating a monolithic block of text, **Notes Maker** treats a subject as a **non-linear knowledge graph**. It creates an interconnected web of structured Markdown notes featuring cross-links, visual diagrams, strict scope boundaries, and fully tailored pedagogical styles.

## 🚀 Key Features

- **Non-Linear Knowledge Graph Architecture**: Breaks down subjects into modular subtopics (`.md` files) with strict scope boundaries to prevent overlapping explanations, automatically cross-linking related modules using relative Markdown links (`[Topic](./other-topic.md)`).
- **Dynamic AI Model Selection**:
  - At the start of the workflow (in `getInput`), users choose which AI model to use throughout the entire process (`chatgpt`, `meta`, `qwen`, `space_bunny`, etc.).
  - The chosen model executes all three key reasoning stages end-to-end:
    - **Curriculum Architect** (`generate_roadmap`): Generates structured JSON curriculum schemas and maps non-linear relational links using `master_prompt`.
    - **Meta-Prompt Engineer** (`generate_meta_prompt`): Dynamically constructs hyper-focused prompt instructions for each module using `meta_prompt_genrator`.
    - **Content Generator** (`generate_content`): Writes in-depth, rich Markdown notes adhering to pedagogical rules, tables, and Mermaid graphs.
  - The selected model is saved in `<topic>/state.json` and automatically restored when resuming via `continue_from_last_state`.
- **Live Model Output Preview (First 3 Lines)**:
  - Configurable option in `getInput` (`show_preview`) to display live previews of each model's generation in real-time.
  - While processing, displays the first 3 lines of output from each stage (Curriculum Roadmap JSON, Meta-Prompt instructions, and generated Markdown notes) in formatted terminal boxes.
- **Deep Pedagogical Customization**:
  - **Methodologies**: Standard Textbook, Cheat Sheet, Case-Study Driven, Visual/Structural Notes, Flashcard/Q&A, or "Like ChatGPT".
  - **Depth Levels**: Primer / Crash Course, Standard Foundation, Comprehensive Review, Advanced / Specialized, Academic / Theoretical.
  - **Verbosity Levels**: From brief (~300–500 words) up to exhaustive (3000+ words).
  - **Teaching Styles**: Direct & Authoritative, Socratic Method, Conversational, Storytelling, Humorous & Witty.
- **State Persistence & Resumption**:
  - Progress, selected model, preview settings, and roadmap states are tracked in `<topic>/state.json`.
  - Pause anytime and resume exactly from the last completed module.
- **Human-in-the-Loop Interactivity**:
  - Interactive CLI prompts powered by `questionary`.
  - Step-by-step review before generating each module.

---

## 🔄 Graph Workflow & Architecture

The graph execution flow is designed as follows:

```mermaid
flowchart TD
    Start([START]) --> StartingNode{Starting Node}

    %% Branch 1: New Project
    StartingNode -->|getInput| GetInput[Get User Inputs &amp; Select AI Model<br/>Topic, Model, Prerequisites, Style, Depth, etc.]
    GetInput --> GenRoadmap[Generate Roadmap<br/>Curriculum Planner via Selected Model]
    GenRoadmap --> CreateFolder[Create Topic Folder<br/>& Save Initial State]
    CreateFolder --> GenMetaPrompt

    %% Branch 2: Continue Project
    StartingNode -->|continue_from_last_state| LoadState[Load State &amp; Model<br/>Read &lt;topic&gt;/state.json]
    LoadState --> GenMetaPrompt[Generate Meta-Prompt<br/>Craft module instructions via Selected Model]

    %% Generation Cycle
    GenMetaPrompt --> GenContent[Generate Content<br/>Selected Model writes Markdown note]
    GenContent --> CreateFile[Create File<br/>Save &lt;topic&gt;/&lt;slug&gt;.md &amp; Increment Progress]
    CreateFile --> HumanCheck{Human Intervention<br/>Continue process?}

    %% Human loop decisions
    HumanCheck -->|Yes &amp; More Topics Left| GenMetaPrompt
    HumanCheck -->|No / Pause| StopProcess[Stop &amp; Save State<br/>Dump state.json]
    HumanCheck -->|Completed All Topics| Done[Process Completed]

    StopProcess --> End([END])
    Done --> End([END])
```

---

## 📂 Project Structure

```
Notes Maker/
├── Graph.py         # LangGraph definition (StateGraph wiring, edges, and compilation) [WIP]
├── Main.py          # Application entry point to run the compiled graph [WIP]
├── Models.py        # Model registry (AVAILABLE_MODELS, get_model, Groq/OpenRouter chat models)
├── Nodes.py         # Graph node implementations, questionary prompts, and state helpers
├── Prompts.py       # Prompt templates (Master Curriculum Prompt & Meta-Prompt Generator)
├── State.py         # TypedDict State definition for tracking workflow execution
├── .env             # API keys and environment variables (GROQ_API_KEY, OPENROUTER_API_KEY)
├── todo.txt         # Developer notes and active tasks
└── readme.md        # Project documentation
```

---

## 🧩 Component Breakdown

### 1. State Definition (`State.py`)
Tracks everything flowing through the LangGraph pipeline:
- **User Configurations**: `topic`, `selected_model`, `show_preview`, `methodology`, `prequeist_knowledge`, `depth`, `level`, `teaching_style`
- **Curriculum & Roadmap**: `question`, `roadmap` (list of modules with file slugs, titles, scope boundaries, and related files), `length`
- **Execution Progress**: `progress` (index of current topic), `last_completed` (last executed node), `no_of_attempts_local`, `no_of_attempts_global`, `failed_at`
- **Prompt & Content Artifacts**: `prompts` / `meta_prompt`, `filename`, `answers`, `summary`

### 2. Models (`Models.py`)
Models are centrally defined and registered in `AVAILABLE_MODELS`. At runtime, `get_model(state.get("selected_model"))` retrieves the chosen model for all graph operations.

| Key | Model Name | Provider | Parameters | Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| `chatgpt` | `openai/gpt-oss-120b` | Groq | `temp=0.1, max_tokens=8000` | High-capacity reasoning, deep conceptual mapping |
| `meta` | `openai/gpt-oss-20b` | Groq | `temp=0.1, max_tokens=8000` | Fast, lightweight reasoning, low latency |
| `qwen` | `qwen/qwen3.8-27b` | Groq | `temp=0.1, max_tokens=8000` | Rich formatting, structured notes, code & diagrams |
| `space_bunny` | `stealth/space-bunny-alpha` | OpenRouter (Free) | `temp=0.1, max_tokens=8000` | 1M context window, fast inference, strong coding & structure |
| `ling` | `inclusionai/ling-3.0-flash-vl:free` | OpenRouter (Free) | `temp=0.1, max_tokens=8000` | Lightweight multimodal utility |

> [!TIP]
> **Extensibility**: To add any other model, simply initialize it in `Models.py` (either via `init_chat_model` for Groq or `OpenRouterChatModel` for OpenRouter) and register it in `AVAILABLE_MODELS`. It will automatically appear in the interactive CLI selection menu!

### 3. Prompts (`Prompts.py`)
- **`master_prompt`**: Deconstructs a high-level topic into a non-linear curriculum adhering to the defined depth, teaching style, and verbosity. Mandates strict JSON matching the curriculum schema.
- **`meta_prompt_genrator`**: Converts module metadata (title, scope bounds, related links) into an explicit directive for the downstream content generator, requiring rich markdown, Mermaid diagrams, and inline cross-links.

### 4. Nodes & Flow Logic (`Nodes.py`)
- `starting_node`: Prompts user to start fresh (`getInput`) or resume (`continue_from_last_state`).
- `getInput`: Collects topic details, AI model selection (`chatgpt`, `meta`, `qwen`, `space_bunny`), and live output preview toggle (`show_preview`).
- `continue_from_last_state`: Loads `<topic>/state.json` and restores the previously selected model and preview preferences.
- `generate_roadmap`: Generates the curriculum JSON using the selected model via `get_model()`, displaying a live 3-line preview if enabled.
- `generate_meta_prompt`: Builds specialized prompt instructions using the selected model via `get_model()`, displaying a live 3-line preview if enabled.
- `generate_content`: Generates full Markdown note content using the selected model via `get_model()`, displaying a live 3-line preview if enabled.
- `createFile`: Writes `<topic>/<filename>.md`, updates progress count.
- `humanIntervention`: Prompts user whether to generate the next topic or pause.
- `save_state` / `load_state`: Handles serialization to/from `<topic>/state.json`.

---

## 🛠️ Installation & Setup

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone <your-repo-url>
cd "Notes Maker"

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install langchain langchain-core langgraph questionary python-dotenv anyio
```

### 3. Configure API Keys
Create a `.env` file in the root directory. Only `GROQ_API_KEY` is required for active pipeline execution:
```env
GROQ_API_KEY="your_groq_api_key"

# Optional (only if extending Models.py for experimental OpenRouter models like ling)
OPENROUTER_API_KEY="your_openrouter_api_key"
```

---

## ⏳ Current Status & Roadmap

- [x] State schema definition (`State.py`)
- [x] Unified model pipeline configuration (`openai/gpt-oss-120b` via Groq in `Models.py`)
- [x] Master curriculum prompt & Meta-prompt template (`Prompts.py`)
- [x] Node functions, interactive Questionary menus, save/load state logic (`Nodes.py`)
- [x] **Complete Graph Wiring (`Graph.py`)**:
  - Add nodes: `starting_node`, `getInput`, `continue_from_last_state`, `generate_roadmap`, `generate_meta_prompt`, `generate_content`, `createFile`, `humanIntervention`, `stop_the_process`, `process_completed`.
  - Add conditional edges from `starting_node` and `humanIntervention`.
  - Compile the graph (`app = graph.compile()`).
- [x] **CLI Execution Harness (`Main.py`)**:
  - Initialize empty state and invoke the compiled LangGraph with graceful interrupt/exit handling.
- [x] **Robust Output Parsing**:
  - Regex and code-fence parsing in `extract_json_payload` replacing fragile `eval()`.
- [x] **Enhanced Error Handling & Retries**:
  - 3-attempt automatic retry engine with progressive backoff.
  - Interactive `error_human_intervention` (Save & stop, Retry, Continue).
  - State tracking for `no_of_attempts_local`, `no_of_attempts_global`, and `failed_at`.
