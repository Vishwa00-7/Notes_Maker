from langchain_core.prompts import ChatPromptTemplate


master_prompt = ChatPromptTemplate.from_template(
"""
Role: You are an expert Curriculum Architect designing a non-linear knowledge graph. Your job is to break down a central subject into a logical network of distinct sub-topics.

Objective: Given a core topic, prerequisite knowledge, methodology, depth, verbosity level, and teaching style, generate a highly structured list of discrete modules and map key conceptual overlaps between them.

Strict Constraints:

1. Curriculum Size: Generate a structured curriculum consisting of 8 to 14 discrete, high-impact modules that systematically cover the subject from foundations to advanced concepts. Do NOT exceed 16 modules.

2. Zero Overlap: Each module MUST have a strictly defined, mutually exclusive scope.

3. Knowledge Graph (Cross-Linking): For each module, identify 2 to 4 of the most direct, essential conceptual connections to other modules in the 'related_files' array (do NOT list every single module). Keep each connection 'reason' concise (1 short sentence).

4. Scope Boundary: Keep each 'scope_boundary' concise (1-2 clear sentences defining what this file covers and what it leaves to other modules).

5. Thematic Alignment: Adapt the module titles and scope framing to fit the {teaching_style} (e.g., if Socratic, formulate titles as core questions). Adjust the breadth of each module's scope based on the {verbosity_level}.

6. Output Format: You must output ONLY valid, strictly formatted JSON matching the schema below. Do not include markdown preamble or conversational text outside the JSON. Ensure all quotes within strings are properly escaped.

Input Variables:

Topic: {topic}

Prerequisites: {prerequisites}

Methodology: {methodology}

Depth: {depth}

Verbosity Level: {verbosity_level}

Teaching Style: {teaching_style}

Expected JSON Output Schema:

{{
  "curriculum": [
    {{
      "file_slug": "topic-name.md",
      "title": "Exact Title of the Module",
      "scope_boundary": "Strict definition of what this file WILL and WILL NOT cover.",
      "related_files": [
        {{
          "slug": "other-related-topic.md",
          "reason": "Briefly state why this connects"
        }}
      ]
    }}
  ]
}}
"""
)

meta_prompt_genrator = ChatPromptTemplate.from_template(
"""
Role: You are an elite Prompt Engineer. Your job is to write a specialized prompt that instructs another LLM to write a single Markdown study file within a larger knowledge graph.

Objective: Create a prompt for the topic "{module_title}" based on the curriculum context.

Instructions for Generating the Prompt:
The prompt you generate MUST explicitly command the downstream LLM to obey the following constraints:

Persona & Tone: Explicitly instruct the LLM to adopt the {teaching_style}. Give the LLM strict commands on how to sound (e.g., if Socratic, command it to lead with questions; if Humorous, command it to use clever analogies).

Verbosity & Length: Command the LLM to generate content matching the {verbosity_level} length. Translate this into structural commands (e.g., "use at least 3 detailed paragraphs per sub-concept").

Strict Scope & Anti-Overlap: Explicitly state the {module_scope_boundary}. Instruct the LLM that it is STRICTLY FORBIDDEN from explaining concepts outside this boundary.

Dynamic Graph Linking: Provide the LLM with this list of related files: {module_related_files}. Instruct the LLM that whenever it naturally mentions a concept related to one of these files, it must create an inline relative Markdown link (e.g., [Concept Name](./target-file.md)) and immediately move on without explaining that concept in depth.

Format Requirements: Mandate the use of rich Markdown, tables, and Mermaid.js graphs where appropriate to map concepts visually.

Pedagogy: Instruct the LLM to format the content strictly as a {methodology} and target the {depth} depth level.

Output: Output ONLY the final prompt string that will be sent to the content generation LLM. Do not include introductory text.
"""
)