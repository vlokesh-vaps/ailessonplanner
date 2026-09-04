import os
import re
import json
import time
import math
from statistics import median

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.generation.pdf_maker import create_lesson_pdfs
from src.retrieval.textbook_retriever import (
    retrieve_documents,
    retrieve_textbook_context_near_pages,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

# The key should be stored in your project .env file:
# GROQ_API_KEY=your_key_here
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "openai/gpt-oss-120b"

# RAG settings.
# These match the retrieval approach that was tested successfully.
RAG_INITIAL_K = 5
RAG_PAGE_RADIUS = 10

# CALL 1 gets a small region-aware context using only the teacher's topic.
# No Groq query-generation call is used anymore.
CALL1_RAG_K = 8

# After CALL 1 identifies the actual sourceCoverage topics, use one
# strongest chunk per topic for CALL 2 and verification.
TARGETED_RAG_K = 1
VERIFICATION_RAG_K = 1


# ============================================================
# 2. RAG TEXTBOOK RETRIEVAL
# ============================================================

def build_rag_context(topic):
    """
    Build CALL 1 context WITHOUT using Groq to generate retrieval queries.

    Pipeline:
    1. Search Chroma using the teacher's topic.
    2. Estimate the dominant textbook region using the median page.
    3. Re-run a region-aware Chroma search for the same topic.
    4. CALL 1 then identifies sourceCoverage topics itself.
    """

    print("\n========================================")
    print("RAG TEXTBOOK RETRIEVAL")
    print("========================================")

    print("\nSTEP 1: Initial textbook search...")

    initial_documents = retrieve_documents(
        topic,
        k_per_query=RAG_INITIAL_K
    )

    if not initial_documents:
        raise ValueError(
            f"No textbook material could be retrieved for topic '{topic}'."
        )

    initial_pages = [
        document.metadata.get("page")
        for document in initial_documents
        if document.metadata.get("page") is not None
    ]

    if not initial_pages:
        raise ValueError(
            "Retrieved textbook chunks do not contain page metadata."
        )

    centre_page = round(
        median(initial_pages)
    )

    print(
        f"Initial retrieved vector pages: {initial_pages}"
    )
    print(
        f"Estimated topic centre vector page: {centre_page}"
    )

    print(
        "\nSTEP 2: Region-aware retrieval using the teacher topic..."
    )

    context_content = retrieve_textbook_context_near_pages(
        topic,
        centre_page=centre_page,
        page_radius=RAG_PAGE_RADIUS,
        k_per_query=CALL1_RAG_K
    )

    if not context_content.strip():
        raise ValueError(
            f"No final textbook context could be retrieved for topic '{topic}'."
        )

    final_vector_pages = sorted({
        int(match)
        for match in re.findall(
            r"SOURCE PAGE:\s*(\d+)",
            context_content
        )
    })

    if final_vector_pages:
        selected_start_page = min(final_vector_pages) + 1
        selected_end_page = max(final_vector_pages) + 1
    else:
        selected_start_page = max(
            1,
            centre_page - RAG_PAGE_RADIUS + 1
        )
        selected_end_page = (
            centre_page + RAG_PAGE_RADIUS + 1
        )

    print("\n----------------------------------------")
    print("RAG RETRIEVAL SUMMARY")
    print("----------------------------------------")
    print(f"Requested topic: {topic}")
    print(f"Initial vector pages: {initial_pages}")
    print(f"Median centre vector page: {centre_page}")
    print(
        f"Final retrieved vector pages: {final_vector_pages}"
    )
    print(
        f"User-facing PDF page range: "
        f"{selected_start_page}-{selected_end_page}"
    )
    print(
        f"CALL 1 context characters: {len(context_content):,}"
    )
    print("----------------------------------------")

    return (
        context_content,
        selected_start_page,
        selected_end_page,
        centre_page
    )


# ============================================================
# 2B. TARGETED RAG AFTER CALL 1
# ============================================================

def build_targeted_rag_context(
    call_1_result,
    centre_page,
    k_per_query=TARGETED_RAG_K
):
    """
    Use CALL 1 sourceCoverage topic names directly as Chroma queries.
    This replaces the old Groq retrieval-query-generation call.
    """

    source_coverage = call_1_result.get(
        "sourceCoverage",
        []
    )

    queries = []

    for item in source_coverage:
        topic_name = str(
            item.get("topic", "")
        ).strip()

        if (
            topic_name
            and topic_name.lower()
            not in {existing.lower() for existing in queries}
        ):
            queries.append(topic_name)

    if not queries:
        raise ValueError(
            "CALL 1 did not produce sourceCoverage topics "
            "for targeted RAG retrieval."
        )

    print("\n========================================")
    print("TARGETED RAG FOR CALL 2")
    print("========================================")
    print("Using CALL 1 sourceCoverage as retrieval queries:")

    for query in queries:
        print(f" - {query}")

    targeted_context = retrieve_textbook_context_near_pages(
        queries,
        centre_page=centre_page,
        page_radius=RAG_PAGE_RADIUS,
        k_per_query=k_per_query
    )

    if not targeted_context.strip():
        raise ValueError(
            "Targeted RAG retrieval returned no textbook material."
        )

    targeted_pages = sorted({
        int(match)
        for match in re.findall(
            r"SOURCE PAGE:\s*(\d+)",
            targeted_context
        )
    })

    print(
        f"Targeted retrieved vector pages: {targeted_pages}"
    )
    print(
        f"Targeted context characters: {len(targeted_context):,}"
    )
    print("----------------------------------------")

    return targeted_context


# ============================================================
# 2C. SMALL RAG CONTEXT FOR CALL 3 VERIFICATION
# ============================================================

def build_verification_rag_context(
    call_1_result,
    centre_page,
    k_per_query=VERIFICATION_RAG_K
):
    """
    Verification only needs the strongest source evidence for each
    sourceCoverage topic, so keep this context deliberately small.
    """

    queries = []

    for item in call_1_result.get(
        "sourceCoverage",
        []
    ):
        topic_name = str(
            item.get("topic", "")
        ).strip()

        if (
            topic_name
            and topic_name.lower()
            not in {existing.lower() for existing in queries}
        ):
            queries.append(topic_name)

    if not queries:
        raise ValueError(
            "No sourceCoverage topics are available "
            "for verification retrieval."
        )

    verification_context = retrieve_textbook_context_near_pages(
        queries,
        centre_page=centre_page,
        page_radius=RAG_PAGE_RADIUS,
        k_per_query=k_per_query
    )

    if not verification_context.strip():
        raise ValueError(
            "Verification RAG retrieval returned no textbook material."
        )

    verification_pages = sorted({
        int(match)
        for match in re.findall(
            r"SOURCE PAGE:\s*(\d+)",
            verification_context
        )
    })

    print("\n========================================")
    print("SMALL RAG CONTEXT FOR CALL 3")
    print("========================================")
    print(
        f"Verification retrieved vector pages: {verification_pages}"
    )
    print(
        f"Verification context characters: "
        f"{len(verification_context):,}"
    )
    print("----------------------------------------")

    return verification_context


# ============================================================
# 3. CALL 1 - SHARED LESSON PLAN
# ============================================================

def build_shared_plan_prompt(
    context_content,
    topic,
    level,
    lesson_length
):
    """
    Combined replacement for the old CALL 1 + CALL 2.

    Generates:
    - lesson metadata
    - sourceCoverage
    - learning objectives
    - prerequisite knowledge
    - key terms
    - questions + expected answers

    Key points are NOT generated here. They are derived later from
    coreNotes summaries so the same content is not paid for twice.
    """

    topic_instruction = (
        topic
        if topic
        else "Determine the topic from the supplied source"
    )

    level_instruction = (
        level
        if level
        else "Determine the level from the supplied source"
    )

    return f"""
Create the shared lesson plan from the textbook material.

LESSON:
Topic: {topic_instruction}
Level: {level_instruction}
Length: {lesson_length} minutes

TEXTBOOK:
{context_content}

SCOPE:
Keep the lesson centred on the requested topic. Nearby textbook
material may be used as prerequisite/context, but do not turn it into
a main lesson topic unless it is directly part of the requested topic.

GROUNDING:
Use the textbook as the factual basis.
Do not invent equations, classifications, definitions, observations,
measurements, constants or curriculum requirements.
Do not introduce more advanced terminology than the source.
Do not classify an example unless the source supports the classification
or it exactly satisfies the source definition.
Verify equations before describing them as balanced/correct.

GENERATE:

1. lessonMetadata
- topic
- level
- lessonLengthMinutes
- scopeNote

2. sourceCoverage
Use T1, T2, ...
Each item:
- id
- topic
- importance: high | medium | low
- estimatedTeachingMinutes

Cover every major part of the requested topic, but avoid trivial details.

3. learningObjectives
Generate 4-6 using LO1, LO2, ...
Each:
- id
- objective
- relatedSourceTopics
- estimatedTeachingMinutes
- teacherFocus

Every T ID must appear in at least one objective.

4. prerequisiteKnowledge
Generate 2-5 concise prerequisite points.

5. keyTerms
Generate 6-10 important textbook terms.
Each:
- term
- definition

6. questions
Generate 4-6 concise questions across the lesson.
Each:
- id
- question
- expectedAnswer
- type
- relatedLearningObjectives
- relatedSourceTopics

Use only existing LO and T IDs.

Do NOT generate:
- keyPoints
- coreNotes
- activities
- realWorldApplications
- activityRanking
- teacher schedule

Return JSON only in this structure:

{{
  "lessonMetadata": {{
    "topic": "{topic if topic else 'Determine from source'}",
    "level": "{level if level else 'Determine from source'}",
    "lessonLengthMinutes": {lesson_length},
    "scopeNote": ""
  }},
  "sourceCoverage": [
    {{
      "id": "T1",
      "topic": "",
      "importance": "high",
      "estimatedTeachingMinutes": 5
    }}
  ],
  "shared": {{
    "learningObjectives": [
      {{
        "id": "LO1",
        "objective": "",
        "relatedSourceTopics": ["T1"],
        "estimatedTeachingMinutes": 6,
        "teacherFocus": ""
      }}
    ],
    "prerequisiteKnowledge": [""],
    "keyTerms": [
      {{
        "term": "",
        "definition": ""
      }}
    ],
    "questions": [
      {{
        "id": "Q1",
        "question": "",
        "expectedAnswer": "",
        "type": "",
        "relatedLearningObjectives": ["LO1"],
        "relatedSourceTopics": ["T1"]
      }}
    ]
  }}
}}
"""


# ============================================================
# 4. CALL 2 - STUDENT NOTES + ENRICHMENT
# ============================================================

def build_content_prompt(
    context_content,
    shared_result
):
    """
    Combined replacement for the old CALL 3 + CALL 4.

    Generates:
    - coreNotes with a short summary per section
    - activities
    - real-world applications
    - activity ranking

    Python later derives shared.keyPoints from coreNotes[].summary.
    """

    shared = shared_result.get(
        "shared",
        {}
    )

    compact_context = json.dumps(
        {
            "sourceCoverage": [
                {
                    "id": item.get("id"),
                    "topic": item.get("topic")
                }
                for item in shared_result.get(
                    "sourceCoverage",
                    []
                )
            ],
            "learningObjectives": [
                {
                    "id": item.get("id"),
                    "objective": item.get("objective"),
                    "relatedSourceTopics": item.get(
                        "relatedSourceTopics",
                        []
                    )
                }
                for item in shared.get(
                    "learningObjectives",
                    []
                )
            ]
        },
        ensure_ascii=False,
        separators=(",", ":")
    )

    return f"""
Create student notes and enrichment for this lesson.

TEXTBOOK:
{context_content}

LESSON STRUCTURE:
{compact_context}

GROUNDING:
Use the textbook as the factual basis.
Do not invent classifications, equations, observations, colours,
products, measurements or advanced terminology.
Do not classify an example unless the source supports it or it exactly
matches the source definition.
Verify equations before describing them as balanced/correct.
Preserve source-supported states, colours, products and conditions.

CORE NOTES:
Generate 5-7 logical sections.

Each section must contain:
- heading
- summary
- content
- relatedSourceTopics

summary:
- one concise key point for that section
- must be factually supported by the textbook
- Python will later use these summaries as the PDF Key Points section

content:
- student-friendly revision notes
- enough detail to cover the source topic
- preserve important equations, methods and examples
- no teacher instructions
- no answer-key wording

Every sourceCoverage T ID must appear in at least one section.

Formatting:
If content contains bullets, put each "•" bullet on its own line.

ACTIVITIES:
Generate 1-2 optional activities.
Each:
- id: A1, A2
- title
- estimatedMinutes: integer 10-15
- description
- relatedLearningObjectives
- relatedSourceTopics

REAL-WORLD APPLICATIONS:
Generate 1-2 where supported by the textbook.
Each:
- id: RW1, RW2
- title
- explanation
- question
- expectedAnswer
- estimatedMinutes
- relatedLearningObjectives
- relatedSourceTopics

ACTIVITY RANKING:
Rank every generated activity exactly once from best to worst.
Return only activity IDs.

Return JSON only:

{{
  "studentOnly": {{
    "coreNotes": [
      {{
        "heading": "",
        "summary": "",
        "content": "",
        "relatedSourceTopics": ["T1"]
      }}
    ]
  }},
  "activities": [
    {{
      "id": "A1",
      "title": "",
      "estimatedMinutes": 12,
      "description": "",
      "relatedLearningObjectives": ["LO1"],
      "relatedSourceTopics": ["T1"]
    }}
  ],
  "realWorldApplications": [
    {{
      "id": "RW1",
      "title": "",
      "explanation": "",
      "question": "",
      "expectedAnswer": "",
      "estimatedMinutes": 3,
      "relatedLearningObjectives": ["LO1"],
      "relatedSourceTopics": ["T1"]
    }}
  ],
  "activityRanking": ["A1", "A2"]
}}
"""


# ============================================================
# 5. DERIVE KEY POINTS FROM CORE NOTES
# ============================================================

def derive_key_points_from_core_notes(
    core_notes
):
    """
    Key Points and Core Notes contain the same academic ideas at
    different levels of detail.

    Instead of paying Groq to generate both independently, derive the
    Key Points section directly from each core-note summary.
    """

    key_points = []

    for note in core_notes:
        summary = str(
            note.get(
                "summary",
                ""
            )
        ).strip()

        related_topics = note.get(
            "relatedSourceTopics",
            []
        )

        if summary:
            key_points.append({
                "point": summary,
                "relatedSourceTopics": related_topics
            })

    return key_points


# ============================================================
# 6. CALL 3 - FINAL FACTUAL VERIFICATION
# ============================================================

def build_verification_prompt(
    source_context,
    lesson_plan_for_verification
):
    compact_lesson = json.dumps(
        lesson_plan_for_verification,
        ensure_ascii=False,
        separators=(",", ":")
    )

    return f"""
Verify the generated lesson against the textbook.

TEXTBOOK:
{source_context}

LESSON:
{compact_lesson}

Check only factual/source-grounding errors:
- wrong classification
- incorrect or unbalanced equations
- wrong products, states, colours, precipitates or conditions
- unsupported definitions or advanced terminology
- examples placed under the wrong category
- statements that conflict with the textbook

Do not redesign the lesson.
Do not change IDs, timings, list order or structure.
Patch only existing fields.

Return JSON only.

If correct:
{{
  "valid": true,
  "issues": []
}}

If corrections are needed:
{{
  "valid": false,
  "issues": [
    {{
      "path": "studentOnly.coreNotes[0].content",
      "problem": "brief reason",
      "replacement": "complete corrected value for that field"
    }}
  ]
}}

Use the smallest number of patches possible.
"""


def build_compact_verification_target(
    shared_result,
    student_result
):
    shared = shared_result.get(
        "shared",
        {}
    )

    return {
        "sourceCoverage": [
            {
                "id": item.get("id"),
                "topic": item.get("topic")
            }
            for item in shared_result.get(
                "sourceCoverage",
                []
            )
        ],

        "shared": {
            "learningObjectives": [
                {
                    "id": item.get("id"),
                    "objective": item.get("objective"),
                    "teacherFocus": item.get("teacherFocus")
                }
                for item in shared.get(
                    "learningObjectives",
                    []
                )
            ],

            "questions": [
                {
                    "id": item.get("id"),
                    "question": item.get("question"),
                    "expectedAnswer": item.get("expectedAnswer")
                }
                for item in shared.get(
                    "questions",
                    []
                )
            ],

            "keyTerms": [
                {
                    "term": item.get("term"),
                    "definition": item.get("definition")
                }
                for item in shared.get(
                    "keyTerms",
                    []
                )
            ],

            "realWorldApplications": [
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "explanation": item.get("explanation"),
                    "question": item.get("question"),
                    "expectedAnswer": item.get("expectedAnswer")
                }
                for item in shared.get(
                    "realWorldApplications",
                    []
                )
            ]
        },

        "studentOnly": {
            "coreNotes": [
                {
                    "heading": item.get("heading"),
                    "summary": item.get("summary"),
                    "content": item.get("content")
                }
                for item in student_result.get(
                    "studentOnly",
                    {}
                ).get(
                    "coreNotes",
                    []
                )
            ]
        }
    }


def _path_tokens(path):
    """
    Convert a simple path such as:
    shared.keyTerms[0].definition
    into:
    ["shared", "keyTerms", 0, "definition"]
    """

    tokens = []

    for name, index in re.findall(
        r"([^\.\[\]]+)|\[(\d+)\]",
        path
    ):
        if name:
            tokens.append(name)
        else:
            tokens.append(int(index))

    return tokens


def apply_verification_patches(
    lesson_plan,
    verification_result
):
    """
    Apply only patches that point to an existing field.

    Invalid paths are skipped rather than allowing the verifier to
    change the JSON structure unexpectedly.
    """

    issues = verification_result.get(
        "issues",
        []
    )

    applied_count = 0

    for issue in issues:

        path = str(
            issue.get(
                "path",
                ""
            )
        ).strip()

        if not path:
            print(
                "Skipping verification issue with no path."
            )
            continue

        if "replacement" not in issue:
            print(
                f"Skipping verification issue without replacement: {path}"
            )
            continue

        tokens = _path_tokens(
            path
        )

        if not tokens:
            print(
                f"Skipping invalid verification path: {path}"
            )
            continue

        current = lesson_plan
        valid_path = True

        try:
            for token in tokens[:-1]:
                if isinstance(token, int):
                    if (
                        not isinstance(current, list)
                        or token < 0
                        or token >= len(current)
                    ):
                        valid_path = False
                        break

                    current = current[token]

                else:
                    if (
                        not isinstance(current, dict)
                        or token not in current
                    ):
                        valid_path = False
                        break

                    current = current[token]

            if not valid_path:
                print(
                    f"Skipping non-existent verification path: {path}"
                )
                continue

            final_token = tokens[-1]

            if isinstance(final_token, int):
                if (
                    not isinstance(current, list)
                    or final_token < 0
                    or final_token >= len(current)
                ):
                    print(
                        f"Skipping non-existent verification path: {path}"
                    )
                    continue

                current[final_token] = clean_ai_strings(
                    issue["replacement"]
                )

            else:
                if (
                    not isinstance(current, dict)
                    or final_token not in current
                ):
                    print(
                        f"Skipping non-existent verification path: {path}"
                    )
                    continue

                current[final_token] = clean_ai_strings(
                    issue["replacement"]
                )

            applied_count += 1

            print(
                f"Applied QA correction: {path}"
            )
            print(
                f"  Problem: {issue.get('problem', '')}"
            )

        except Exception as error:
            print(
                f"Could not apply QA correction at {path}: {error}"
            )

    return applied_count


def validate_after_verification(
    lesson_plan,
    activity_ranking
):
    """
    Re-check the final structure after CALL 3 factual patches.
    """

    print(
        "\n========================================"
    )
    print(
        "POST-VERIFICATION VALIDATION"
    )
    print(
        "========================================"
    )

    call_1_like = {
        "lessonMetadata":
            lesson_plan.get(
                "lessonMetadata",
                {}
            ),

        "sourceCoverage":
            lesson_plan.get(
                "sourceCoverage",
                []
            ),

        "shared": {
            "learningObjectives":
                lesson_plan.get(
                    "shared",
                    {}
                ).get(
                    "learningObjectives",
                    []
                ),

            "prerequisiteKnowledge":
                lesson_plan.get(
                    "shared",
                    {}
                ).get(
                    "prerequisiteKnowledge",
                    []
                ),

            "keyTerms":
                lesson_plan.get(
                    "shared",
                    {}
                ).get(
                    "keyTerms",
                    []
                ),

            "questions":
                lesson_plan.get(
                    "shared",
                    {}
                ).get(
                    "questions",
                    []
                )
        }
    }

    call_2_like = {
        "studentOnly":
            lesson_plan.get(
                "studentOnly",
                {}
            ),

        "activities":
            lesson_plan.get(
                "shared",
                {}
            ).get(
                "activities",
                []
            ),

        "realWorldApplications":
            lesson_plan.get(
                "shared",
                {}
            ).get(
                "realWorldApplications",
                []
            ),

        "activityRanking":
            activity_ranking
    }

    valid = all([
        validate_call_1(
            call_1_like
        ),

        validate_call_2(
            call_2_like,
            lesson_plan
        )
    ])

    if valid:
        print(
            "POST-VERIFICATION VALIDATION PASSED"
        )
    else:
        print(
            "POST-VERIFICATION VALIDATION FAILED"
        )

    print(
        "========================================"
    )

    return valid


# ============================================================
# 6. CLEAN / NORMALISE AI TEXT
# ============================================================

def clean_ai_strings(value):
    """
    Recursively clean strings returned by the AI.

    Fixes:
    - literal escaped newline/tab sequences such as "\\n"
    - bullet points that the model places inline instead of
      starting them on separate lines
    """

    if isinstance(value, dict):
        return {
            key: clean_ai_strings(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            clean_ai_strings(item)
            for item in value
        ]

    if isinstance(value, str):

        value = (
            value
            .replace("\\r\\n", "\n")
            .replace("\\n", "\n")
            .replace("\\r", "\n")
            .replace("\\t", " ")
        )

        # Put every bullet point on its own line.
        value = re.sub(
            r"[ \t]*•[ \t]*",
            "\n• ",
            value
        )

        # Remove whitespace immediately before line breaks.
        value = re.sub(
            r"[ \t]+\n",
            "\n",
            value
        )

        # Prevent excessive blank lines.
        value = re.sub(
            r"\n{3,}",
            "\n\n",
            value
        )

        return value.strip()

    return value


# ============================================================
# 6. GROQ CALL THROUGH LANGCHAIN
# ============================================================

def call_groq(
    prompt,
    max_tokens,
    max_retries=3,
    temperature=0.2,
    force_json=True
):
    """
    Send one lesson-planning call to Groq through LangChain.

    The prompts and returned JSON structure are kept the same as
    the original planner. LangChain is only replacing the old
    direct requests.post(...) transport layer here.
    """

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=temperature,
        max_tokens=max_tokens
    )

    if force_json:
        runnable_llm = llm.bind(
            response_format={
                "type": "json_object"
            }
        )
    else:
        runnable_llm = llm

    messages = [
        (
            "system",
            (
                "You are an expert educator and lesson designer. "
                "Use the supplied source as the primary factual basis. "
                "Return valid JSON only."
            )
        ),
        (
            "human",
            prompt
        )
    ]

    for attempt in range(
        max_retries + 1
    ):

        try:
            response = runnable_llm.invoke(
                messages
            )

            response_metadata = getattr(
                response,
                "response_metadata",
                {}
            ) or {}

            usage_metadata = getattr(
                response,
                "usage_metadata",
                {}
            ) or {}

            token_usage = response_metadata.get(
                "token_usage",
                {}
            ) or {}

            finish_reason = response_metadata.get(
                "finish_reason"
            )

            prompt_tokens = (
                usage_metadata.get(
                    "input_tokens"
                )
                or token_usage.get(
                    "prompt_tokens"
                )
                or "unknown"
            )

            completion_tokens = (
                usage_metadata.get(
                    "output_tokens"
                )
                or token_usage.get(
                    "completion_tokens"
                )
                or "unknown"
            )

            total_tokens = (
                usage_metadata.get(
                    "total_tokens"
                )
                or token_usage.get(
                    "total_tokens"
                )
                or "unknown"
            )

            print(
                f"Finish reason: "
                f"{finish_reason}"
            )

            print(
                f"Prompt tokens: "
                f"{prompt_tokens}"
            )

            print(
                f"Completion tokens: "
                f"{completion_tokens}"
            )

            print(
                f"Total tokens: "
                f"{total_tokens}"
            )

            if finish_reason == "length":
                print(
                    "\nERROR: Model hit output token limit."
                )

                return None

            ai_output = response.content

            if isinstance(
                ai_output,
                list
            ):
                # Defensive handling for content-block style responses.
                text_parts = []

                for block in ai_output:
                    if isinstance(
                        block,
                        dict
                    ):
                        text_parts.append(
                            str(
                                block.get(
                                    "text",
                                    ""
                                )
                            )
                        )
                    else:
                        text_parts.append(
                            str(block)
                        )

                ai_output = "".join(
                    text_parts
                )

            try:
                ai_output = str(
                    ai_output
                ).strip()

                if ai_output.startswith(
                    "```"
                ):
                    ai_output = re.sub(
                        r"^```(?:json)?\s*",
                        "",
                        ai_output,
                        flags=re.IGNORECASE
                    )

                    ai_output = re.sub(
                        r"\s*```$",
                        "",
                        ai_output
                    ).strip()

                # Defensive fallback for non-enforced JSON mode:
                # keep only the outermost JSON object if the model adds
                # a short sentence before/after it.
                first_brace = ai_output.find("{")
                last_brace = ai_output.rfind("}")

                if (
                    first_brace != -1
                    and last_brace != -1
                    and last_brace > first_brace
                ):
                    ai_output = ai_output[
                        first_brace:
                        last_brace + 1
                    ]

                parsed_output = json.loads(
                    ai_output
                )

                parsed_output = clean_ai_strings(
                    parsed_output
                )

                return parsed_output

            except json.JSONDecodeError as error:
                print(
                    "\nERROR: Invalid JSON."
                )

                print(
                    f"JSON error: {error}"
                )

                print(
                    "\nRAW OUTPUT:\n"
                )

                print(
                    ai_output
                )

                if attempt < max_retries:
                    print(
                        "\nRetrying in 5 seconds..."
                    )

                    time.sleep(
                        5
                    )

                    continue

                return None

        except Exception as error:
            error_message = str(
                error
            )

            is_request_too_large = (
                "413" in error_message
                or "request too large" in error_message.lower()
            )

            if is_request_too_large:
                print(
                    "\nERROR: Groq rejected this request because "
                    "it is too large for the current token limit."
                )
                print(
                    error_message
                )
                print(
                    "This request will not be retried because "
                    "waiting does not make an oversized request smaller."
                )
                return None

            is_rate_limit = (
                "429" in error_message
                or "rate limit" in error_message.lower()
            )

            if (
                attempt < max_retries
            ):
                if is_rate_limit:
                    match = re.search(
                        r"try again in ([0-9.]+)s",
                        error_message,
                        re.IGNORECASE
                    )

                    if match:
                        wait_seconds = (
                            float(
                                match.group(1)
                            )
                            + 2
                        )
                    else:
                        wait_seconds = 65

                    print(
                        "\nRate limit reached."
                    )

                    print(
                        error_message
                    )

                    print(
                        f"Waiting {wait_seconds:.1f} "
                        f"seconds..."
                    )
                else:
                    wait_seconds = 5

                    print(
                        "\nGroq/LangChain request failed."
                    )

                    print(
                        error_message
                    )

                    print(
                        f"Retrying in "
                        f"{wait_seconds} seconds..."
                    )

                time.sleep(
                    wait_seconds
                )

                continue

            print(
                "\nGroq/LangChain request failed "
                "after all retries."
            )

            print(
                error_message
            )

            return None

    return None


# ============================================================
# 7. NORMALISE CALL 1 SCHEMA
# ============================================================

def normalise_call_1_result(
    call_1_result
):
    if not isinstance(
        call_1_result,
        dict
    ):
        return call_1_result

    shared = call_1_result.get(
        "shared"
    )

    if not isinstance(
        shared,
        dict
    ):
        shared = {}
        call_1_result[
            "shared"
        ] = shared

    return call_1_result


# ============================================================
# 8. VALIDATE CALL 1
# ============================================================

def validate_call_1(
    call_1_result
):
    print("\n----------------------------------------")
    print("VALIDATING CALL 1 - SHARED LESSON PLAN")
    print("----------------------------------------")

    source_coverage = call_1_result.get(
        "sourceCoverage",
        []
    )

    shared = call_1_result.get(
        "shared",
        {}
    )

    objectives = shared.get(
        "learningObjectives",
        []
    )

    prerequisites = shared.get(
        "prerequisiteKnowledge",
        []
    )

    key_terms = shared.get(
        "keyTerms",
        []
    )

    questions = shared.get(
        "questions",
        []
    )

    print(f"Source topics: {len(source_coverage)}")
    print(f"Learning objectives: {len(objectives)}")
    print(f"Prerequisites: {len(prerequisites)}")
    print(f"Key terms: {len(key_terms)}")
    print(f"Questions: {len(questions)}")

    valid = True

    if not source_coverage:
        print("ERROR: No sourceCoverage.")
        valid = False

    if not 4 <= len(objectives) <= 6:
        print("ERROR: Expected 4-6 learning objectives.")
        valid = False

    if not 2 <= len(prerequisites) <= 5:
        print("ERROR: Expected 2-5 prerequisite points.")
        valid = False

    if not 6 <= len(key_terms) <= 10:
        print("ERROR: Expected 6-10 key terms.")
        valid = False

    if not 4 <= len(questions) <= 6:
        print("ERROR: Expected 4-6 questions.")
        valid = False

    topic_ids = {
        item.get("id")
        for item in source_coverage
        if item.get("id")
    }

    objective_ids = {
        item.get("id")
        for item in objectives
        if item.get("id")
    }

    covered_topics = set()

    for objective in objectives:
        covered_topics.update(
            objective.get(
                "relatedSourceTopics",
                []
            )
        )

        estimated = objective.get(
            "estimatedTeachingMinutes"
        )

        if (
            not isinstance(
                estimated,
                (int, float)
            )
            or estimated <= 0
        ):
            print(
                f"ERROR: Invalid teaching time for "
                f"{objective.get('id')}."
            )
            valid = False

        if not str(
            objective.get(
                "teacherFocus",
                ""
            )
        ).strip():
            print(
                f"ERROR: Missing teacherFocus for "
                f"{objective.get('id')}."
            )
            valid = False

    missing_topics = (
        topic_ids
        - covered_topics
    )

    if missing_topics:
        print(
            "ERROR: Source topics missing from objectives: "
            f"{sorted(missing_topics)}"
        )
        valid = False

    for topic in source_coverage:
        if topic.get(
            "importance"
        ) not in {
            "high",
            "medium",
            "low"
        }:
            print(
                f"ERROR: Invalid importance for "
                f"{topic.get('id')}."
            )
            valid = False

        estimated = topic.get(
            "estimatedTeachingMinutes"
        )

        if (
            not isinstance(
                estimated,
                (int, float)
            )
            or estimated <= 0
        ):
            print(
                f"ERROR: Invalid source topic time for "
                f"{topic.get('id')}."
            )
            valid = False

    for question in questions:
        invalid_los = (
            set(
                question.get(
                    "relatedLearningObjectives",
                    []
                )
            )
            - objective_ids
        )

        invalid_topics = (
            set(
                question.get(
                    "relatedSourceTopics",
                    []
                )
            )
            - topic_ids
        )

        if invalid_los:
            print(
                f"ERROR: {question.get('id')} invalid LO IDs: "
                f"{sorted(invalid_los)}"
            )
            valid = False

        if invalid_topics:
            print(
                f"ERROR: {question.get('id')} invalid T IDs: "
                f"{sorted(invalid_topics)}"
            )
            valid = False

    print("----------------------------------------")

    if valid:
        print("CALL 1 VALIDATION PASSED")
    else:
        print("CALL 1 VALIDATION FAILED")

    print("----------------------------------------")

    return valid


# ============================================================
# 9. VALIDATE CALL 2
# ============================================================

def validate_call_2(
    call_2_result,
    shared_result
):
    print("\n----------------------------------------")
    print("VALIDATING CALL 2 - NOTES + ENRICHMENT")
    print("----------------------------------------")

    notes = call_2_result.get(
        "studentOnly",
        {}
    ).get(
        "coreNotes",
        []
    )

    activities = call_2_result.get(
        "activities",
        []
    )

    applications = call_2_result.get(
        "realWorldApplications",
        []
    )

    ranking = call_2_result.get(
        "activityRanking",
        []
    )

    print(f"Core-note sections: {len(notes)}")
    print(f"Activities: {len(activities)}")
    print(f"Real-world applications: {len(applications)}")
    print(f"Activity ranking: {ranking}")

    valid = True

    if not 5 <= len(notes) <= 7:
        print("ERROR: Expected 5-7 core-note sections.")
        valid = False

    topic_ids = {
        item.get("id")
        for item in shared_result.get(
            "sourceCoverage",
            []
        )
        if item.get("id")
    }

    objective_ids = {
        item.get("id")
        for item in shared_result.get(
            "shared",
            {}
        ).get(
            "learningObjectives",
            []
        )
        if item.get("id")
    }

    covered_topics = set()

    for note in notes:
        covered_topics.update(
            note.get(
                "relatedSourceTopics",
                []
            )
        )

        if not str(
            note.get(
                "summary",
                ""
            )
        ).strip():
            print(
                f"ERROR: Core note '{note.get('heading')}' "
                "has no summary."
            )
            valid = False

    missing_topics = (
        topic_ids
        - covered_topics
    )

    if missing_topics:
        print(
            "ERROR: Core notes missing topics: "
            f"{sorted(missing_topics)}"
        )
        valid = False

    if not 1 <= len(activities) <= 2:
        print("ERROR: Expected 1-2 activities.")
        valid = False

    activity_ids = {
        item.get("id")
        for item in activities
        if item.get("id")
    }

    for activity in activities:
        duration = activity.get(
            "estimatedMinutes"
        )

        if (
            not isinstance(
                duration,
                int
            )
            or not 10 <= duration <= 15
        ):
            print(
                f"ERROR: Invalid duration for "
                f"{activity.get('id')}: {duration}"
            )
            valid = False

        if (
            set(
                activity.get(
                    "relatedLearningObjectives",
                    []
                )
            )
            - objective_ids
        ):
            print(
                f"ERROR: Invalid LO IDs in "
                f"{activity.get('id')}."
            )
            valid = False

        if (
            set(
                activity.get(
                    "relatedSourceTopics",
                    []
                )
            )
            - topic_ids
        ):
            print(
                f"ERROR: Invalid T IDs in "
                f"{activity.get('id')}."
            )
            valid = False

    if not 1 <= len(applications) <= 2:
        print(
            "ERROR: Expected 1-2 real-world applications."
        )
        valid = False

    for application in applications:
        if (
            set(
                application.get(
                    "relatedLearningObjectives",
                    []
                )
            )
            - objective_ids
        ):
            print(
                f"ERROR: Invalid LO IDs in "
                f"{application.get('id')}."
            )
            valid = False

        if (
            set(
                application.get(
                    "relatedSourceTopics",
                    []
                )
            )
            - topic_ids
        ):
            print(
                f"ERROR: Invalid T IDs in "
                f"{application.get('id')}."
            )
            valid = False

    ranking_ids = [
        activity_id
        for activity_id in ranking
        if activity_id
    ]

    if (
        set(ranking_ids) != activity_ids
        or len(ranking_ids) != len(set(ranking_ids))
    ):
        print(
            "ERROR: Activity ranking must contain every "
            "activity exactly once."
        )
        valid = False

    print("----------------------------------------")

    if valid:
        print("CALL 2 VALIDATION PASSED")
    else:
        print("CALL 2 VALIDATION FAILED")

    print("----------------------------------------")

    return valid


# ============================================================
# 10. POST-VERIFICATION STRUCTURAL VALIDATION
# ============================================================

def validate_after_verification(
    lesson_plan,
    activity_ranking
):
    """
    Re-check the final structure after CALL 3 patches.
    """

    call_1_like = {
        "lessonMetadata":
            lesson_plan.get(
                "lessonMetadata",
                {}
            ),
        "sourceCoverage":
            lesson_plan.get(
                "sourceCoverage",
                []
            ),
        "shared": {
            "learningObjectives":
                lesson_plan.get(
                    "shared",
                    {}
                ).get(
                    "learningObjectives",
                    []
                ),
            "prerequisiteKnowledge":
                lesson_plan.get(
                    "shared",
                    {}
                ).get(
                    "prerequisiteKnowledge",
                    []
                ),
            "keyTerms":
                lesson_plan.get(
                    "shared",
                    {}
                ).get(
                    "keyTerms",
                    []
                ),
            "questions":
                lesson_plan.get(
                    "shared",
                    {}
                ).get(
                    "questions",
                    []
                )
        }
    }

    call_2_like = {
        "studentOnly":
            lesson_plan.get(
                "studentOnly",
                {}
            ),
        "activities":
            lesson_plan.get(
                "shared",
                {}
            ).get(
                "activities",
                []
            ),
        "realWorldApplications":
            lesson_plan.get(
                "shared",
                {}
            ).get(
                "realWorldApplications",
                []
            ),
        "activityRanking":
            activity_ranking
    }

    return (
        validate_call_1(
            call_1_like
        )
        and validate_call_2(
            call_2_like,
            lesson_plan
        )
    )


# ============================================================
# 10. TEACHING MINUTE ALLOCATION
# ============================================================

def allocate_teaching_minutes(
    weights,
    total_minutes,
    minimum_each=2
):

    count = len(
        weights
    )


    if count == 0:

        return []


    if total_minutes < count:

        raise ValueError(
            "Not enough teaching time to allocate "
            "at least one minute per learning objective."
        )


    if total_minutes < (
        count * minimum_each
    ):

        minimum_each = 1


    safe_weights = [
        max(
            float(
                weight or 0
            ),
            0.1
        )

        for weight in weights
    ]


    total_weight = sum(
        safe_weights
    )


    raw_allocations = [
        (
            weight
            / total_weight
        )
        * total_minutes

        for weight in safe_weights
    ]


    allocations = [
        max(
            minimum_each,
            math.floor(
                raw_value
            )
        )

        for raw_value in raw_allocations
    ]


    while sum(
        allocations
    ) > total_minutes:

        reducible_indices = [
            index

            for index in range(
                count
            )

            if allocations[
                index
            ] > minimum_each
        ]


        if not reducible_indices:

            break


        index_to_reduce = max(

            reducible_indices,

            key=lambda index:
                allocations[index]
                - raw_allocations[index]
        )


        allocations[
            index_to_reduce
        ] -= 1


    remaining = (
        total_minutes
        - sum(
            allocations
        )
    )


    fractional_remainders = [
        raw_allocations[index]
        - math.floor(
            raw_allocations[index]
        )

        for index in range(
            count
        )
    ]


    ranked_indices = sorted(

        range(
            count
        ),

        key=lambda index:
            (
                fractional_remainders[index],
                safe_weights[index]
            ),

        reverse=True
    )


    while remaining > 0:

        for index in ranked_indices:

            if remaining <= 0:

                break


            allocations[
                index
            ] += 1


            remaining -= 1


    return allocations


# ============================================================
# 11. PRESERVE AI TEACHING TIMES
# ============================================================

def get_teaching_allocations(
    objective_weights,
    available_teaching_minutes
):

    rounded_estimates = [
        max(
            1,
            int(
                round(
                    weight
                )
            )
        )

        for weight in objective_weights
    ]


    requested_total = sum(
        rounded_estimates
    )


    if requested_total <= available_teaching_minutes:

        return (
            rounded_estimates,
            requested_total,
            False
        )


    compressed = allocate_teaching_minutes(
        weights=
            objective_weights,

        total_minutes=
            available_teaching_minutes,

        minimum_each=
            2
    )


    return (
        compressed,
        available_teaching_minutes,
        True
    )


# ============================================================
# 12. SELECT BEST ACTIVITY THAT FITS
# ============================================================

def select_activity_that_fits(
    activities,
    activity_ranking,
    available_minutes
):

    if available_minutes < 10:

        return None


    activities_by_id = {
        activity.get("id"):
            activity

        for activity in activities

        if activity.get("id")
    }


    for activity_id in activity_ranking:

        activity = activities_by_id.get(
            activity_id
        )


        if activity is None:

            continue


        duration = activity.get(
            "estimatedMinutes"
        )


        if (
            isinstance(
                duration,
                int
            )
            and 10 <= duration <= 15
            and duration <= available_minutes
        ):

            return activity


    for activity in activities:

        duration = activity.get(
            "estimatedMinutes"
        )


        if (
            isinstance(
                duration,
                int
            )
            and 10 <= duration <= 15
            and duration <= available_minutes
        ):

            return activity


    return None


# ============================================================
# 13. TEACHER SCHEDULE
# ============================================================

def build_teacher_schedule(
    shared_result,
    student_result,
    lesson_length
):

    shared = shared_result.get(
        "shared",
        {}
    )


    objectives = shared.get(
        "learningObjectives",
        []
    )

    activities = shared.get(
        "activities",
        []
    )

    questions = shared.get(
        "questions",
        []
    )

    real_world_applications = shared.get(
        "realWorldApplications",
        []
    )


    if not objectives:

        raise ValueError(
            "No learning objectives available "
            "for teacher schedule."
        )


    intro_minutes = 5

    reflection_minutes = 5


    question_minutes = min(
        max(
            len(
                questions
            ),
            5
        ),
        8
    )


    fixed_minutes = (
        intro_minutes
        + question_minutes
        + reflection_minutes
    )


    maximum_teaching_activity_space = (
        lesson_length
        - fixed_minutes
    )


    if maximum_teaching_activity_space <= 0:

        raise ValueError(
            "Lesson length is too short for "
            "the required lesson structure."
        )


    objective_weights = [
        max(
            float(
                objective.get(
                    "estimatedTeachingMinutes",
                    5
                )
                or 5
            ),
            1
        )

        for objective in objectives
    ]


    desired_teaching_minutes = round(
        sum(
            objective_weights
        )
    )


    minimum_teaching_needed = (
        len(
            objectives
        )
        * 2
    )


    if (
        maximum_teaching_activity_space
        < minimum_teaching_needed
    ):

        reducible_question_time = max(
            question_minutes - 3,
            0
        )


        needed = (
            minimum_teaching_needed
            - maximum_teaching_activity_space
        )


        reduction = min(
            needed,
            reducible_question_time
        )


        question_minutes -= reduction

        fixed_minutes -= reduction

        maximum_teaching_activity_space += reduction


    if (
        maximum_teaching_activity_space
        < minimum_teaching_needed
    ):

        raise ValueError(
            "Lesson is too short to give every "
            "learning objective sufficient teaching time."
        )


    (
        teaching_allocations,
        teaching_budget,
        teaching_was_compressed
    ) = get_teaching_allocations(

        objective_weights=
            objective_weights,

        available_teaching_minutes=
            maximum_teaching_activity_space
    )


    genuine_spare_time = (
        maximum_teaching_activity_space
        - teaching_budget
    )


    activity_ranking = student_result.get(
        "activityRanking",
        []
    )


    selected_activity = select_activity_that_fits(
        activities=
            activities,

        activity_ranking=
            activity_ranking,

        available_minutes=
            genuine_spare_time
    )


    if selected_activity is not None:

        activity_minutes = int(
            selected_activity.get(
                "estimatedMinutes"
            )
        )

    else:

        activity_minutes = 0


    activity_scheduled = (
        selected_activity is not None
    )


    consolidation_minutes = (
        genuine_spare_time
        - activity_minutes
    )


    if consolidation_minutes < 0:

        consolidation_minutes = 0


    if activity_scheduled:

        suggested_if_time_available = []

        activity_recommendation_heading = (
            "Activity Options"
        )

    else:

        suggested_if_time_available = activities

        activity_recommendation_heading = (
            "Suggested Activities If Time Available"
        )


    applications_by_objective = {}


    objective_ids = [
        objective.get(
            "id"
        )

        for objective in objectives

        if objective.get(
            "id"
        )
    ]


    for application in real_world_applications:

        related_objectives = application.get(
            "relatedLearningObjectives",
            []
        )


        target_objective = None


        for objective_id in objective_ids:

            if objective_id in related_objectives:

                target_objective = objective_id

                break


        if target_objective:

            applications_by_objective.setdefault(
                target_objective,
                []
            ).append(
                application
            )


    schedule = []

    current_minute = 0


    # ========================================================
    # INTRODUCTION
    # ========================================================

    schedule.append({

        "startMinute":
            current_minute,

        "endMinute":
            current_minute
            + intro_minutes,

        "durationMinutes":
            intro_minutes,

        "stage":
            "Introduction",

        "teacherDirection":
            (
                "Introduce the lesson topic and learning "
                "objectives. Use brief retrieval questioning "
                "to activate relevant prerequisite knowledge."
            ),

        "studentAction":
            (
                "Respond to retrieval questions and review "
                "the learning objectives."
            ),

        "relatedLearningObjectives":
            objective_ids,

        "relatedSourceTopics":
            []
    })


    current_minute += intro_minutes


    # ========================================================
    # TEACHING PHASES
    # ========================================================

    for index, objective in enumerate(
        objectives
    ):

        duration = teaching_allocations[
            index
        ]


        objective_id = objective.get(
            "id",
            f"LO{index + 1}"
        )


        teacher_focus = objective.get(
            "teacherFocus",
            objective.get(
                "objective",
                ""
            )
        ).strip()


        related_applications = (
            applications_by_objective.get(
                objective_id,
                []
            )
        )


        for application in related_applications:

            title = application.get(
                "title",
                ""
            )

            explanation = application.get(
                "explanation",
                ""
            )

            question = application.get(
                "question",
                ""
            )


            teacher_focus += (
                f" Use the real-world application "
                f"'{title}': {explanation} "
                f"Ask students: {question}"
            )


        schedule.append({

            "startMinute":
                current_minute,

            "endMinute":
                current_minute
                + duration,

            "durationMinutes":
                duration,

            "stage":
                f"Teaching Phase {index + 1}",

            "teacherDirection":
                teacher_focus,

            "studentAction":
                (
                    "Listen, annotate notes, respond to "
                    "checks for understanding and apply "
                    "knowledge to examples."
                ),

            "relatedLearningObjectives":
                [
                    objective_id
                ],

            "relatedSourceTopics":
                objective.get(
                    "relatedSourceTopics",
                    []
                )
        })


        current_minute += duration


    # ========================================================
    # SCHEDULED ACTIVITY
    # ========================================================

    if selected_activity is not None:

        schedule.append({

            "startMinute":
                current_minute,

            "endMinute":
                current_minute
                + activity_minutes,

            "durationMinutes":
                activity_minutes,

            "stage":
                "Activity",

            "teacherDirection":
                (
                    f"Use activity "
                    f"'{selected_activity.get('title', '')}': "
                    f"{selected_activity.get('description', '')}"
                ),

            "studentAction":
                (
                    "Complete the selected lesson activity."
                ),

            "relatedLearningObjectives":
                selected_activity.get(
                    "relatedLearningObjectives",
                    []
                ),

            "relatedSourceTopics":
                selected_activity.get(
                    "relatedSourceTopics",
                    []
                )
        })


        current_minute += activity_minutes


    # ========================================================
    # CONSOLIDATION / REVIEW
    # ========================================================

    if consolidation_minutes > 0:

        schedule.append({

            "startMinute":
                current_minute,

            "endMinute":
                current_minute
                + consolidation_minutes,

            "durationMinutes":
                consolidation_minutes,

            "stage":
                "Consolidation / Review",

            "teacherDirection":
                (
                    "Use the remaining time to revisit key "
                    "points, check understanding, clarify "
                    "difficult concepts or practise a short "
                    "retrieval question."
                ),

            "studentAction":
                (
                    "Review key learning and ask questions "
                    "about any remaining areas of uncertainty."
                ),

            "relatedLearningObjectives":
                objective_ids,

            "relatedSourceTopics":
                sorted({
                    topic_id
                    for objective in objectives
                    for topic_id in objective.get(
                        "relatedSourceTopics",
                        []
                    )
                })
        })


        current_minute += consolidation_minutes


    # ========================================================
    # QUESTION PHASE
    # ========================================================

    question_ids = [
        question.get(
            "id"
        )

        for question in questions

        if question.get(
            "id"
        )
    ]


    question_topics = set()

    question_objectives = set()


    for question in questions:

        question_topics.update(
            question.get(
                "relatedSourceTopics",
                []
            )
        )

        question_objectives.update(
            question.get(
                "relatedLearningObjectives",
                []
            )
        )


    schedule.append({

        "startMinute":
            current_minute,

        "endMinute":
            current_minute
            + question_minutes,

        "durationMinutes":
            question_minutes,

        "stage":
            "Question Phase",

        "teacherDirection":
            (
                "Students complete "
                + ", ".join(
                    question_ids
                )
                + ". Review answers and address "
                "remaining gaps in understanding."
            ),

        "studentAction":
            (
                "Answer the lesson questions and "
                "review responses."
            ),

        "relatedLearningObjectives":
            sorted(
                question_objectives
            ),

        "relatedSourceTopics":
            sorted(
                question_topics
            )
    })


    current_minute += question_minutes


    # ========================================================
    # REFLECTION / REVIEW
    # ========================================================

    all_topics = sorted({
        topic_id

        for objective in objectives

        for topic_id in objective.get(
            "relatedSourceTopics",
            []
        )
    })


    reflection_duration = (
        lesson_length
        - current_minute
    )


    if reflection_duration < 0:

        raise ValueError(
            "Schedule exceeded the lesson length."
        )


    schedule.append({

        "startMinute":
            current_minute,

        "endMinute":
            lesson_length,

        "durationMinutes":
            reflection_duration,

        "stage":
            "Reflection / Review",

        "teacherDirection":
            (
                "Review the learning objectives and identify "
                "areas requiring further review."
            ),

        "studentAction":
            (
                "Review the learning objectives and identify "
                "any topic requiring further review."
            ),

        "relatedLearningObjectives":
            objective_ids,

        "relatedSourceTopics":
            all_topics
    })


    # ========================================================
    # TIMING VALIDATION
    # ========================================================

    if schedule[0][
        "startMinute"
    ] != 0:

        raise ValueError(
            "Schedule does not start at 0."
        )


    if schedule[-1][
        "endMinute"
    ] != lesson_length:

        raise ValueError(
            "Schedule does not end at the "
            "lesson length."
        )


    for index in range(
        len(
            schedule
        )
        - 1
    ):

        current_end = schedule[
            index
        ][
            "endMinute"
        ]

        next_start = schedule[
            index + 1
        ][
            "startMinute"
        ]


        if current_end != next_start:

            raise ValueError(
                "Schedule contains a timing "
                "gap or overlap."
            )


    # ========================================================
    # SOURCE COVERAGE VALIDATION
    # ========================================================

    source_topic_ids = {
        item.get(
            "id"
        )

        for item in shared_result.get(
            "sourceCoverage",
            []
        )

        if item.get(
            "id"
        )
    }


    teaching_topics = set()


    for stage in schedule:

        if stage.get(
            "stage",
            ""
        ).startswith(
            "Teaching Phase"
        ):

            teaching_topics.update(
                stage.get(
                    "relatedSourceTopics",
                    []
                )
            )


    missing_topics = (
        source_topic_ids
        - teaching_topics
    )


    if missing_topics:

        raise ValueError(
            "Teacher schedule is missing source topics: "
            + ", ".join(
                sorted(
                    missing_topics
                )
            )
        )


    # ========================================================
    # DISPLAY SCHEDULER INFORMATION
    # ========================================================

    print(
        "\n----------------------------------------"
    )

    print(
        "PYTHON TEACHER SCHEDULER"
    )

    print(
        "----------------------------------------"
    )


    print(
        f"Introduction: "
        f"{intro_minutes} min"
    )


    print(
        f"AI requested teaching time: "
        f"{desired_teaching_minutes} min"
    )


    print(
        f"Final teaching time: "
        f"{teaching_budget} min"
    )


    print(
        f"Teaching compressed: "
        f"{teaching_was_compressed}"
    )


    print(
        f"Question phase: "
        f"{question_minutes} min"
    )


    print(
        f"Reflection target: "
        f"{reflection_minutes} min"
    )


    print(
        f"Genuine spare time after teaching: "
        f"{genuine_spare_time} min"
    )


    if selected_activity is not None:

        print(
            f"Selected activity: "
            f"{selected_activity.get('id')} - "
            f"{selected_activity.get('title')} "
            f"({activity_minutes} min)"
        )

    else:

        print(
            "Selected activity: None"
        )

        print(
            "Suggested Activities If Time Available:"
        )


        for activity in activities:

            print(
                f"  {activity.get('id')} - "
                f"{activity.get('title')} "
                f"({activity.get('estimatedMinutes')} min)"
            )


    if consolidation_minutes > 0:

        print(
            f"Consolidation / review: "
            f"{consolidation_minutes} min"
        )


    print(
        "\nAll activity options:"
    )


    for activity in activities:

        print(
            f"  {activity.get('id')} - "
            f"{activity.get('title')} "
            f"({activity.get('estimatedMinutes')} min)"
        )


    print(
        "\nTeaching phase allocation:"
    )


    for index, objective in enumerate(
        objectives
    ):

        print(
            f"  "
            f"{objective.get('id')}: "
            f"{teaching_allocations[index]} min "
            f"(AI estimate: "
            f"{objective.get('estimatedTeachingMinutes')} min)"
        )


    print(
        f"\nSchedule stages: "
        f"{len(schedule)}"
    )


    print(
        f"Schedule: "
        f"0-{lesson_length} minutes"
    )


    print(
        "All source topics explicitly taught."
    )


    print(
        "----------------------------------------"
    )


    return {

        "lessonSchedule":
            schedule,

        "activityOptions":
            activities,

        "activityScheduled":
            activity_scheduled,

        "selectedActivity":
            selected_activity,

        "scheduledActivityMinutes":
            activity_minutes,

        "activityRecommendationHeading":
            activity_recommendation_heading,

        "suggestedActivitiesIfTimeAvailable":
            suggested_if_time_available,

        "aiEstimatedTeachingMinutes":
            desired_teaching_minutes,

        "finalTeachingMinutes":
            teaching_budget,

        "teachingWasCompressed":
            teaching_was_compressed,

        "genuineSpareMinutes":
            genuine_spare_time,

        "consolidationMinutes":
            consolidation_minutes,

        "teachingTimeByObjective": [
            {
                "learningObjectiveId":
                    objective.get(
                        "id"
                    ),

                "aiEstimatedMinutes":
                    objective.get(
                        "estimatedTeachingMinutes"
                    ),

                "scheduledMinutes":
                    teaching_allocations[
                        index
                    ]
            }

            for index, objective in enumerate(
                objectives
            )
        ]
    }


# ============================================================
# 14. GENERATE COMPLETE LESSON
# ============================================================

def generate_lesson_plan(
    chapter_title,
    level=None,
    lesson_length=60
):
    """
    Three-Groq-call architecture:

    CALL 1:
        shared plan
        - metadata
        - sourceCoverage
        - learning objectives
        - prerequisites
        - key terms
        - questions + answers

    CALL 2:
        student notes + enrichment
        - core notes with summaries
        - activities
        - real-world applications
        - activity ranking

    CALL 3:
        factual/source verification

    Key Points are derived in Python from core-note summaries.
    """

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file."
        )

    if not chapter_title or not str(
        chapter_title
    ).strip():
        raise ValueError(
            "A lesson topic/chapter title is required."
        )

    topic = str(
        chapter_title
    ).strip()

    print(
        "Retrieving relevant textbook material with RAG..."
    )

    (
        context_content,
        selected_start_page,
        selected_end_page,
        centre_page
    ) = build_rag_context(
        topic
    )

    if not context_content.strip():
        raise ValueError(
            "No readable RAG context was produced "
            "for the requested topic."
        )

    # ========================================================
    # CALL 1 - SHARED LESSON PLAN
    # ========================================================

    maximum_call_1_attempts = 3
    call_1_result = None

    for attempt in range(
        1,
        maximum_call_1_attempts + 1
    ):
        print(
            f"\nBuilding CALL 1 prompt "
            f"(attempt {attempt}/{maximum_call_1_attempts})..."
        )

        prompt = build_shared_plan_prompt(
            context_content,
            topic,
            level,
            lesson_length
        )

        print(
            f"Call 1 prompt characters: {len(prompt):,}"
        )

        print(
            "Sending CALL 1 - SHARED LESSON PLAN..."
        )

        candidate = call_groq(
            prompt,
            max_tokens=3600
        )

        if candidate is not None:
            candidate = normalise_call_1_result(
                candidate
            )

            if level is not None:
                candidate.setdefault(
                    "lessonMetadata",
                    {}
                )[
                    "level"
                ] = str(
                    level
                ).strip()

        if candidate is None:
            print(
                "\nCall 1 API request failed."
            )

        elif validate_call_1(
            candidate
        ):
            call_1_result = candidate
            break

        else:
            print(
                "\nCall 1 output failed validation."
            )

        if attempt < maximum_call_1_attempts:
            print(
                "\nWaiting 60 seconds before "
                "retrying CALL 1..."
            )
            time.sleep(
                60
            )

    if call_1_result is None:
        print(
            "\nGeneration stopped: "
            "Call 1 failed after all attempts."
        )
        return None

    # ========================================================
    # TARGETED RAG FOR CALL 2
    # ========================================================

    targeted_context = build_targeted_rag_context(
        call_1_result=call_1_result,
        centre_page=centre_page,
        k_per_query=TARGETED_RAG_K
    )

    print(
        f"CALL 1 context characters: "
        f"{len(context_content):,}"
    )
    print(
        f"CALL 2 targeted context characters: "
        f"{len(targeted_context):,}"
    )

    # ========================================================
    # CALL 2 - NOTES + ENRICHMENT
    # ========================================================

    print(
        "\nWaiting before CALL 2..."
    )
    time.sleep(
        60
    )

    maximum_call_2_attempts = 3
    call_2_result = None

    for attempt in range(
        1,
        maximum_call_2_attempts + 1
    ):
        print(
            f"\nBuilding CALL 2 prompt "
            f"(attempt {attempt}/{maximum_call_2_attempts})..."
        )

        prompt = build_content_prompt(
            targeted_context,
            call_1_result
        )

        print(
            f"Call 2 prompt characters: {len(prompt):,}"
        )

        print(
            "Sending CALL 2 - NOTES + ENRICHMENT..."
        )

        candidate = call_groq(
            prompt,
            max_tokens=3800
        )

        if candidate is None:
            print(
                "\nCall 2 API request failed."
            )

        elif validate_call_2(
            candidate,
            call_1_result
        ):
            call_2_result = candidate
            break

        else:
            print(
                "\nCall 2 output failed validation."
            )

        if attempt < maximum_call_2_attempts:
            print(
                "\nWaiting 60 seconds before "
                "retrying CALL 2..."
            )
            time.sleep(
                60
            )

    if call_2_result is None:
        print(
            "\nGeneration stopped: "
            "Call 2 failed after all attempts."
        )
        return None

    # ========================================================
    # MERGE CALL 1 + CALL 2
    # ========================================================

    shared_result = {
        "lessonMetadata":
            call_1_result.get(
                "lessonMetadata",
                {}
            ),

        "sourceCoverage":
            call_1_result.get(
                "sourceCoverage",
                []
            ),

        "shared":
            call_1_result.get(
                "shared",
                {}
            )
    }

    shared_result[
        "shared"
    ][
        "activities"
    ] = call_2_result.get(
        "activities",
        []
    )

    shared_result[
        "shared"
    ][
        "realWorldApplications"
    ] = call_2_result.get(
        "realWorldApplications",
        []
    )

    student_result = {
        "studentOnly":
            call_2_result.get(
                "studentOnly",
                {}
            ),

        "activityRanking":
            call_2_result.get(
                "activityRanking",
                []
            )
    }

    # Key Points are derived from Core Notes instead of generated
    # separately by another Groq call.
    core_notes = student_result.get(
        "studentOnly",
        {}
    ).get(
        "coreNotes",
        []
    )

    shared_result[
        "shared"
    ][
        "keyPoints"
    ] = derive_key_points_from_core_notes(
        core_notes
    )

    print(
        f"\nDerived "
        f"{len(shared_result['shared']['keyPoints'])} "
        "key points from core-note summaries."
    )

    # ========================================================
    # CALL 3 - FINAL FACTUAL VERIFICATION
    # ========================================================

    verification_context = build_verification_rag_context(
        call_1_result=call_1_result,
        centre_page=centre_page,
        k_per_query=VERIFICATION_RAG_K
    )

    verification_target = build_compact_verification_target(
        shared_result=shared_result,
        student_result=student_result
    )

    print(
        "\nWaiting before CALL 3..."
    )
    time.sleep(
        60
    )

    print(
        "\nBuilding CALL 3 prompt - FINAL FACTUAL VERIFICATION..."
    )

    verification_prompt = build_verification_prompt(
        verification_context,
        verification_target
    )

    print(
        f"Call 3 prompt characters: "
        f"{len(verification_prompt):,}"
    )

    print(
        "Sending CALL 3 - FINAL FACTUAL VERIFICATION..."
    )

    verification_result = call_groq(
        verification_prompt,
        max_tokens=3500,
        max_retries=2,
        temperature=0,
        force_json=False
    )

    if verification_result is None:
        print(
            "\nGeneration stopped: "
            "Call 3 factual verification failed."
        )
        return None

    verification_issues = verification_result.get(
        "issues",
        []
    )

    print("\n----------------------------------------")
    print("CALL 3 VERIFICATION RESULT")
    print("----------------------------------------")
    print(
        f"Verifier marked lesson valid: "
        f"{bool(verification_result.get('valid', False))}"
    )
    print(
        f"Issues reported: {len(verification_issues)}"
    )

    if verification_issues:
        full_patch_target = {
            "lessonMetadata":
                shared_result.get(
                    "lessonMetadata",
                    {}
                ),
            "sourceCoverage":
                shared_result.get(
                    "sourceCoverage",
                    []
                ),
            "shared":
                shared_result.get(
                    "shared",
                    {}
                ),
            "studentOnly":
                student_result.get(
                    "studentOnly",
                    {}
                )
        }

        for index, issue in enumerate(
            verification_issues,
            start=1
        ):
            print(
                f"{index}. {issue.get('path', '')}"
            )
            print(
                f"   {issue.get('problem', '')}"
            )

        applied_count = apply_verification_patches(
            full_patch_target,
            verification_result
        )

        print(
            f"Corrections applied: {applied_count}"
        )

        shared_result[
            "lessonMetadata"
        ] = full_patch_target.get(
            "lessonMetadata",
            {}
        )

        shared_result[
            "sourceCoverage"
        ] = full_patch_target.get(
            "sourceCoverage",
            []
        )

        shared_result[
            "shared"
        ] = full_patch_target.get(
            "shared",
            {}
        )

        student_result[
            "studentOnly"
        ] = full_patch_target.get(
            "studentOnly",
            {}
        )

    else:
        print(
            "No factual/source-grounding corrections required."
        )

    print("----------------------------------------")

    # If CALL 3 changed a core-note summary, regenerate Key Points
    # so both PDF sections stay perfectly consistent.
    corrected_core_notes = student_result.get(
        "studentOnly",
        {}
    ).get(
        "coreNotes",
        []
    )

    shared_result[
        "shared"
    ][
        "keyPoints"
    ] = derive_key_points_from_core_notes(
        corrected_core_notes
    )

    if level is not None:
        shared_result.setdefault(
            "lessonMetadata",
            {}
        )[
            "level"
        ] = str(
            level
        ).strip()

    full_verified_lesson = {
        "lessonMetadata":
            shared_result.get(
                "lessonMetadata",
                {}
            ),
        "sourceCoverage":
            shared_result.get(
                "sourceCoverage",
                []
            ),
        "shared":
            shared_result.get(
                "shared",
                {}
            ),
        "studentOnly":
            student_result.get(
                "studentOnly",
                {}
            )
    }

    if not validate_after_verification(
        full_verified_lesson,
        student_result.get(
            "activityRanking",
            []
        )
    ):
        print(
            "\nGeneration stopped: corrected lesson failed "
            "post-verification structural validation."
        )
        return None

    # ========================================================
    # PYTHON TEACHER SCHEDULER
    # ========================================================

    print(
        "\nBuilding teacher schedule in Python..."
    )

    teacher_only = build_teacher_schedule(
        shared_result=shared_result,
        student_result=student_result,
        lesson_length=lesson_length
    )

    # ========================================================
    # FINAL MERGE
    # ========================================================

    lesson_plan = {
        "lessonMetadata":
            shared_result.get(
                "lessonMetadata",
                {}
            ),

        "sourceCoverage":
            shared_result.get(
                "sourceCoverage",
                []
            ),

        "shared":
            shared_result.get(
                "shared",
                {}
            ),

        "studentOnly":
            student_result.get(
                "studentOnly",
                {}
            ),

        "teacherOnly":
            teacher_only
    }

    lesson_plan.setdefault(
        "lessonMetadata",
        {}
    )[
        "sourcePageRange"
    ] = {
        "startPage": selected_start_page,
        "endPage": selected_end_page
    }

    print("\n========================================")
    print("LESSON PLAN GENERATED SUCCESSFULLY")
    print("3 GROQ CALLS + PYTHON SCHEDULER")
    print("========================================")

    return lesson_plan


# ============================================================
# 15. SAVE LESSON PLAN
# ============================================================

def save_lesson_plan(
    lesson_plan,
    output_path="generated_lesson_plan.json"
):

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            lesson_plan,
            file,
            indent=4,
            ensure_ascii=False
        )


    print(
        f"\nLesson plan saved to: "
        f"{output_path}"
    )


# ============================================================
# 16. MAIN
# ============================================================

if __name__ == "__main__":

    # Enter the lesson topic you want to retrieve from the
    # textbook already stored in Chroma.
    #Types of Chemical Reactions, Acids, Bases and Salts, Carbon and its Compounds,Life Processes
    CHAPTER_TITLE = "Carbon and its Compounds"

    # Leave as None if you want the model to infer the level
    # from the retrieved textbook material.
    LEVEL = "Class X"

    LESSON_LENGTH = 60

    result = generate_lesson_plan(
        chapter_title=CHAPTER_TITLE,
        level=LEVEL,
        lesson_length=LESSON_LENGTH
    )

    if result is not None:

        print(
            "\n========================================"
        )
        print(
            "FINAL GENERATED LESSON PLAN"
        )
        print(
            "========================================\n"
        )

        print(
            json.dumps(
                result,
                indent=4,
                ensure_ascii=False
            )
        )

        save_lesson_plan(
            result
        )

        pdf_files = create_lesson_pdfs(
            result
        )

        print(
            "\nTeacher PDF:",
            pdf_files["teacherPdf"]
        )

        print(
            "Student PDF:",
            pdf_files["studentPdf"]
        )

        print(
            f"\nSource pages used: "
            f"{result['lessonMetadata']['sourcePageRange']['startPage']}"
            f"-"
            f"{result['lessonMetadata']['sourcePageRange']['endPage']}"
        )


