"""Task-neutral input adapters and deterministic answer scorers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .scoring import extract_final_number, gsm8k_correct


_FINAL_CHOICE = re.compile(r"(?:final answer|answer)\s*[:：]?\s*\(?([A-Z])\)?", re.I)
_STANDALONE_CHOICE = re.compile(r"\b([A-Z])\b")


@dataclass(frozen=True)
class TaskExample:
    id: str
    prompt: str
    reference: Any
    task_type: str
    split: str = "screening"
    metadata: dict[str, Any] = field(default_factory=dict)


def _format_choices(question: str, choices: list[str]) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    rendered = "\n".join(f"{labels[index]}. {choice}" for index, choice in enumerate(choices))
    return f"{question.rstrip()}\n\n{rendered}"


def adapt_row(row: dict[str, Any], default_task_type: str = "numeric") -> TaskExample:
    """Accept the new schema plus legacy math and common MMLU-style rows."""
    task_type = str(row.get("task_type", default_task_type))
    prompt = row.get("prompt", row.get("question"))
    reference = row.get("reference", row.get("answer"))
    metadata = dict(row.get("task_metadata", row.get("metadata", {})))
    choices = row.get("choices")
    if choices is not None:
        prompt = _format_choices(str(prompt), list(choices))
        metadata["choices"] = list(choices)
        task_type = "multiple_choice"
    if prompt is None or reference is None:
        raise ValueError("Each row requires prompt/question and reference/answer")
    return TaskExample(
        id=str(row.get("id", "")),
        prompt=str(prompt),
        reference=reference,
        task_type=task_type,
        split=str(row.get("split", "screening")),
        metadata=metadata,
    )


def system_prompt(task_type: str, prompt_mode: str = "task") -> str:
    if task_type == "numeric" and prompt_mode == "micro_reasoning":
        return (
            "Solve the problem using exactly one compact calculation line with no prose. "
            "Then write a second line exactly as 'Final answer: <number>'. "
            "Keep the entire response under 48 tokens."
        )
    if task_type == "numeric" and prompt_mode == "answer_only":
        return (
            "Solve the problem independently, but return only one line in the exact form "
            "'Final answer: <number>'. Do not include reasoning or explanation."
        )
    if task_type == "multiple_choice" and prompt_mode == "answer_only":
        return (
            "Choose the correct option internally, but return only one line in the exact form "
            "'Final answer: <letter>'. Do not include reasoning or explanation."
        )
    if prompt_mode != "task":
        raise ValueError(f"Prompt mode {prompt_mode!r} is only supported for numeric tasks")
    prompts = {
        "numeric": (
            "Solve the math problem. Keep the reasoning to at most three short lines. "
            "Your final line must be exactly 'Final answer: <number>'."
        ),
        "multiple_choice": "Choose the correct option. End with exactly 'Final answer: <letter>'.",
        "exact_match": "Answer concisely. End with exactly 'Final answer: <answer>'.",
        "code": "Return only the requested Python code without Markdown fences.",
        "instruction_following": "Follow every instruction in the user request exactly.",
    }
    try:
        return prompts[task_type]
    except KeyError as error:
        raise ValueError(f"Unsupported task type: {task_type}") from error


def extract_choice(text: str, choice_count: int = 26) -> str | None:
    upper = text.upper()
    matches = _FINAL_CHOICE.findall(upper) or _STANDALONE_CHOICE.findall(upper)
    valid = [value for value in matches if ord(value) - ord("A") < choice_count]
    return valid[-1] if valid else None


def extract_exact(text: str) -> str:
    match = re.search(r"final answer\s*:\s*(.+)", text, re.I)
    value = match.group(1) if match else text
    return " ".join(value.strip().casefold().split())


def score_prediction(prediction: str, example: TaskExample) -> tuple[Any, bool | None]:
    if example.task_type == "numeric":
        parsed = extract_final_number(prediction)
        return parsed, gsm8k_correct(prediction, str(example.reference))
    if example.task_type == "multiple_choice":
        choices = example.metadata.get("choices", [])
        parsed = extract_choice(prediction, len(choices) or 26)
        reference = example.reference
        if isinstance(reference, int):
            reference = chr(ord("A") + reference)
        return parsed, parsed == str(reference).strip().upper()
    if example.task_type == "exact_match":
        parsed = extract_exact(prediction)
        return parsed, parsed == extract_exact(str(example.reference))
    if example.task_type in {"code", "instruction_following"}:
        return prediction.strip(), None
    raise ValueError(f"Unsupported task type: {example.task_type}")
