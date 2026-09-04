"""Model metadata used by the cross-family screening harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    family: str
    size_billions: float
    access: str = "public"
    default_4bit: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


MODEL_REGISTRY = {
    spec.key: spec
    for spec in (
        ModelSpec("qwen_1_5b", "Qwen/Qwen2.5-1.5B-Instruct", "qwen2.5", 1.5),
        ModelSpec("qwen_7b", "Qwen/Qwen2.5-7B-Instruct", "qwen2.5", 7.0),
        ModelSpec("smollm2_360m", "HuggingFaceTB/SmolLM2-360M-Instruct", "smollm2", 0.36, default_4bit=False),
        ModelSpec("smollm2_1_7b", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "smollm2", 1.7, default_4bit=False),
        ModelSpec("gemma3_1b", "google/gemma-3-1b-it", "gemma3", 1.0, access="terms"),
        ModelSpec("gemma3_4b", "google/gemma-3-4b-it", "gemma3", 4.0, access="terms"),
        ModelSpec("llama3_2_1b", "meta-llama/Llama-3.2-1B-Instruct", "llama3", 1.0, access="approval"),
        ModelSpec("llama3_1_8b", "meta-llama/Llama-3.1-8B-Instruct", "llama3", 8.0, access="approval"),
    )
}


def get_model_spec(key: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[key]
    except KeyError as error:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model key {key!r}. Available: {choices}") from error

