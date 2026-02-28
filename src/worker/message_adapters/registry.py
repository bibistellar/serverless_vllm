from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .models import (
    build_default_vllm_inputs,
    build_qwen_vl_vllm_inputs,
    convert_default_messages,
    convert_qwen_vl_for_llama_cpp,
    convert_qwen_vl_for_vllm,
)

BACKEND_VLLM = "vllm"
BACKEND_LLAMA_CPP = "llama.cpp"
SUPPORTED_BACKENDS = {BACKEND_VLLM, BACKEND_LLAMA_CPP}

AdapterFn = Callable[[Any], List[Dict[str, Any]]]
VLLMInputBuilderFn = Callable[..., Any]


@dataclass(frozen=True)
class ModelAdapterEntry:
    name: str
    matcher: Callable[[str], bool]
    backend_adapters: Dict[str, Callable[..., List[Dict[str, Any]]]]
    vllm_input_builder: Optional[VLLMInputBuilderFn] = None



def _is_qwen_vl(model_name: str) -> bool:
    normalized = model_name.lower().replace("_", "-")
    return "qwen" in normalized and "vl" in normalized


_MODEL_ADAPTERS: List[ModelAdapterEntry] = [
    ModelAdapterEntry(
        name="qwen_vl",
        matcher=_is_qwen_vl,
        backend_adapters={
            BACKEND_VLLM: convert_qwen_vl_for_vllm,
            BACKEND_LLAMA_CPP: convert_qwen_vl_for_llama_cpp,
        },
        vllm_input_builder=build_qwen_vl_vllm_inputs,
    ),
]



def convert_messages_for_backend(
    messages: Any,
    backend_type: str,
    *,
    base_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if backend_type not in SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported backend_type: {backend_type}")

    model_name = (base_model or "").strip()
    for entry in _MODEL_ADAPTERS:
        if not entry.matcher(model_name):
            continue
        adapter = entry.backend_adapters.get(backend_type)
        if adapter:
            return adapter(messages, base_model=base_model)
        break

    return convert_default_messages(messages, base_model=base_model)


def build_vllm_generation_inputs(
    messages: Any,
    *,
    processor: Any,
    text_prompt_cls: Any,
    base_model: Optional[str] = None,
) -> Any:
    model_name = (base_model or "").strip()
    for entry in _MODEL_ADAPTERS:
        if not entry.matcher(model_name):
            continue
        if entry.vllm_input_builder:
            return entry.vllm_input_builder(
                messages,
                processor=processor,
                text_prompt_cls=text_prompt_cls,
                base_model=base_model,
            )
        break

    return build_default_vllm_inputs(
        messages,
        processor=processor,
        text_prompt_cls=text_prompt_cls,
        base_model=base_model,
    )
