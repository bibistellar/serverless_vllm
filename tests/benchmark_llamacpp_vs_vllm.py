#!/usr/bin/env python3
"""Standalone benchmark: llama.cpp vs vLLM AsyncLLMEngine.

Metrics:
1) cold_start_s: model load time
2) ttft_s: time to first generated token
3) token_rate_tps: generated token rate after first token
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import gc
import json
import multiprocessing as mp
import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class BenchmarkResult:
    backend: str
    cold_start_s: Optional[float] = None
    ttft_s: Optional[float] = None
    output_tokens: Optional[int] = None
    token_rate_tps: Optional[float] = None
    gpu_before_mb: Optional[float] = None
    gpu_after_load_mb: Optional[float] = None
    gpu_peak_mb: Optional[float] = None
    gpu_after_cleanup_mb: Optional[float] = None
    output_preview: str = ""
    gpu_note: Optional[str] = None
    error: Optional[str] = None


_IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".avif": "image/avif",
    ".jp2": "image/jp2",
    ".j2k": "image/jp2",
    ".jpx": "image/jp2",
    ".bmp": "image/bmp",
    ".ico": "image/x-icon",
    ".pcx": "image/x-pcx",
    ".tga": "image/x-tga",
    ".icns": "image/icns",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".eps": "application/postscript",
    ".dds": "image/vnd-ms.dds",
    ".dib": "image/dib",
    ".sgi": "image/sgi",
    ".pbm": "image/x-portable-bitmap",
    ".pgm": "image/x-portable-graymap",
    ".ppm": "image/x-portable-pixmap",
    ".xbm": "image/x-xbitmap",
    ".mpo": "image/mpo",
    ".msp": "image/msp",
    ".im": "image/x-pillow-im",
    ".qoi": "image/qoi",
}

DEFAULT_SINGLE_IMAGE_PATH = "/root/Code/video_anomaly_analysis_system/model_pool_serve/demo.png"


class GPUMemoryTracker:
    def __init__(self, gpu_index: int, sample_interval_s: float) -> None:
        self.gpu_index = gpu_index
        self.sample_interval_s = max(sample_interval_s, 0.05)
        self.available = False
        self.error: Optional[str] = None
        self.before_mb: Optional[float] = None
        self.peak_mb: Optional[float] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pynvml = None
        self._handle = None

        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.available = True
        except Exception as exc:
            self.error = f"NVML unavailable: {exc}"

    def _read_used_mb(self) -> Optional[float]:
        if not self.available or self._pynvml is None or self._handle is None:
            return None
        try:
            info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return info.used / (1024.0 * 1024.0)
        except Exception:
            return None

    def snapshot(self) -> Optional[float]:
        used_mb = self._read_used_mb()
        if used_mb is not None:
            if self.peak_mb is None or used_mb > self.peak_mb:
                self.peak_mb = used_mb
        return used_mb

    def start(self) -> None:
        self.before_mb = self.snapshot()
        if not self.available:
            return

        def _loop() -> None:
            while not self._stop_event.wait(self.sample_interval_s):
                self.snapshot()

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=1.0)
        self.snapshot()


def patch_llamacpp_destructor_bug() -> None:
    """Work around llama_cpp destructor bug in some versions.

    Symptom:
    AttributeError: 'LlamaModel' object has no attribute 'sampler'
    """
    try:
        from llama_cpp import _internals
    except Exception:
        return

    llama_model_cls = getattr(_internals, "LlamaModel", None)
    if llama_model_cls is None:
        return
    old_del = getattr(llama_model_cls, "__del__", None)
    if old_del is None:
        return
    if getattr(old_del, "__name__", "") == "_safe_llama_model_del":
        return

    def _safe_llama_model_del(self: Any) -> None:
        try:
            old_del(self)
        except AttributeError as exc:
            # Ignore known destructor issue from partial initialization.
            if "sampler" not in str(exc):
                raise

    setattr(llama_model_cls, "__del__", _safe_llama_model_del)


def image_to_base64_data_uri(file_path: str, fallback_mime: str = "application/octet-stream") -> str:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    extension = os.path.splitext(file_path)[1].lower()
    mime_type = _IMAGE_MIME_TYPES.get(extension, fallback_mime)
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def pick_single_image_source(args: argparse.Namespace) -> Optional[str]:
    if args.image_path:
        return args.image_path
    if args.image_url:
        return args.image_url
    return None


def configure_vllm_spawn() -> None:
    """Force vLLM worker multiprocessing to use spawn.

    This avoids CUDA re-init failures after another backend (e.g. llama.cpp)
    has already initialized GPU runtime in the parent process.
    """
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method may already be set in some environments.
        pass


def build_llamacpp_user_content(args: argparse.Namespace) -> Any:
    image_source = pick_single_image_source(args)
    if not image_source:
        return args.prompt

    content: List[Dict[str, Any]] = []
    if image_source.startswith("http://") or image_source.startswith("https://"):
        content.append({"type": "image_url", "image_url": {"url": image_source}})
    else:
        content.append({"type": "image_url", "image_url": {"url": image_to_base64_data_uri(image_source)}})
    content.append({"type": "text", "text": args.prompt})
    return content


def build_vllm_messages(args: argparse.Namespace) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})

    image_source = pick_single_image_source(args)
    if not image_source:
        messages.append({"role": "user", "content": args.prompt})
        return messages

    user_content: List[Dict[str, Any]] = []
    user_content.append({"type": "image", "image": image_source})
    user_content.append({"type": "text", "text": args.prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def build_token_counter(model_id: str) -> Callable[[str], int]:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        def _count(text: str) -> int:
            if not text:
                return 0
            return len(tokenizer.encode(text, add_special_tokens=False))

        return _count
    except Exception as exc:
        print(f"[WARN] tokenizer load failed ({exc}), fallback to whitespace tokens.")

        def _count(text: str) -> int:
            return len(text.split()) if text else 0

        return _count


def extract_delta_text(choice: Dict[str, Any]) -> str:
    delta = choice.get("delta", {})
    content = delta.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str):
                    chunks.append(txt)
        return "".join(chunks)
    return ""


def stream_print(enabled: bool, backend: str, text: str) -> None:
    if not enabled or not text:
        return
    print(text, end="", flush=True)


def benchmark_llamacpp(args: argparse.Namespace, count_tokens: Callable[[str], int]) -> BenchmarkResult:
    result = BenchmarkResult(backend="llama.cpp")
    llm: Optional[Any] = None
    gpu_tracker = GPUMemoryTracker(args.gpu_index, args.gpu_mem_sample_interval)
    gpu_tracker.start()
    result.gpu_before_mb = gpu_tracker.before_mb
    if gpu_tracker.error:
        result.gpu_note = gpu_tracker.error
    try:
        patch_llamacpp_destructor_bug()
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen3VLChatHandler
        from huggingface_hub import hf_hub_download

        if not args.llama_repo_id:
            result.error = "missing --llama-repo-id"
            return result
        if not args.llama_filename:
            result.error = "missing --llama-filename"
            return result

        mmproj_path = args.llama_mmproj_path
        if not mmproj_path:
            if not args.llama_mmproj_filename:
                result.error = "missing --llama-mmproj-filename (or --llama-mmproj-path)"
                return result
            mmproj_path = hf_hub_download(
                repo_id=args.llama_repo_id,
                filename=args.llama_mmproj_filename,
                cache_dir=args.llama_cache_dir,
                local_dir=args.llama_local_dir,
            )

        init_kwargs: Dict[str, Any] = {
            "repo_id": args.llama_repo_id,
            "filename": args.llama_filename,
            "chat_handler": Qwen3VLChatHandler(
                clip_model_path=mmproj_path,
                force_reasoning=bool(args.llama_force_reasoning),
                image_min_tokens=args.llama_image_min_tokens,
            ),
            "n_gpu_layers": args.llama_n_gpu_layers,
            "n_ctx": args.max_model_len,
            "verbose": False,
        }
        if args.llama_cache_dir:
            init_kwargs["cache_dir"] = args.llama_cache_dir
        if args.llama_local_dir:
            init_kwargs["local_dir"] = args.llama_local_dir
            init_kwargs["local_dir_use_symlinks"] = "auto"
        if args.llama_mmproj_filename:
            init_kwargs["additional_files"] = [args.llama_mmproj_filename]
        if args.llama_swa_full:
            init_kwargs["swa_full"] = True

        t0 = time.perf_counter()
        llm = Llama.from_pretrained(**init_kwargs)
        result.cold_start_s = time.perf_counter() - t0
        result.gpu_after_load_mb = gpu_tracker.snapshot()

        req_start = time.perf_counter()
        first_token_at: Optional[float] = None
        pieces: List[str] = []
        printed_any = False

        messages: List[Dict[str, Any]] = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": build_llamacpp_user_content(args)})

        stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stream=True,
        )
        if args.print_stream:
            print("\n[llama.cpp stream]")
        for chunk in stream:
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = extract_delta_text(choices[0])
            if not delta:
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter()
            pieces.append(delta)
            stream_print(bool(args.print_stream), "llama.cpp", delta)
            printed_any = True
        if args.print_stream and printed_any:
            print()

        end_at = time.perf_counter()
        generated = "".join(pieces)
        result.output_preview = generated[:160]
        result.output_tokens = count_tokens(generated)

        if first_token_at is None:
            result.error = "no streamed token received from llama.cpp"
            return result

        result.ttft_s = first_token_at - req_start
        gen_duration = max(end_at - first_token_at, 1e-9)
        result.token_rate_tps = (result.output_tokens or 0) / gen_duration
        return result
    except Exception as exc:
        result.error = str(exc)
        return result
    finally:
        if llm is not None and hasattr(llm, "close"):
            try:
                llm.close()
            except Exception:
                pass
        del llm
        gc.collect()
        result.gpu_after_cleanup_mb = gpu_tracker.snapshot()
        gpu_tracker.stop()
        result.gpu_peak_mb = gpu_tracker.peak_mb


async def _benchmark_vllm_async_engine(
    args: argparse.Namespace,
    count_tokens: Callable[[str], int],
) -> BenchmarkResult:
    result = BenchmarkResult(backend="vLLM(AsyncLLMEngine)")
    engine = None
    request_id = f"bench-{int(time.time() * 1000)}"
    gpu_tracker = GPUMemoryTracker(args.gpu_index, args.gpu_mem_sample_interval)
    gpu_tracker.start()
    result.gpu_before_mb = gpu_tracker.before_mb
    if gpu_tracker.error:
        result.gpu_note = gpu_tracker.error
    try:
        configure_vllm_spawn()
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.inputs.data import TextPrompt
        from vllm.sampling_params import SamplingParams

        model_ref = args.vllm_model_path or args.vllm_model
        tp_size = args.tensor_parallel_size
        if tp_size is None:
            tp_size = torch.cuda.device_count() or 1

        t0 = time.perf_counter()
        processor = AutoProcessor.from_pretrained(model_ref, trust_remote_code=True)
        engine_args = AsyncEngineArgs(
            model=model_ref,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            dtype=args.vllm_dtype,
            mm_encoder_tp_mode="data",
            enable_expert_parallel=False,
            seed=0,
        )
        engine = await asyncio.to_thread(AsyncLLMEngine.from_engine_args, engine_args)
        result.cold_start_s = time.perf_counter() - t0
        result.gpu_after_load_mb = gpu_tracker.snapshot()

        messages = build_vllm_messages(args)

        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        mm_data: Dict[str, Any] = {}
        video_kwargs: Dict[str, Any] = {}
        if pick_single_image_source(args):
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=processor.image_processor.patch_size,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs

        if mm_data:
            inputs: Any = TextPrompt(
                prompt=prompt,
                multi_modal_data=mm_data,
                mm_processor_kwargs=video_kwargs,
            )
        else:
            inputs = prompt

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )

        req_start = time.perf_counter()
        first_token_at: Optional[float] = None
        pieces: List[str] = []
        prev_text = ""
        printed_any = False
        if args.print_stream:
            print("\n[vLLM stream]")

        async for output in engine.generate(inputs, sampling_params, request_id):
            for completion in output.outputs:
                current = completion.text or ""
                delta = current[len(prev_text) :] if current.startswith(prev_text) else current
                if delta:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    pieces.append(delta)
                    stream_print(bool(args.print_stream), "vLLM", delta)
                    printed_any = True
                prev_text = current
        if args.print_stream and printed_any:
            print()

        end_at = time.perf_counter()
        generated = "".join(pieces)
        result.output_preview = generated[:160]
        result.output_tokens = count_tokens(generated)

        if first_token_at is None:
            result.error = "no streamed token received from vLLM AsyncLLMEngine"
            return result

        result.ttft_s = first_token_at - req_start
        gen_duration = max(end_at - first_token_at, 1e-9)
        result.token_rate_tps = (result.output_tokens or 0) / gen_duration
        return result
    except Exception as exc:
        msg = str(exc)
        if "Cannot re-initialize CUDA in forked subprocess" in msg:
            msg += (
                " | hint: vLLM needs spawn multiprocessing; "
                "this script now sets VLLM_WORKER_MULTIPROC_METHOD=spawn automatically."
            )
        result.error = msg
        return result
    finally:
        if engine is not None:
            try:
                await engine.abort(request_id)
            except Exception:
                pass
        del engine
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        result.gpu_after_cleanup_mb = gpu_tracker.snapshot()
        gpu_tracker.stop()
        result.gpu_peak_mb = gpu_tracker.peak_mb


def benchmark_vllm_async_engine(
    args: argparse.Namespace,
    count_tokens: Callable[[str], int],
) -> BenchmarkResult:
    return asyncio.run(_benchmark_vllm_async_engine(args, count_tokens))


def fmt(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.3f}"


def print_report(results: List[BenchmarkResult]) -> None:
    print("\n=== Benchmark Report ===")
    print(
        f"{'backend':<24} {'cold_start_s':>12} {'ttft_s':>10} "
        f"{'output_tokens':>14} {'token_rate_tps':>15} {'gpu_peak_mb':>12}"
    )
    for r in results:
        tokens = "-" if r.output_tokens is None else str(r.output_tokens)
        print(
            f"{r.backend:<24} {fmt(r.cold_start_s):>12} {fmt(r.ttft_s):>10} "
            f"{tokens:>14} {fmt(r.token_rate_tps):>15} {fmt(r.gpu_peak_mb):>12}"
        )
        print(
            f"  gpu_mb: before={fmt(r.gpu_before_mb)} "
            f"load={fmt(r.gpu_after_load_mb)} cleanup={fmt(r.gpu_after_cleanup_mb)}"
        )
        if r.gpu_note:
            print(f"  gpu_note: {r.gpu_note}")
        if r.error:
            print(f"  error: {r.error}")
        if r.output_preview:
            print(f"  preview: {r.output_preview}")

    by_name = {r.backend: r for r in results}
    llama = by_name.get("llama.cpp")
    vllm = by_name.get("vLLM(AsyncLLMEngine)")
    if llama and vllm and not llama.error and not vllm.error:
        if llama.ttft_s is not None and vllm.ttft_s is not None:
            print(f"\nTTFT diff (vLLM - llama.cpp): {vllm.ttft_s - llama.ttft_s:+.3f}s")
        if llama.token_rate_tps is not None and vllm.token_rate_tps is not None:
            print(
                f"Token rate diff (vLLM - llama.cpp): "
                f"{vllm.token_rate_tps - llama.token_rate_tps:+.3f} tok/s"
            )
        if llama.cold_start_s is not None and vllm.cold_start_s is not None:
            print(
                f"Cold-start diff (vLLM - llama.cpp): "
                f"{vllm.cold_start_s - llama.cold_start_s:+.3f}s"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone benchmark: llama.cpp vs vLLM AsyncLLMEngine."
    )
    parser.add_argument("--prompt", default="Describe this image in one sentence.")
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are a highly accurate vision-language assistant. "
            "Provide detailed, precise, and well-structured image descriptions."
        ),
    )
    parser.add_argument("--image-path", default=DEFAULT_SINGLE_IMAGE_PATH, help="Single local image path.")
    parser.add_argument("--image-url", default=None, help="Optional remote image url if no image-path.")

    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--print-stream", type=int, default=1, choices=[0, 1], help="Print streamed output.")

    parser.add_argument("--llama-repo-id", default="Qwen/Qwen3-VL-2B-Instruct-GGUF")
    parser.add_argument("--llama-filename", default="Qwen3VL-2B-Instruct-F16.gguf")
    parser.add_argument("--llama-mmproj-filename", default="mmproj-Qwen3VL-2B-Instruct-F16.gguf")
    parser.add_argument("--llama-mmproj-path", default=None, help="Optional local mmproj path.")
    parser.add_argument("--llama-cache-dir", default=None, help="HF cache dir for llama.cpp files.")
    parser.add_argument("--llama-local-dir", default=None, help="Optional local download dir.")
    parser.add_argument("--llama-n-gpu-layers", type=int, default=-1)
    parser.add_argument("--llama-force-reasoning", type=int, default=0, choices=[0, 1])
    parser.add_argument("--llama-image-min-tokens", type=int, default=1024)
    parser.add_argument("--llama-swa-full", type=int, default=0, choices=[0, 1])

    parser.add_argument("--vllm-model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--vllm-model-path", default=None)
    parser.add_argument("--vllm-dtype", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU index for memory sampling.")
    parser.add_argument(
        "--gpu-mem-sample-interval",
        type=float,
        default=0.1,
        help="Sampling interval (seconds) for peak GPU memory tracking.",
    )

    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output-json", default=None, help="Optional report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    count_tokens = build_token_counter(args.tokenizer_model)

    print("=== Run llama.cpp benchmark ===")
    llama_result = benchmark_llamacpp(args, count_tokens)

    print("\n=== Run vLLM AsyncLLMEngine benchmark ===")
    vllm_result = benchmark_vllm_async_engine(args, count_tokens)

    results = [llama_result, vllm_result]
    print_report(results)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
        print(f"\njson report saved to: {args.output_json}")

    return 1 if any(r.error for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
