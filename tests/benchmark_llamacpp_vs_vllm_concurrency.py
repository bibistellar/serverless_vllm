#!/usr/bin/env python3
"""Concurrent benchmark: llama.cpp vs vLLM AsyncLLMEngine.

Features:
1) Cold-start time per backend.
2) Concurrent requests benchmark with configurable concurrency/total requests.
3) Per-request TTFT/E2E stats (p50/p95) and aggregate throughput.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import gc
import json
import multiprocessing as mp
import os
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional


DEFAULT_SINGLE_IMAGE_PATH = "/root/Code/video_anomaly_analysis_system/model_pool_serve/demo.png"

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

STREAM_PRINT_LOCK = threading.Lock()


@dataclass
class RequestMetric:
    request_id: str
    ttft_s: Optional[float]
    e2e_s: Optional[float]
    output_tokens: int
    text: str
    error: Optional[str] = None


@dataclass
class BackendResult:
    backend: str
    instance_count: int = 1
    cold_start_s: Optional[float] = None
    concurrency: int = 1
    total_requests: int = 1
    succeeded_requests: int = 0
    failed_requests: int = 0
    wall_time_s: Optional[float] = None
    throughput_rps: Optional[float] = None
    throughput_tps: Optional[float] = None
    ttft_p50_s: Optional[float] = None
    ttft_p95_s: Optional[float] = None
    e2e_p50_s: Optional[float] = None
    e2e_p95_s: Optional[float] = None
    output_tokens_total: int = 0
    sample_preview: str = ""
    error: Optional[str] = None


class GPUMemoryTracker:
    def __init__(self, gpu_index: int, sample_interval_s: float) -> None:
        self.gpu_index = gpu_index
        self.sample_interval_s = max(sample_interval_s, 0.05)
        self.available = False
        self.error: Optional[str] = None
        self.before_mb: Optional[float] = None
        self.after_load_mb: Optional[float] = None
        self.after_cleanup_mb: Optional[float] = None
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

    def _read_mb(self) -> Optional[float]:
        if not self.available or self._pynvml is None or self._handle is None:
            return None
        try:
            info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return info.used / (1024.0 * 1024.0)
        except Exception:
            return None

    def snapshot(self) -> Optional[float]:
        used_mb = self._read_mb()
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
            if "sampler" not in str(exc):
                raise

    setattr(llama_model_cls, "__del__", _safe_llama_model_del)


def configure_vllm_spawn() -> None:
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


def image_to_base64_data_uri(file_path: str, fallback_mime: str = "application/octet-stream") -> str:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    mime_type = _IMAGE_MIME_TYPES.get(ext, fallback_mime)
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def pick_single_image_source(args: argparse.Namespace) -> Optional[str]:
    if args.image_path:
        return args.image_path
    if args.image_url:
        return args.image_url
    return None


def build_llamacpp_messages(args: argparse.Namespace) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})

    image_source = pick_single_image_source(args)
    if not image_source:
        user_content: Any = args.prompt
    elif image_source.startswith("http://") or image_source.startswith("https://"):
        user_content = [
            {"type": "image_url", "image_url": {"url": image_source}},
            {"type": "text", "text": args.prompt},
        ]
    else:
        user_content = [
            {"type": "image_url", "image_url": {"url": image_to_base64_data_uri(image_source)}},
            {"type": "text", "text": args.prompt},
        ]
    messages.append({"role": "user", "content": user_content})
    return messages


def build_vllm_messages(args: argparse.Namespace) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    image_source = pick_single_image_source(args)
    if not image_source:
        messages.append({"role": "user", "content": args.prompt})
        return messages
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_source},
                {"type": "text", "text": args.prompt},
            ],
        }
    )
    return messages


def build_token_counter(model_id: str) -> Callable[[str], int]:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        def _count(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False)) if text else 0

        return _count
    except Exception:
        return lambda text: len(text.split()) if text else 0


def extract_delta_text(choice: Dict[str, Any]) -> str:
    delta = choice.get("delta", {})
    content = delta.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)
    return ""


def stream_print(enabled: bool, prefix: str, text: str) -> None:
    if not enabled or not text:
        return
    with STREAM_PRINT_LOCK:
        print(f"{prefix}{text}", end="", flush=True)


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = max(0.0, min(1.0, q / 100.0)) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def aggregate_metrics(result: BackendResult, metrics: List[RequestMetric], wall_time_s: float) -> None:
    result.total_requests = len(metrics)
    result.wall_time_s = wall_time_s
    result.succeeded_requests = sum(1 for m in metrics if not m.error)
    result.failed_requests = result.total_requests - result.succeeded_requests

    ok = [m for m in metrics if not m.error]
    if not ok:
        result.error = next((m.error for m in metrics if m.error), "all requests failed")
        return

    ttfts = [m.ttft_s for m in ok if m.ttft_s is not None]
    e2es = [m.e2e_s for m in ok if m.e2e_s is not None]
    result.ttft_p50_s = percentile([x for x in ttfts if x is not None], 50.0)
    result.ttft_p95_s = percentile([x for x in ttfts if x is not None], 95.0)
    result.e2e_p50_s = percentile([x for x in e2es if x is not None], 50.0)
    result.e2e_p95_s = percentile([x for x in e2es if x is not None], 95.0)

    result.output_tokens_total = sum(m.output_tokens for m in ok)
    result.throughput_rps = result.succeeded_requests / max(wall_time_s, 1e-9)
    result.throughput_tps = result.output_tokens_total / max(wall_time_s, 1e-9)
    result.sample_preview = (ok[0].text or "")[:160]


def _llama_worker_main(
    worker_idx: int,
    args_dict: Dict[str, Any],
    mmproj_path: str,
    req_q: Any,
    res_q: Any,
) -> None:
    llm = None
    try:
        patch_llamacpp_destructor_bug()
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen3VLChatHandler

        init_kwargs: Dict[str, Any] = {
            "repo_id": args_dict["llama_repo_id"],
            "filename": args_dict["llama_filename"],
            "chat_handler": Qwen3VLChatHandler(
                clip_model_path=mmproj_path,
                force_reasoning=bool(args_dict["llama_force_reasoning"]),
                image_min_tokens=args_dict["llama_image_min_tokens"],
            ),
            "n_gpu_layers": args_dict["llama_n_gpu_layers"],
            "n_ctx": args_dict["max_model_len"],
            "verbose": False,
        }
        if args_dict.get("llama_cache_dir"):
            init_kwargs["cache_dir"] = args_dict["llama_cache_dir"]
        if args_dict.get("llama_local_dir"):
            init_kwargs["local_dir"] = args_dict["llama_local_dir"]
            init_kwargs["local_dir_use_symlinks"] = "auto"
        if args_dict.get("llama_mmproj_filename"):
            init_kwargs["additional_files"] = [args_dict["llama_mmproj_filename"]]
        if args_dict.get("llama_swa_full"):
            init_kwargs["swa_full"] = True

        load_start = time.perf_counter()
        llm = Llama.from_pretrained(**init_kwargs)
        load_s = time.perf_counter() - load_start
        res_q.put({"type": "ready", "worker_idx": worker_idx, "load_s": load_s})

        while True:
            task = req_q.get()
            if task is None:
                break
            req_id = task["request_id"]
            messages = task["messages"]
            max_tokens = task["max_tokens"]
            temperature = task["temperature"]
            top_p = task["top_p"]
            print_stream = bool(task["print_stream"])
            print_prefix = bool(task["print_stream_prefix"])

            start = time.perf_counter()
            first_token_at: Optional[float] = None
            pieces: List[str] = []
            try:
                stream = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                )
                prefix = f"[llama#{worker_idx} {req_id}] " if print_prefix else ""
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
                    stream_print(print_stream, prefix, delta)
                if print_stream:
                    with STREAM_PRINT_LOCK:
                        print()
                end = time.perf_counter()
                res_q.put(
                    {
                        "type": "result",
                        "request_id": req_id,
                        "ttft_s": None if first_token_at is None else first_token_at - start,
                        "e2e_s": end - start,
                        "text": "".join(pieces),
                        "error": None,
                    }
                )
            except Exception as exc:
                res_q.put(
                    {
                        "type": "result",
                        "request_id": req_id,
                        "ttft_s": None,
                        "e2e_s": None,
                        "text": "",
                        "error": str(exc),
                    }
                )
    except Exception as exc:
        res_q.put({"type": "fatal", "worker_idx": worker_idx, "error": str(exc)})
    finally:
        if llm is not None and hasattr(llm, "close"):
            try:
                llm.close()
            except Exception:
                pass
        res_q.put({"type": "stopped", "worker_idx": worker_idx})


def run_llamacpp_concurrent(args: argparse.Namespace, count_tokens: Callable[[str], int]) -> Dict[str, Any]:
    from huggingface_hub import hf_hub_download

    patch_llamacpp_destructor_bug()

    mmproj_path = args.llama_mmproj_path
    if not mmproj_path:
        mmproj_path = hf_hub_download(
            repo_id=args.llama_repo_id,
            filename=args.llama_mmproj_filename,
            cache_dir=args.llama_cache_dir,
            local_dir=args.llama_local_dir,
        )

    instance_count = max(1, int(args.llama_instance_count))
    ctx = mp.get_context("spawn")
    res_q = ctx.Queue()
    req_queues = [ctx.Queue() for _ in range(instance_count)]
    args_dict = {
        "llama_repo_id": args.llama_repo_id,
        "llama_filename": args.llama_filename,
        "llama_mmproj_filename": args.llama_mmproj_filename,
        "llama_cache_dir": args.llama_cache_dir,
        "llama_local_dir": args.llama_local_dir,
        "llama_force_reasoning": args.llama_force_reasoning,
        "llama_image_min_tokens": args.llama_image_min_tokens,
        "llama_n_gpu_layers": args.llama_n_gpu_layers,
        "llama_swa_full": args.llama_swa_full,
        "max_model_len": args.max_model_len,
    }

    t0 = time.perf_counter()
    workers = [
        ctx.Process(
            target=_llama_worker_main,
            args=(idx, args_dict, mmproj_path, req_queues[idx], res_q),
            daemon=True,
        )
        for idx in range(instance_count)
    ]
    for p in workers:
        p.start()

    ready = 0
    while ready < instance_count:
        try:
            msg = res_q.get(timeout=1200)
        except queue.Empty:
            raise TimeoutError("llama workers start timeout")
        if msg.get("type") == "ready":
            ready += 1
        elif msg.get("type") == "fatal":
            raise RuntimeError(f"llama worker failed during startup: {msg.get('error')}")
    cold_start_s = time.perf_counter() - t0

    messages = build_llamacpp_messages(args)
    metrics: List[RequestMetric] = []
    wall_start = time.perf_counter()
    for i in range(args.total_requests):
        req_id = f"llama-{i}"
        worker_idx = i % instance_count
        req_queues[worker_idx].put(
            {
                "request_id": req_id,
                "messages": messages,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "print_stream": args.print_stream,
                "print_stream_prefix": args.print_stream_prefix,
            }
        )

    collected = 0
    while collected < args.total_requests:
        try:
            msg = res_q.get(timeout=1200)
        except queue.Empty:
            dead = [p.pid for p in workers if not p.is_alive()]
            raise RuntimeError(f"llama result timeout, dead_workers={dead}")
        if msg.get("type") == "result":
            text = str(msg.get("text") or "")
            metrics.append(
                RequestMetric(
                    request_id=str(msg.get("request_id")),
                    ttft_s=msg.get("ttft_s"),
                    e2e_s=msg.get("e2e_s"),
                    output_tokens=count_tokens(text),
                    text=text,
                    error=msg.get("error"),
                )
            )
            collected += 1
        elif msg.get("type") == "fatal":
            raise RuntimeError(f"llama worker crashed: {msg.get('error')}")
    wall_time_s = time.perf_counter() - wall_start

    for q in req_queues:
        q.put(None)
    for p in workers:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)

    gc.collect()
    return {"cold_start_s": cold_start_s, "metrics": metrics, "wall_time_s": wall_time_s}


async def run_vllm_concurrent_async(
    args: argparse.Namespace, count_tokens: Callable[[str], int]
) -> Dict[str, Any]:
    configure_vllm_spawn()
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.inputs.data import TextPrompt
    from vllm.sampling_params import SamplingParams

    model_ref = args.vllm_model_path or args.vllm_model
    tp_size = args.tensor_parallel_size if args.tensor_parallel_size is not None else (torch.cuda.device_count() or 1)

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
    cold_start_s = time.perf_counter() - t0

    messages = build_vllm_messages(args)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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
        base_inputs: Any = TextPrompt(
            prompt=prompt,
            multi_modal_data=mm_data,
            mm_processor_kwargs=video_kwargs,
        )
    else:
        base_inputs = prompt

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    sem = asyncio.Semaphore(args.concurrency)

    async def _one_request(i: int) -> RequestMetric:
        req_id = f"vllm-{uuid.uuid4().hex[:8]}-{i}"
        async with sem:
            start = time.perf_counter()
            first_token_at: Optional[float] = None
            pieces: List[str] = []
            prev_text = ""
            try:
                prefix = f"[vllm {req_id}] " if args.print_stream_prefix else ""
                async for output in engine.generate(base_inputs, sampling_params, req_id):
                    for completion in output.outputs:
                        current = completion.text or ""
                        delta = current[len(prev_text) :] if current.startswith(prev_text) else current
                        if not delta:
                            continue
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        pieces.append(delta)
                        prev_text = current
                        stream_print(bool(args.print_stream), prefix, delta)
                if args.print_stream:
                    with STREAM_PRINT_LOCK:
                        print()
            except Exception as exc:
                try:
                    await engine.abort(req_id)
                except Exception:
                    pass
                return RequestMetric(req_id, None, None, 0, "", error=str(exc))

            end = time.perf_counter()
            text = "".join(pieces)
            return RequestMetric(
                req_id,
                None if first_token_at is None else first_token_at - start,
                end - start,
                count_tokens(text),
                text,
            )

    wall_start = time.perf_counter()
    tasks = [asyncio.create_task(_one_request(i)) for i in range(args.total_requests)]
    metrics = await asyncio.gather(*tasks)
    wall_time_s = time.perf_counter() - wall_start

    del engine
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return {"cold_start_s": cold_start_s, "metrics": list(metrics), "wall_time_s": wall_time_s}


def run_vllm_concurrent(args: argparse.Namespace, count_tokens: Callable[[str], int]) -> Dict[str, Any]:
    return asyncio.run(run_vllm_concurrent_async(args, count_tokens))


def fmt(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.3f}"


def print_report(results: List[BackendResult], gpu_stats: Dict[str, Dict[str, Optional[float]]]) -> None:
    print("\n=== Concurrency Benchmark Report ===")
    print(
        f"{'backend':<24} {'inst':>5} {'cold_start_s':>12} {'conc':>6} {'req':>6} {'succ':>6} {'fail':>6} "
        f"{'wall_s':>10} {'rps':>10} {'tps':>10} {'ttft_p50':>10} {'ttft_p95':>10} "
        f"{'e2e_p50':>10} {'e2e_p95':>10}"
    )
    for r in results:
        print(
            f"{r.backend:<24} {r.instance_count:>5} {fmt(r.cold_start_s):>12} {r.concurrency:>6} {r.total_requests:>6} "
            f"{r.succeeded_requests:>6} {r.failed_requests:>6} {fmt(r.wall_time_s):>10} "
            f"{fmt(r.throughput_rps):>10} {fmt(r.throughput_tps):>10} {fmt(r.ttft_p50_s):>10} "
            f"{fmt(r.ttft_p95_s):>10} {fmt(r.e2e_p50_s):>10} {fmt(r.e2e_p95_s):>10}"
        )
        if r.sample_preview:
            print(f"  preview: {r.sample_preview}")
        if r.error:
            print(f"  error: {r.error}")
        g = gpu_stats.get(r.backend)
        if g:
            print(
                f"  gpu_mb: before={fmt(g.get('before'))} load={fmt(g.get('load'))} "
                f"peak={fmt(g.get('peak'))} cleanup={fmt(g.get('cleanup'))}"
            )
            if g.get("note"):
                print(f"  gpu_note: {g['note']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concurrent benchmark: llama.cpp vs vLLM.")
    parser.add_argument("--backend", default="both", choices=["both", "llama", "vllm"])
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--total-requests", type=int, default=16)

    parser.add_argument("--prompt", default="Describe this image in one sentence.")
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are a highly accurate vision-language assistant. "
            "Provide detailed, precise, and well-structured image descriptions."
        ),
    )
    parser.add_argument("--image-path", default=DEFAULT_SINGLE_IMAGE_PATH)
    parser.add_argument("--image-url", default=None)

    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--print-stream", type=int, default=0, choices=[0, 1])
    parser.add_argument("--print-stream-prefix", type=int, default=1, choices=[0, 1])

    parser.add_argument("--llama-repo-id", default="Qwen/Qwen3-VL-2B-Instruct-GGUF")
    parser.add_argument("--llama-filename", default="Qwen3VL-2B-Instruct-F16.gguf")
    parser.add_argument("--llama-mmproj-filename", default="mmproj-Qwen3VL-2B-Instruct-F16.gguf")
    parser.add_argument("--llama-mmproj-path", default=None)
    parser.add_argument("--llama-cache-dir", default=None)
    parser.add_argument("--llama-local-dir", default=None)
    parser.add_argument("--llama-n-gpu-layers", type=int, default=-1)
    parser.add_argument("--llama-force-reasoning", type=int, default=0, choices=[0, 1])
    parser.add_argument("--llama-image-min-tokens", type=int, default=1024)
    parser.add_argument("--llama-swa-full", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--llama-instance-count",
        type=int,
        default=1,
        help="Number of llama.cpp instances to create for round-robin request routing.",
    )
    parser.add_argument(
        "--llama-serialize-requests",
        type=int,
        default=1,
        choices=[0, 1],
        help="Deprecated in process mode (ignored). Kept for CLI compatibility.",
    )

    parser.add_argument("--vllm-model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--vllm-model-path", default=None)
    parser.add_argument("--vllm-dtype", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)

    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-mem-sample-interval", type=float, default=0.1)
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    count_tokens = build_token_counter(args.tokenizer_model)
    results: List[BackendResult] = []
    gpu_stats: Dict[str, Dict[str, Optional[float]]] = {}

    if args.backend in {"both", "llama"}:
        print("=== Run llama.cpp concurrent benchmark ===")
        if args.concurrency > args.llama_instance_count:
            print(
                f"[INFO] llama instances={args.llama_instance_count}, concurrency={args.concurrency}. "
                "Requests are round-robin dispatched to isolated worker processes."
            )
        tracker = GPUMemoryTracker(args.gpu_index, args.gpu_mem_sample_interval)
        tracker.start()
        try:
            payload = run_llamacpp_concurrent(args, count_tokens)
            tracker.after_load_mb = tracker.snapshot()
            result = BackendResult(
                backend="llama.cpp",
                instance_count=args.llama_instance_count,
                cold_start_s=payload["cold_start_s"],
                concurrency=args.concurrency,
            )
            aggregate_metrics(result, payload["metrics"], payload["wall_time_s"])
            results.append(result)
        except Exception as exc:
            results.append(
                BackendResult(
                    backend="llama.cpp",
                    instance_count=args.llama_instance_count,
                    concurrency=args.concurrency,
                    total_requests=args.total_requests,
                    failed_requests=args.total_requests,
                    error=str(exc),
                )
            )
        finally:
            gc.collect()
            tracker.after_cleanup_mb = tracker.snapshot()
            tracker.stop()
            gpu_stats["llama.cpp"] = {
                "before": tracker.before_mb,
                "load": tracker.after_load_mb,
                "peak": tracker.peak_mb,
                "cleanup": tracker.after_cleanup_mb,
                "note": tracker.error,
            }

    if args.backend in {"both", "vllm"}:
        print("\n=== Run vLLM concurrent benchmark ===")
        tracker = GPUMemoryTracker(args.gpu_index, args.gpu_mem_sample_interval)
        tracker.start()
        try:
            payload = run_vllm_concurrent(args, count_tokens)
            tracker.after_load_mb = tracker.snapshot()
            result = BackendResult(
                backend="vLLM(AsyncLLMEngine)",
                instance_count=1,
                cold_start_s=payload["cold_start_s"],
                concurrency=args.concurrency,
            )
            aggregate_metrics(result, payload["metrics"], payload["wall_time_s"])
            results.append(result)
        except Exception as exc:
            results.append(
                BackendResult(
                    backend="vLLM(AsyncLLMEngine)",
                    instance_count=1,
                    concurrency=args.concurrency,
                    total_requests=args.total_requests,
                    failed_requests=args.total_requests,
                    error=str(exc),
                )
            )
        finally:
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            tracker.after_cleanup_mb = tracker.snapshot()
            tracker.stop()
            gpu_stats["vLLM(AsyncLLMEngine)"] = {
                "before": tracker.before_mb,
                "load": tracker.after_load_mb,
                "peak": tracker.peak_mb,
                "cleanup": tracker.after_cleanup_mb,
                "note": tracker.error,
            }

    print_report(results, gpu_stats)

    if args.output_json:
        payload = {
            "results": [asdict(r) for r in results],
            "gpu_stats": gpu_stats,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\njson report saved to: {args.output_json}")

    return 1 if any(r.error for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
