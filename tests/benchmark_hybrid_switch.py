#!/usr/bin/env python3
"""Hybrid switch benchmark: pure llama.cpp vs pure vLLM vs switch-over.

Flow:
1) Concurrency ramps from 1..max_concurrency.
2) Pure llama.cpp baseline (multiple llama instances).
3) Pure vLLM baseline.
4) Hybrid switch:
   - Start llama.cpp instances (default 2) and one vLLM startup task simultaneously.
   - Route requests to llama.cpp until vLLM is ready.
   - Route all remaining requests to vLLM after cutover.

Metrics:
- cold_start_s
- throughput_rps / throughput_tps
- ttft_p50 / ttft_p95
- e2e_p50 / e2e_p95
- token_rate_avg_tps (average per-request generation speed after first token)
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import multiprocessing as mp
import queue
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import benchmark_llamacpp_vs_vllm_concurrency as base


@dataclass
class RequestMetric:
    request_id: str
    ttft_s: Optional[float]
    e2e_s: Optional[float]
    output_tokens: int
    text: str
    first_token_at_abs: Optional[float] = None
    error: Optional[str] = None


@dataclass
class LevelResult:
    backend: str
    concurrency: int
    total_requests: int
    succeeded_requests: int
    failed_requests: int
    wall_time_s: float
    throughput_rps: Optional[float]
    throughput_tps: Optional[float]
    token_rate_avg_tps: Optional[float]
    ttft_p50_s: Optional[float]
    ttft_p95_s: Optional[float]
    e2e_p50_s: Optional[float]
    e2e_p95_s: Optional[float]
    error: Optional[str] = None


@dataclass
class ScenarioResult:
    backend: str
    cold_start_s: Optional[float]
    total_requests: int
    succeeded_requests: int
    failed_requests: int
    wall_time_s: float
    throughput_rps: Optional[float]
    throughput_tps: Optional[float]
    token_rate_avg_tps: Optional[float]
    ttft_p50_s: Optional[float]
    ttft_p95_s: Optional[float]
    e2e_p50_s: Optional[float]
    e2e_p95_s: Optional[float]
    cutover_concurrency: Optional[int] = None
    llama_requests_before_cutover: Optional[int] = None
    vllm_requests_after_cutover: Optional[int] = None
    error: Optional[str] = None


@dataclass
class VLLMRuntime:
    engine: Any
    base_inputs: Any
    sampling_params: Any
    cold_start_s: float


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


def _token_rate_tps(metric: RequestMetric) -> Optional[float]:
    if metric.error or metric.ttft_s is None or metric.e2e_s is None:
        return None
    gen_s = metric.e2e_s - metric.ttft_s
    if gen_s <= 1e-9:
        return None
    return metric.output_tokens / gen_s


def aggregate_level(backend: str, concurrency: int, metrics: List[RequestMetric], wall_s: float) -> LevelResult:
    total = len(metrics)
    ok = [m for m in metrics if not m.error]
    fail = total - len(ok)

    if not ok:
        return LevelResult(
            backend=backend,
            concurrency=concurrency,
            total_requests=total,
            succeeded_requests=0,
            failed_requests=fail,
            wall_time_s=wall_s,
            throughput_rps=None,
            throughput_tps=None,
            token_rate_avg_tps=None,
            ttft_p50_s=None,
            ttft_p95_s=None,
            e2e_p50_s=None,
            e2e_p95_s=None,
            error=next((m.error for m in metrics if m.error), "all requests failed"),
        )

    ttft_vals = [m.ttft_s for m in ok if m.ttft_s is not None]
    e2e_vals = [m.e2e_s for m in ok if m.e2e_s is not None]
    token_rate_vals = [x for x in (_token_rate_tps(m) for m in ok) if x is not None]
    out_tokens = sum(m.output_tokens for m in ok)

    return LevelResult(
        backend=backend,
        concurrency=concurrency,
        total_requests=total,
        succeeded_requests=len(ok),
        failed_requests=fail,
        wall_time_s=wall_s,
        throughput_rps=len(ok) / max(wall_s, 1e-9),
        throughput_tps=out_tokens / max(wall_s, 1e-9),
        token_rate_avg_tps=(sum(token_rate_vals) / len(token_rate_vals)) if token_rate_vals else None,
        ttft_p50_s=percentile(ttft_vals, 50.0),
        ttft_p95_s=percentile(ttft_vals, 95.0),
        e2e_p50_s=percentile(e2e_vals, 50.0),
        e2e_p95_s=percentile(e2e_vals, 95.0),
        error=None,
    )


def aggregate_scenario(
    backend: str,
    cold_start_s: Optional[float],
    metrics: List[RequestMetric],
    wall_s: float,
    cutover_concurrency: Optional[int] = None,
    llama_count: Optional[int] = None,
    vllm_count: Optional[int] = None,
) -> ScenarioResult:
    level = aggregate_level(backend, concurrency=0, metrics=metrics, wall_s=wall_s)
    return ScenarioResult(
        backend=backend,
        cold_start_s=cold_start_s,
        total_requests=level.total_requests,
        succeeded_requests=level.succeeded_requests,
        failed_requests=level.failed_requests,
        wall_time_s=wall_s,
        throughput_rps=level.throughput_rps,
        throughput_tps=level.throughput_tps,
        token_rate_avg_tps=level.token_rate_avg_tps,
        ttft_p50_s=level.ttft_p50_s,
        ttft_p95_s=level.ttft_p95_s,
        e2e_p50_s=level.e2e_p50_s,
        e2e_p95_s=level.e2e_p95_s,
        cutover_concurrency=cutover_concurrency,
        llama_requests_before_cutover=llama_count,
        vllm_requests_after_cutover=vllm_count,
        error=level.error,
    )


def fmt(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.3f}"


def first_token_delay_from_start(metrics: List[RequestMetric], start_ts: float) -> Optional[float]:
    candidates = [m.first_token_at_abs for m in metrics if m.first_token_at_abs is not None]
    if not candidates:
        return None
    return max(0.0, min(candidates) - start_ts)


def progress(args: argparse.Namespace, msg: str) -> None:
    if bool(args.show_progress):
        print(f"[progress] {msg}", flush=True)


def _llama_args_dict(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "text_only": bool(args.text_only),
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


def _build_llama_messages(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if bool(args.text_only):
        messages: List[Dict[str, Any]] = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": args.prompt})
        return messages
    return base.build_llamacpp_messages(args)


def _llama_worker_main(
    worker_idx: int,
    args_dict: Dict[str, Any],
    mmproj_path: Optional[str],
    req_q: Any,
    res_q: Any,
) -> None:
    llm = None
    try:
        base.patch_llamacpp_destructor_bug()
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen3VLChatHandler

        text_only = bool(args_dict.get("text_only", False))
        init_kwargs: Dict[str, Any] = {
            "repo_id": args_dict["llama_repo_id"],
            "filename": args_dict["llama_filename"],
            "n_gpu_layers": args_dict["llama_n_gpu_layers"],
            "n_ctx": args_dict["max_model_len"],
            "verbose": False,
        }
        if not text_only:
            init_kwargs["chat_handler"] = Qwen3VLChatHandler(
                clip_model_path=mmproj_path,
                force_reasoning=bool(args_dict["llama_force_reasoning"]),
                image_min_tokens=args_dict["llama_image_min_tokens"],
            )

        if args_dict.get("llama_cache_dir"):
            init_kwargs["cache_dir"] = args_dict["llama_cache_dir"]
        if args_dict.get("llama_local_dir"):
            init_kwargs["local_dir"] = args_dict["llama_local_dir"]
            init_kwargs["local_dir_use_symlinks"] = "auto"
        if (not text_only) and args_dict.get("llama_mmproj_filename"):
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
            submit_at_mono = task.get("submit_at_mono")
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
                    delta = base.extract_delta_text(choices[0])
                    if not delta:
                        continue
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    pieces.append(delta)
                    base.stream_print(print_stream, prefix, delta)
                if print_stream:
                    with base.STREAM_PRINT_LOCK:
                        print()
                end = time.perf_counter()
                if isinstance(submit_at_mono, (int, float)):
                    e2e_s = max(0.0, end - float(submit_at_mono))
                else:
                    e2e_s = end - start
                res_q.put(
                    {
                        "type": "result",
                        "request_id": req_id,
                        "ttft_s": None if first_token_at is None else first_token_at - start,
                        "e2e_s": e2e_s,
                        "first_token_at_abs": first_token_at,
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
                        "first_token_at_abs": None,
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


class LlamaProcessPool:
    def __init__(self, args: argparse.Namespace, instance_count: int) -> None:
        self.args = args
        self.instance_count = max(1, int(instance_count))
        self.ctx = mp.get_context("spawn")
        self.req_queues: List[Any] = []
        self.res_q: Any = None
        self.workers: List[Any] = []
        self.next_worker = 0
        self.cold_start_s: Optional[float] = None

    def start(self) -> None:
        base.patch_llamacpp_destructor_bug()
        mmproj_path: Optional[str] = None
        if not bool(self.args.text_only):
            from huggingface_hub import hf_hub_download

            mmproj_path = self.args.llama_mmproj_path
            if not mmproj_path:
                mmproj_path = hf_hub_download(
                    repo_id=self.args.llama_repo_id,
                    filename=self.args.llama_mmproj_filename,
                    cache_dir=self.args.llama_cache_dir,
                    local_dir=self.args.llama_local_dir,
                )

        self.res_q = self.ctx.Queue()
        self.req_queues = [self.ctx.Queue() for _ in range(self.instance_count)]
        args_dict = _llama_args_dict(self.args)

        t0 = time.perf_counter()
        self.workers = [
            self.ctx.Process(
                target=_llama_worker_main,
                args=(idx, args_dict, mmproj_path, self.req_queues[idx], self.res_q),
                daemon=True,
            )
            for idx in range(self.instance_count)
        ]
        for proc in self.workers:
            proc.start()

        ready = 0
        while ready < self.instance_count:
            try:
                msg = self.res_q.get(timeout=1800)
            except queue.Empty as exc:
                raise TimeoutError("llama workers startup timeout") from exc
            if msg.get("type") == "ready":
                ready += 1
            elif msg.get("type") == "fatal":
                raise RuntimeError(f"llama worker startup failed: {msg.get('error')}")
        self.cold_start_s = time.perf_counter() - t0

    def submit(self, request_id: str, messages: List[Dict[str, Any]]) -> None:
        idx = self.next_worker
        self.next_worker = (self.next_worker + 1) % self.instance_count
        self.req_queues[idx].put(
            {
                "request_id": request_id,
                "messages": messages,
                "max_tokens": self.args.max_tokens,
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "submit_at_mono": time.perf_counter(),
                "print_stream": self.args.print_stream,
                "print_stream_prefix": self.args.print_stream_prefix,
            }
        )

    def collect(self, expect: int, count_tokens: Any, timeout_s: float = 1800.0) -> List[RequestMetric]:
        if expect <= 0:
            return []
        out: List[RequestMetric] = []
        deadline = time.time() + timeout_s
        while len(out) < expect:
            remain = max(0.1, deadline - time.time())
            try:
                msg = self.res_q.get(timeout=remain)
            except queue.Empty as exc:
                dead = [p.pid for p in self.workers if not p.is_alive()]
                raise RuntimeError(f"llama collect timeout, dead_workers={dead}") from exc

            if msg.get("type") == "result":
                text = str(msg.get("text") or "")
                out.append(
                    RequestMetric(
                        request_id=str(msg.get("request_id")),
                        ttft_s=msg.get("ttft_s"),
                        e2e_s=msg.get("e2e_s"),
                        output_tokens=count_tokens(text),
                        text=text,
                        first_token_at_abs=msg.get("first_token_at_abs"),
                        error=msg.get("error"),
                    )
                )
            elif msg.get("type") == "fatal":
                raise RuntimeError(f"llama worker crashed: {msg.get('error')}")
        return out

    def stop(self) -> None:
        for q in self.req_queues:
            q.put(None)
        for proc in self.workers:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)


async def start_vllm_runtime(args: argparse.Namespace) -> VLLMRuntime:
    base.configure_vllm_spawn()
    import torch
    from transformers import AutoProcessor, AutoTokenizer
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.inputs.data import TextPrompt
    from vllm.sampling_params import SamplingParams

    model_ref = args.vllm_model_path or args.vllm_model
    tp_size = args.tensor_parallel_size if args.tensor_parallel_size is not None else (torch.cuda.device_count() or 1)

    t0 = time.perf_counter()
    text_only = bool(args.text_only)
    processor = None
    tokenizer = None
    if text_only:
        tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    else:
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
    cold_s = time.perf_counter() - t0

    if text_only:
        messages: List[Dict[str, Any]] = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": args.prompt})
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        base_inputs = prompt
    else:
        from qwen_vl_utils import process_vision_info

        messages = base.build_vllm_messages(args)
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        mm_data: Dict[str, Any] = {}
        video_kwargs: Dict[str, Any] = {}
        if base.pick_single_image_source(args):
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
            base_inputs = TextPrompt(
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
    return VLLMRuntime(engine=engine, base_inputs=base_inputs, sampling_params=sampling_params, cold_start_s=cold_s)


async def run_vllm_requests(
    runtime: VLLMRuntime,
    count_tokens: Any,
    req_count: int,
    concurrency: int,
    print_stream: bool,
    print_stream_prefix: bool,
) -> List[RequestMetric]:
    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def _one(idx: int) -> RequestMetric:
        req_id = f"vllm-{uuid.uuid4().hex[:8]}-{idx}"
        submit_at = time.perf_counter()
        async with sem:
            start = time.perf_counter()
            first_token_at: Optional[float] = None
            pieces: List[str] = []
            prev_text = ""
            try:
                prefix = f"[vllm {req_id}] " if print_stream_prefix else ""
                async for output in runtime.engine.generate(runtime.base_inputs, runtime.sampling_params, req_id):
                    for completion in output.outputs:
                        current = completion.text or ""
                        delta = current[len(prev_text) :] if current.startswith(prev_text) else current
                        if not delta:
                            continue
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        pieces.append(delta)
                        prev_text = current
                        base.stream_print(bool(print_stream), prefix, delta)
                if print_stream:
                    with base.STREAM_PRINT_LOCK:
                        print()
            except Exception as exc:
                try:
                    await runtime.engine.abort(req_id)
                except Exception:
                    pass
                return RequestMetric(req_id, None, None, 0, "", error=str(exc))

            end = time.perf_counter()
            text = "".join(pieces)
            return RequestMetric(
                request_id=req_id,
                ttft_s=None if first_token_at is None else first_token_at - start,
                e2e_s=end - submit_at,
                output_tokens=count_tokens(text),
                text=text,
                first_token_at_abs=first_token_at,
                error=None,
            )

    tasks = [asyncio.create_task(_one(i)) for i in range(req_count)]
    return list(await asyncio.gather(*tasks))


async def stop_vllm_runtime(runtime: Optional[VLLMRuntime]) -> None:
    if runtime is None:
        return
    del runtime.engine
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_pure_llama(args: argparse.Namespace, count_tokens: Any) -> Tuple[List[LevelResult], ScenarioResult]:
    messages = _build_llama_messages(args)
    pool = LlamaProcessPool(args=args, instance_count=args.llama_max_instances)
    level_results: List[LevelResult] = []
    all_metrics: List[RequestMetric] = []
    wall_start = time.perf_counter()
    try:
        progress(args, f"pure-llama: starting pool instances={args.llama_max_instances}")
        pool.start()
        progress(args, f"pure-llama: pool ready cold_start_s={fmt(pool.cold_start_s)}")
        for c in range(1, args.max_concurrency + 1):
            progress(args, f"pure-llama: level concurrency={c} begin")
            level_metrics: List[RequestMetric] = []
            t0 = time.perf_counter()
            for r in range(args.rounds_per_level):
                progress(args, f"pure-llama: level c={c} round={r + 1}/{args.rounds_per_level}")
                for i in range(c):
                    rid = f"llama-c{c}-r{r}-{i}"
                    pool.submit(rid, messages)
                level_metrics.extend(pool.collect(c, count_tokens))
            wall_s = time.perf_counter() - t0
            all_metrics.extend(level_metrics)
            level_results.append(aggregate_level("llama.cpp", c, level_metrics, wall_s))
            progress(
                args,
                f"pure-llama: level c={c} done req={len(level_metrics)} wall_s={fmt(wall_s)} "
                f"rps={fmt(level_results[-1].throughput_rps)}",
            )
    finally:
        pool.stop()
        progress(args, "pure-llama: pool stopped")

    total_wall = time.perf_counter() - wall_start
    scenario = aggregate_scenario(
        "llama.cpp",
        first_token_delay_from_start(all_metrics, wall_start),
        all_metrics,
        total_wall,
    )
    return level_results, scenario


async def run_pure_vllm(args: argparse.Namespace, count_tokens: Any) -> Tuple[List[LevelResult], ScenarioResult]:
    runtime: Optional[VLLMRuntime] = None
    level_results: List[LevelResult] = []
    all_metrics: List[RequestMetric] = []
    wall_start = time.perf_counter()
    try:
        progress(args, "pure-vllm: starting runtime")
        runtime = await start_vllm_runtime(args)
        progress(args, f"pure-vllm: runtime ready cold_start_s={fmt(runtime.cold_start_s)}")
        for c in range(1, args.max_concurrency + 1):
            req_count = c * args.rounds_per_level
            progress(args, f"pure-vllm: level concurrency={c} req_count={req_count} begin")
            t0 = time.perf_counter()
            level_metrics = await run_vllm_requests(
                runtime=runtime,
                count_tokens=count_tokens,
                req_count=req_count,
                concurrency=c,
                print_stream=bool(args.print_stream),
                print_stream_prefix=bool(args.print_stream_prefix),
            )
            wall_s = time.perf_counter() - t0
            all_metrics.extend(level_metrics)
            level_results.append(aggregate_level("vLLM(AsyncLLMEngine)", c, level_metrics, wall_s))
            progress(
                args,
                f"pure-vllm: level c={c} done req={len(level_metrics)} wall_s={fmt(wall_s)} "
                f"rps={fmt(level_results[-1].throughput_rps)}",
            )
        total_wall = time.perf_counter() - wall_start
        scenario = aggregate_scenario(
            "vLLM(AsyncLLMEngine)",
            first_token_delay_from_start(all_metrics, wall_start),
            all_metrics,
            total_wall,
        )
        return level_results, scenario
    finally:
        await stop_vllm_runtime(runtime)
        progress(args, "pure-vllm: runtime stopped")


async def run_hybrid_switch(args: argparse.Namespace, count_tokens: Any) -> Tuple[List[LevelResult], ScenarioResult]:
    llama_messages = _build_llama_messages(args)
    llama_pool = LlamaProcessPool(args=args, instance_count=args.switch_llama_instances)
    vllm_task: Optional[asyncio.Task] = None
    runtime: Optional[VLLMRuntime] = None

    level_results: List[LevelResult] = []
    all_metrics: List[RequestMetric] = []
    cutover_level: Optional[int] = None
    llama_before = 0
    vllm_after = 0
    vllm_failed = False
    vllm_fail_reason: Optional[str] = None
    first_char_at_abs: Optional[float] = None
    llama_running = False

    wall_start = time.perf_counter()

    def mark_first_char(metrics: List[RequestMetric]) -> None:
        nonlocal first_char_at_abs
        candidates = [m.first_token_at_abs for m in metrics if m.first_token_at_abs is not None]
        if not candidates:
            return
        cur = min(candidates)
        if first_char_at_abs is None or cur < first_char_at_abs:
            first_char_at_abs = cur

    def release_llama(reason: str) -> None:
        nonlocal llama_running
        if not llama_running:
            return
        progress(args, f"switch: releasing llama resources ({reason})")
        llama_pool.stop()
        llama_running = False
        progress(args, "switch: llama stopped")

    try:
        progress(args, f"switch: starting llama({args.switch_llama_instances})")
        llama_pool.start()
        llama_running = True
        progress(args, f"switch: llama ready cold_start_s={fmt(llama_pool.cold_start_s)}")
        progress(args, "switch: starting vllm async warmup")
        vllm_task = asyncio.create_task(start_vllm_runtime(args))

        for c in range(1, args.max_concurrency + 1):
            if runtime is None and vllm_task is not None and vllm_task.done():
                try:
                    runtime = await vllm_task
                    if cutover_level is None:
                        cutover_level = c
                        progress(args, f"switch: cutover at concurrency={c}")
                    release_llama("vllm ready before level processing")
                except Exception as exc:
                    vllm_failed = True
                    vllm_fail_reason = str(exc)
                    runtime = None
                    vllm_task = None
                    progress(args, f"switch: vllm warmup failed: {vllm_fail_reason}")

            t0 = time.perf_counter()
            if runtime is None:
                progress(args, f"switch: level c={c} using llama phase")
                level_metrics = []
                backend_name = "switch(llama-phase)"
                for r in range(args.rounds_per_level):
                    progress(args, f"switch-llama: level c={c} round={r + 1}/{args.rounds_per_level}")
                    for i in range(c):
                        rid = f"switch-llama-c{c}-r{r}-{i}"
                        llama_pool.submit(rid, llama_messages)
                    # Run blocking queue collection off the event loop so vLLM warmup can proceed.
                    round_metrics = await asyncio.to_thread(llama_pool.collect, c, count_tokens)
                    level_metrics.extend(round_metrics)
                    llama_before += len(round_metrics)
                    mark_first_char(round_metrics)

                    if runtime is None and vllm_task is not None and vllm_task.done():
                        try:
                            runtime = await vllm_task
                            if cutover_level is None:
                                cutover_level = c
                                progress(args, f"switch: cutover at concurrency={c}")
                        except Exception as exc:
                            vllm_failed = True
                            vllm_fail_reason = str(exc)
                            runtime = None
                            vllm_task = None
                            progress(args, f"switch: vllm warmup failed: {vllm_fail_reason}")

                    if runtime is not None and r < args.rounds_per_level - 1:
                        remain_rounds = args.rounds_per_level - r - 1
                        req_count = remain_rounds * c
                        release_llama("mid-level cutover")
                        progress(
                            args,
                            f"switch: level c={c} mid-cutover, remaining_rounds={remain_rounds}, "
                            f"route remaining {req_count} req to vllm",
                        )
                        v_metrics = await run_vllm_requests(
                            runtime=runtime,
                            count_tokens=count_tokens,
                            req_count=req_count,
                            concurrency=c,
                            print_stream=bool(args.print_stream),
                            print_stream_prefix=bool(args.print_stream_prefix),
                        )
                        level_metrics.extend(v_metrics)
                        vllm_after += len(v_metrics)
                        mark_first_char(v_metrics)
                        backend_name = "switch(mixed-level)"
                        break
            else:
                req_count = c * args.rounds_per_level
                release_llama("vllm phase")
                progress(args, f"switch: level c={c} using vllm phase req_count={req_count}")
                level_metrics = await run_vllm_requests(
                    runtime=runtime,
                    count_tokens=count_tokens,
                    req_count=req_count,
                    concurrency=c,
                    print_stream=bool(args.print_stream),
                    print_stream_prefix=bool(args.print_stream_prefix),
                )
                vllm_after += len(level_metrics)
                mark_first_char(level_metrics)
                backend_name = "switch(vllm-phase)"

            wall_s = time.perf_counter() - t0
            all_metrics.extend(level_metrics)
            level_results.append(aggregate_level(backend_name, c, level_metrics, wall_s))
            progress(
                args,
                f"switch: level c={c} done backend={backend_name} req={len(level_metrics)} "
                f"wall_s={fmt(wall_s)}",
            )

        if runtime is None and vllm_task is not None:
            # vLLM not ready before the traffic ended; still await for cold-start visibility.
            try:
                runtime = await vllm_task
                progress(args, f"switch: vllm finished after traffic cold_start_s={fmt(runtime.cold_start_s)}")
            except Exception:
                vllm_failed = True
                if not vllm_fail_reason:
                    vllm_fail_reason = "vLLM startup failed"
                runtime = None

        total_wall = time.perf_counter() - wall_start
        first_char_s = first_token_delay_from_start(all_metrics, wall_start)
        scenario = aggregate_scenario(
            backend="hybrid-switch",
            cold_start_s=first_char_s,
            metrics=all_metrics,
            wall_s=total_wall,
            cutover_concurrency=cutover_level,
            llama_count=llama_before,
            vllm_count=vllm_after,
        )
        if runtime is None:
            tail = "vLLM never became ready during run"
            if vllm_failed and vllm_fail_reason:
                tail = f"vLLM failed: {vllm_fail_reason}"
            scenario.error = f"{scenario.error + ' | ' if scenario.error else ''}{tail}"
        return level_results, scenario
    finally:
        if llama_running:
            llama_pool.stop()
            progress(args, "switch: llama stopped")
        if vllm_task is not None and not vllm_task.done():
            vllm_task.cancel()
            try:
                await vllm_task
            except Exception:
                pass
        await stop_vllm_runtime(runtime)
        progress(args, "switch: vllm stopped")


def print_level_report(all_levels: List[LevelResult]) -> None:
    print("\n=== Level Report (Concurrency Ramp) ===")
    print(
        f"{'backend':<24} {'conc':>4} {'req':>6} {'succ':>6} {'fail':>6} "
        f"{'wall_s':>9} {'rps':>9} {'tps':>9} {'tok_rate':>10} "
        f"{'ttft_p50':>10} {'ttft_p95':>10} {'e2e_p50':>10} {'e2e_p95':>10}"
    )
    for r in all_levels:
        print(
            f"{r.backend:<24} {r.concurrency:>4} {r.total_requests:>6} {r.succeeded_requests:>6} {r.failed_requests:>6} "
            f"{fmt(r.wall_time_s):>9} {fmt(r.throughput_rps):>9} {fmt(r.throughput_tps):>9} {fmt(r.token_rate_avg_tps):>10} "
            f"{fmt(r.ttft_p50_s):>10} {fmt(r.ttft_p95_s):>10} {fmt(r.e2e_p50_s):>10} {fmt(r.e2e_p95_s):>10}"
        )
        if r.error:
            print(f"  error: {r.error}")


def print_scenario_report(scenarios: List[ScenarioResult]) -> None:
    print("\n=== Scenario Summary ===")
    print(
        f"{'backend':<24} {'cold_s':>9} {'req':>6} {'succ':>6} {'fail':>6} "
        f"{'wall_s':>9} {'rps':>9} {'tps':>9} {'tok_rate':>10} "
        f"{'ttft_p50':>10} {'ttft_p95':>10} {'e2e_p50':>10} {'e2e_p95':>10}"
    )
    for s in scenarios:
        print(
            f"{s.backend:<24} {fmt(s.cold_start_s):>9} {s.total_requests:>6} {s.succeeded_requests:>6} {s.failed_requests:>6} "
            f"{fmt(s.wall_time_s):>9} {fmt(s.throughput_rps):>9} {fmt(s.throughput_tps):>9} {fmt(s.token_rate_avg_tps):>10} "
            f"{fmt(s.ttft_p50_s):>10} {fmt(s.ttft_p95_s):>10} {fmt(s.e2e_p50_s):>10} {fmt(s.e2e_p95_s):>10}"
        )
        if s.backend == "hybrid-switch":
            print(
                f"  cutover_conc={s.cutover_concurrency} "
                f"llama_before={s.llama_requests_before_cutover} "
                f"vllm_after={s.vllm_requests_after_cutover}"
            )
        if s.error:
            print(f"  error: {s.error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid switch benchmark: llama.cpp <-> vLLM.")
    parser.add_argument(
        "--show-progress",
        type=int,
        default=1,
        choices=[0, 1],
        help="Print per-stage/per-level progress logs.",
    )
    parser.add_argument(
        "--text-only",
        type=int,
        default=0,
        choices=[0, 1],
        help="Run pure text benchmark; default is multimodal (image+text).",
    )
    parser.add_argument(
        "--skip-hybrid",
        type=int,
        default=0,
        choices=[0, 1],
        help="Skip hybrid switch phase; only run pure llama.cpp and pure vLLM baselines.",
    )
    parser.add_argument("--max-concurrency", type=int, default=8, help="Ramp from 1..max_concurrency.")
    parser.add_argument(
        "--rounds-per-level",
        type=int,
        default=4,
        help="Per concurrency level, run this many rounds; each round sends <concurrency> requests.",
    )

    parser.add_argument("--prompt", default="Describe this image in one sentence.")
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are a highly accurate vision-language assistant. "
            "Provide detailed, precise, and well-structured image descriptions."
        ),
    )
    parser.add_argument("--image-path", default=base.DEFAULT_SINGLE_IMAGE_PATH)
    parser.add_argument("--image-url", default=None)

    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=2048)
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
        "--llama-max-instances",
        type=int,
        default=1,
        help="Pure llama baseline instance count (multiple llama.cpp).",
    )
    parser.add_argument(
        "--switch-llama-instances",
        type=int,
        default=2,
        help="llama.cpp instances used during hybrid switch llama phase.",
    )

    parser.add_argument("--vllm-model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--vllm-model-path", default=None)
    parser.add_argument("--vllm-dtype", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="vLLM gpu_memory_utilization for Qwen3-VL-2B runs.",
    )

    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


async def _main_async(args: argparse.Namespace) -> int:
    count_tokens = base.build_token_counter(args.tokenizer_model)

    print("=== Pure llama.cpp baseline (multiple instances) ===")
    llama_levels, llama_scenario = run_pure_llama(args, count_tokens)

    print("\n=== Pure vLLM baseline ===")
    vllm_levels, vllm_scenario = await run_pure_vllm(args, count_tokens) 

    if bool(args.skip_hybrid):
        all_levels = llama_levels + vllm_levels
        scenarios = [llama_scenario, vllm_scenario]
    else:
        print(
            f"\n=== Hybrid switch test ({args.switch_llama_instances} llama + 1 vLLM startup) ==="
        )
        switch_levels, switch_scenario = await run_hybrid_switch(args, count_tokens)
        all_levels = llama_levels + vllm_levels + switch_levels
        scenarios = [llama_scenario, vllm_scenario, switch_scenario]

    print_level_report(all_levels)
    print_scenario_report(scenarios)

    if args.output_json:
        payload = {
            "args": vars(args),
            "levels": [asdict(x) for x in all_levels],
            "scenarios": [asdict(x) for x in scenarios],
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\njson report saved to: {args.output_json}")

    return 1 if any(x.error for x in scenarios) else 0


def main() -> int:
    args = parse_args()
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be >= 1")
    if args.rounds_per_level < 1:
        raise ValueError("--rounds-per-level must be >= 1")
    if args.llama_max_instances < 1:
        raise ValueError("--llama-max-instances must be >= 1")
    if args.switch_llama_instances < 1:
        raise ValueError("--switch-llama-instances must be >= 1")
    if args.llama_max_instances < args.max_concurrency:
        print(
            f"[WARN] llama_max_instances({args.llama_max_instances}) < max_concurrency({args.max_concurrency}). "
            "Pure llama baseline may saturate early."
        )
    if bool(args.text_only) and (args.image_path or args.image_url):
        print("[INFO] --text-only=1, image inputs will be ignored.")
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
