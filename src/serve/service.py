"""Ray Serve Router+Manager Service

将 Router 与 Manager 合并为一个 Ray Serve 服务（单一 actor），
对接外部 Worker 的 vLLM 实例（HTTP 方式）。

默认网络可连通，不做心跳保活逻辑，减少控制面复杂度。
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import ray
from ray import serve

from src.serve.autoscaler import LoadBasedAutoscaler

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

BACKEND_VLLM = "vllm"
BACKEND_LLAMA_CPP = "llama.cpp"
SUPPORTED_BACKENDS = {BACKEND_VLLM, BACKEND_LLAMA_CPP}

ROUTING_COLD = "COLD"
ROUTING_FAST_ONLY = "FAST_ONLY"
ROUTING_WARMING_VLLM = "WARMING_VLLM"
ROUTING_MIXED = "MIXED"
ROUTING_VLLM_PRIMARY = "VLLM_PRIMARY"
ROUTING_DEGRADED_FAST = "DEGRADED_FAST"
SUPPORTED_ROUTING_STATES = {
    ROUTING_COLD,
    ROUTING_FAST_ONLY,
    ROUTING_WARMING_VLLM,
    ROUTING_MIXED,
    ROUTING_VLLM_PRIMARY,
    ROUTING_DEGRADED_FAST,
}


class WorkerRegistryCore:
    """管理外部 Worker 与 vLLM 实例的注册与路由信息。"""

    def __init__(
        self,
        healthcheck_interval: float = 15.0,
        healthcheck_timeout: float = 3.0,
        max_failures: int = 3,
        instance_capacity: int = 4,
        min_replicas: int = 1,
        max_replicas: int = 8,
        scale_interval: float = 10.0,
    ) -> None:
        self.workers: Dict[str, Dict] = {}
        self.model_instances: Dict[str, list] = {}
        self.model_configs: Dict[str, Dict] = {}
        self.failure_counts: Dict[str, int] = {}
        self.instance_capacity = max(1, int(instance_capacity))
        self._worker_replica_counters: Dict[str, int] = {}
        self._rr_counters: Dict[str, int] = {}
        self._lock = asyncio.Lock()
        self.healthcheck_interval = healthcheck_interval
        self.healthcheck_timeout = healthcheck_timeout
        self.max_failures = max_failures
        self._health_task = asyncio.create_task(self._health_loop())
        scale_up_latency_threshold = float(os.getenv("SCALE_UP_LATENCY_THRESHOLD", "5"))
        scale_down_latency_threshold = float(os.getenv("SCALE_DOWN_LATENCY_THRESHOLD", "2"))
        scale_up_cooldown_s = float(os.getenv("SCALE_UP_COOLDOWN", "30"))
        scale_down_cooldown_s = float(os.getenv("SCALE_DOWN_COOLDOWN", "30"))
        latency_sample_window_s = float(os.getenv("SCALE_LATENCY_WINDOW_S", "30"))
        self.scale_up_load_threshold = float(os.getenv("SCALE_UP_LOAD_THRESHOLD", "0.7"))
        self.scale_down_load_threshold = float(os.getenv("SCALE_DOWN_LOAD_THRESHOLD", "0.2"))
        self.autoscaler_enabled = os.getenv("AUTOSCALER_ENABLED", "1") == "1"
        baseline_latency_multiplier = float(os.getenv("SCALE_BASELINE_MULTIPLIER", "2.0"))
        baseline_max_tokens = int(os.getenv("SCALE_BASELINE_MAX_TOKENS", "16"))
        baseline_timeout_s = float(os.getenv("SCALE_BASELINE_TIMEOUT_S", "30"))
        self.autoscaler: Optional[LoadBasedAutoscaler] = None
        if self.autoscaler_enabled:
            self.autoscaler = LoadBasedAutoscaler(
                instance_capacity=self.instance_capacity,
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                check_interval=scale_interval,
                health_timeout=healthcheck_timeout,
                scale_up_latency_threshold=scale_up_latency_threshold,
                scale_down_latency_threshold=scale_down_latency_threshold,
                scale_up_cooldown_s=scale_up_cooldown_s,
                scale_down_cooldown_s=scale_down_cooldown_s,
                latency_sample_window_s=latency_sample_window_s,
                baseline_latency_multiplier=baseline_latency_multiplier,
                baseline_max_tokens=baseline_max_tokens,
                baseline_timeout_s=baseline_timeout_s,
            )
            self.autoscaler.start(self)
        else:
            logger.info("Autoscaler disabled (AUTOSCALER_ENABLED=0): routing-only mode")

    async def register_worker(
        self,
        worker_id: str,
        worker_url: str,
        public_worker_url: Optional[str] = None,
        gpu_info: Optional[Dict] = None,
    ) -> Dict:
        if not worker_id or not worker_url:
            return {"status": "error", "message": "worker_id and worker_url are required"}

        async with self._lock:
            self.workers[worker_id] = {
                "worker_id": worker_id,
                "worker_url": worker_url.rstrip("/"),
                "public_worker_url": public_worker_url.rstrip("/") if public_worker_url else None,
                "gpu_info": gpu_info or {},
                "registered_at": time.time(),
            }
            self.failure_counts[worker_id] = 0

        return {"status": "success", "message": f"Worker {worker_id} registered"}

    async def unregister_worker(self, worker_id: str) -> Dict:
        async with self._lock:
            return self._unregister_worker_locked(worker_id)

    async def list_workers(self) -> Dict:
        return {"workers": list(self.workers.values())}

    async def list_models(self) -> Dict:
        await self._prune_orphan_instances()
        return {
            "models": [
                {
                    "alias": alias,
                    "instances": list(instances),
                    "config": self.model_configs.get(alias, {}),
                }
                for alias, instances in self.model_instances.items()
            ]
        }

    async def get_model_routing(self, alias: str) -> Optional[Dict]:
        instances = self.model_instances.get(alias, [])
        if not instances:
            return None
        return instances[0]

    def _pick_active_instance(self, alias: str, active: List[Dict]) -> Dict:
        idx = self._rr_counters.get(alias, 0)
        if idx >= len(active):
            idx = 0
        self._rr_counters[alias] = (idx + 1) % len(active)
        return active[idx]

    @staticmethod
    def _normalize_routing_state(value: Optional[str]) -> str:
        state = (value or ROUTING_FAST_ONLY).strip().upper()
        if state not in SUPPORTED_ROUTING_STATES:
            raise ValueError(f"unsupported routing_state: {state}")
        return state

    @staticmethod
    def _normalize_backend_type(value: Optional[str]) -> str:
        backend = (value or "").strip()
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"unsupported backend_type: {backend}")
        return backend

    @staticmethod
    def _default_backend_payload(backend_type: str) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"backend_type": backend_type}
        if backend_type == BACKEND_VLLM:
            payload["tensor_parallel_size"] = 1
        return payload

    def _validate_backends_config(self, backends: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not isinstance(backends, dict) or not backends:
            raise ValueError("backends is required and must be a non-empty object")

        normalized: Dict[str, Dict[str, Any]] = {}
        for backend_type, cfg in backends.items():
            backend = self._normalize_backend_type(backend_type)
            if not isinstance(cfg, dict):
                raise ValueError(f"backend config for {backend} must be an object")
            model_name = cfg.get("model_name")
            if not model_name or not isinstance(model_name, str):
                raise ValueError(f"backends.{backend}.model_name is required")

            merged = self._default_backend_payload(backend)
            merged.update(cfg)
            merged["backend_type"] = backend
            if "min_replicas" in merged and merged["min_replicas"] is not None:
                merged["min_replicas"] = max(0, int(merged["min_replicas"]))
            if "max_replicas" in merged and merged["max_replicas"] is not None:
                merged["max_replicas"] = max(0, int(merged["max_replicas"]))
            normalized[backend] = merged

        return normalized

    @staticmethod
    def _primary_backend_by_state(routing_state: str) -> str:
        if routing_state in {ROUTING_FAST_ONLY, ROUTING_WARMING_VLLM, ROUTING_DEGRADED_FAST}:
            return BACKEND_LLAMA_CPP
        if routing_state == ROUTING_VLLM_PRIMARY:
            return BACKEND_VLLM
        if routing_state == ROUTING_MIXED:
            return BACKEND_VLLM
        return BACKEND_LLAMA_CPP

    def _choose_backend_for_scale(self, config: Dict[str, Any]) -> Optional[str]:
        backends = config.get("backends") or {}
        if not backends:
            return None
        routing_state = self._normalize_routing_state(config.get("routing_state"))
        preferred = self._primary_backend_by_state(routing_state)
        if preferred in backends:
            return preferred
        return next(iter(backends.keys()), None)

    def _sleep_level_value(self, instance: Dict) -> int:
        backend_type = instance.get("backend_type")
        if backend_type == BACKEND_LLAMA_CPP:
            return 0
        value = instance.get("sleep_level_value")
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
        level_name = instance.get("sleep_level")
        if isinstance(level_name, str):
            mapping = {
                "ACTIVE": 0,
                "SLEEP_1": 1,
                "SLEEP_2": 2,
                "UNLOADED": 3,
            }
            return mapping.get(level_name.upper(), 3)
        return 3

    def _is_instance_ready(self, instance: Dict) -> bool:
        status = str(instance.get("status", "")).lower()
        if status != "running":
            return False
        backend_type = instance.get("backend_type")
        if backend_type == BACKEND_LLAMA_CPP:
            return True
        return self._sleep_level_value(instance) == 0

    async def _probe_instance(self, instance: Dict) -> Optional[Dict]:
        worker_url = instance.get("control_url") or instance.get("worker_url")
        inst_alias = instance.get("instance_alias")
        if not worker_url or not inst_alias:
            return None
        async with httpx.AsyncClient(timeout=self.healthcheck_timeout) as client:
            resp = await client.get(f"{worker_url}/instances/{inst_alias}/status")
            if resp.status_code != 200:
                return None
            status = resp.json()
        inflight = int(status.get("inflight_requests", 0))
        sleep_level = int(status.get("sleep_level_value", 0))
        e2e_avg = status.get("e2e_avg")
        e2e_last = status.get("e2e_last")
        capacity = status.get("capacity") or self.instance_capacity
        capacity = max(1, int(capacity))
        load = min(inflight / capacity, 1.0)
        async with self._lock:
            instance["inflight_requests"] = inflight
            instance["sleep_level_value"] = sleep_level
            instance["load"] = load
            instance["capacity"] = capacity
            instance["e2e_avg"] = e2e_avg
            instance["e2e_last"] = e2e_last
            if "backend_type" in status:
                instance["backend_type"] = status["backend_type"]
            if "status" in status:
                instance["status"] = status["status"]
        return instance

    async def resolve_instance_for_request(self, alias: str) -> tuple[Optional[Dict], str]:
        await self._prune_orphan_instances()
        async with self._lock:
            instances = list(self.model_instances.get(alias, []))
            config = dict(self.model_configs.get(alias, {}))
            if not instances:
                return None, "not_found"

            if self.autoscaler_enabled:
                active = [i for i in instances if i.get("active")]
            else:
                # Routing-only mode: do not depend on autoscaler-maintained active flags.
                active = list(instances)
            if not active:
                return None, "not_ready"

            routing_state = self._normalize_routing_state(config.get("routing_state"))
            mix_weight = int(config.get("mix_weight", 50) or 50)
            mix_weight = max(0, min(100, mix_weight))

            fast_active = [i for i in active if i.get("backend_type") == BACKEND_LLAMA_CPP]
            slow_active = [i for i in active if i.get("backend_type") == BACKEND_VLLM]

            candidate_pools: List[tuple[str, List[Dict[str, Any]]]] = []
            if routing_state in {ROUTING_FAST_ONLY, ROUTING_WARMING_VLLM, ROUTING_DEGRADED_FAST}:
                candidate_pools = [("fast", fast_active), ("slow", slow_active), ("all", active)]
            elif routing_state == ROUTING_VLLM_PRIMARY:
                candidate_pools = [("slow", slow_active), ("fast", fast_active), ("all", active)]
            elif routing_state == ROUTING_MIXED:
                choose_slow = random.random() < (mix_weight / 100.0)
                if choose_slow:
                    candidate_pools = [("slow", slow_active), ("fast", fast_active), ("all", active)]
                else:
                    candidate_pools = [("fast", fast_active), ("slow", slow_active), ("all", active)]
            else:
                candidate_pools = [("all", active)]

        for tag, pool in candidate_pools:
            if not pool:
                continue
            # Try the whole pool to avoid missing a ready instance by chance.
            for _ in range(len(pool)):
                candidate = self._pick_active_instance(f"{alias}:{tag}", pool)
                await self._probe_instance(candidate)
                if self._is_instance_ready(candidate):
                    if not self.autoscaler_enabled:
                        async with self._lock:
                            candidate["active"] = True
                            candidate["pending_active"] = False
                    return candidate, "ready"
        return None, "not_ready"

    def _iter_worker_devices(self, worker: Dict) -> List[Dict]:
        gpu_info = worker.get("gpu_info") or {}
        devices = gpu_info.get("devices") or []
        if devices:
            return devices
        total_memory = float(gpu_info.get("total_memory_gb", 0.0) or 0.0)
        available_memory = float(gpu_info.get("available_memory_gb", 0.0) or 0.0)
        if total_memory <= 0 and available_memory <= 0:
            return []
        return [
            {
                "index": 0,
                "name": "gpu0",
                "total_memory_gb": total_memory,
                "available_memory_gb": available_memory,
            }
        ]

    def _select_device(self, devices: List[Dict], required_memory_gb: Optional[float]) -> Optional[Dict]:
        if not devices:
            return None
        if required_memory_gb is None:
            return max(devices, key=lambda d: float(d.get("available_memory_gb", 0.0) or 0.0))
        candidates = [
            device
            for device in devices
            if float(device.get("available_memory_gb", 0.0) or 0.0) >= required_memory_gb
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda d: float(d.get("available_memory_gb", 0.0) or 0.0))

    def _select_worker(
        self,
        required_memory_gb: Optional[float],
        worker_id: Optional[str],
    ) -> tuple[Optional[Dict], Optional[Dict]]:
        if required_memory_gb is not None and required_memory_gb <= 0:
            required_memory_gb = None
        if worker_id:
            worker = self.workers.get(worker_id)
            if not worker:
                return None, None
            device = self._select_device(self._iter_worker_devices(worker), required_memory_gb)
            if required_memory_gb is not None and device is None:
                return None, None
            return worker, device
        if not self.workers:
            return None, None
        best_worker = None
        best_device = None
        best_available = -1.0
        for worker in self.workers.values():
            device = self._select_device(self._iter_worker_devices(worker), required_memory_gb)
            if device is None:
                continue
            available = float(device.get("available_memory_gb", 0.0) or 0.0)
            if available > best_available:
                best_available = available
                best_worker = worker
                best_device = device
        return best_worker, best_device

    def _compute_gpu_utilization(self, device: Optional[Dict], gpu_memory_gb: float) -> float:
        if device is None:
            return 0.0
        total_memory = float(device.get("total_memory_gb", 0.0) or 0.0)
        if total_memory <= 0:
            raise ValueError("worker total_memory_gb unavailable")
        utilization = gpu_memory_gb / total_memory
        return max(0.0, min(1.0, utilization))

    def _next_instance_alias(self, worker_id: str, model_alias: str) -> str:
        key = f"{worker_id}:{model_alias}"
        count = self._worker_replica_counters.get(key, 0) + 1
        self._worker_replica_counters[key] = count
        if count == 1:
            return f"{worker_id}-{model_alias}"
        return f"{worker_id}-{model_alias}-r{count}"

    async def _start_instance_on_worker(self, worker_url: str, instance_alias: str, payload: Dict) -> Dict:
        new_payload = dict(payload)
        new_payload["alias"] = instance_alias
        async with httpx.AsyncClient(timeout=1800.0) as client:
            response = await client.post(f"{worker_url}/instances/start", json=new_payload)
            response.raise_for_status()
            return response.json().get("instance", {})

    async def _wake_instance(self, instance: Dict) -> None:
        worker_url = instance.get("control_url") or instance.get("worker_url")
        inst_alias = instance.get("instance_alias")
        if not worker_url or not inst_alias:
            return
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(f"{worker_url}/instances/{inst_alias}/wake")

    async def _sleep_instance(self, instance: Dict, level: int) -> None:
        worker_url = instance.get("control_url") or instance.get("worker_url")
        inst_alias = instance.get("instance_alias")
        if not worker_url or not inst_alias:
            return
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(f"{worker_url}/instances/{inst_alias}/sleep", json={"level": level})

    async def _start_new_instance(
        self,
        model_alias: str,
        config: Dict[str, Any],
        *,
        backend_type: Optional[str] = None,
        worker_id: Optional[str] = None,
    ) -> None:
        backends = config.get("backends") or {}
        selected_backend = backend_type or self._choose_backend_for_scale(config)
        if not selected_backend:
            return
        backend_cfg = backends.get(selected_backend)
        if not isinstance(backend_cfg, dict):
            return

        gpu_memory_gb = backend_cfg.get("gpu_memory_gb")
        fake = bool(backend_cfg.get("fake"))
        if not fake and selected_backend == BACKEND_VLLM and gpu_memory_gb is None:
            return

        worker, device = self._select_worker(gpu_memory_gb, worker_id)
        if not worker and fake:
            worker = self.workers.get(worker_id) if worker_id else next(iter(self.workers.values()), None)
            device = None
        if not worker:
            return

        payload: Dict[str, Any] = {
            "backend_type": selected_backend,
            "model_name": backend_cfg.get("model_name"),
            "model_path": backend_cfg.get("model_path"),
            "max_model_len": backend_cfg.get("max_model_len"),
            "fake": backend_cfg.get("fake"),
            "fake_response": backend_cfg.get("fake_response"),
            "fake_delay": backend_cfg.get("fake_delay"),
            "fake_delay_ms": backend_cfg.get("fake_delay_ms"),
            "fake_capacity": backend_cfg.get("fake_capacity"),
        }

        if selected_backend == BACKEND_VLLM:
            payload["tensor_parallel_size"] = backend_cfg.get("tensor_parallel_size", 1)
            payload["gpu_memory_utilization"] = self._compute_gpu_utilization(
                device,
                float(gpu_memory_gb or 0.0),
            )
        else:
            payload["llama_filename"] = backend_cfg.get("llama_filename")
            payload["llama_mmproj_path"] = backend_cfg.get("llama_mmproj_path")
            payload["llama_n_gpu_layers"] = backend_cfg.get("llama_n_gpu_layers", -1)

        control_url = worker["worker_url"]
        instance_alias = self._next_instance_alias(worker["worker_id"], model_alias)
        instance_info = await self._start_instance_on_worker(
            worker_url=control_url,
            instance_alias=instance_alias,
            payload=payload,
        )

        route_url = worker.get("public_worker_url") or worker.get("worker_url")
        instance = {
            "model_alias": model_alias,
            "instance_alias": instance_alias,
            "backend_type": selected_backend,
            "model_name": backend_cfg.get("model_name"),
            "worker_id": worker["worker_id"],
            "worker_url": route_url,
            "control_url": control_url,
            "vllm_port": instance_info.get("port", 0),
            "status": instance_info.get("status"),
            "created_at": time.time(),
            "sleep_level_value": 0,
            "inflight_requests": 0,
            "load": 0.0,
            "capacity": instance_info.get("capacity") or backend_cfg.get("fake_capacity"),
            "active": not self.autoscaler_enabled,
            "pending_active": self.autoscaler_enabled,
        }
        async with self._lock:
            self.model_instances.setdefault(model_alias, []).append(instance)

    async def register_model(
        self,
        alias: str,
        backends: Dict[str, Dict[str, Any]],
        routing_state: str = ROUTING_FAST_ONLY,
        mix_weight: int = 50,
        worker_id: Optional[str] = None,
    ) -> Dict:
        if not alias:
            return {"status": "error", "message": "alias is required"}

        try:
            normalized_backends = self._validate_backends_config(backends)
            normalized_state = self._normalize_routing_state(routing_state)
            normalized_mix_weight = max(0, min(100, int(mix_weight)))
        except (ValueError, TypeError) as exc:
            return {"status": "error", "message": str(exc)}

        primary_backend = self._primary_backend_by_state(normalized_state)
        if primary_backend not in normalized_backends:
            if normalized_state in {ROUTING_FAST_ONLY, ROUTING_WARMING_VLLM, ROUTING_DEGRADED_FAST}:
                return {"status": "error", "message": f"routing_state={normalized_state} requires backend {primary_backend}"}
            primary_backend = next(iter(normalized_backends.keys()))

        async with self._lock:
            if alias in self.model_instances:
                return {
                    "status": "exists",
                    "message": f"Model {alias} already registered",
                    "routing": self.model_instances[alias][0],
                }
            self.model_configs[alias] = {
                "alias": alias,
                "routing_state": normalized_state,
                "mix_weight": normalized_mix_weight,
                "backends": normalized_backends,
            }

        started = 0
        try:
            initial_backend = self._primary_backend_by_state(normalized_state)
            if initial_backend not in normalized_backends:
                initial_backend = primary_backend
            initial_cfg = normalized_backends.get(initial_backend, {})
            initial_replicas = max(1, int(initial_cfg.get("min_replicas", 1)))
            for _ in range(initial_replicas):
                before = len(self.model_instances.get(alias, []))
                await self._start_new_instance(
                    alias,
                    self.model_configs[alias],
                    backend_type=initial_backend,
                    worker_id=worker_id,
                )
                after = len(self.model_instances.get(alias, []))
                if after > before:
                    started += 1

            if started == 0:
                return {
                    "status": "error",
                    "message": f"Failed to start initial instances for {alias}",
                }

            async with self._lock:
                routing = self.model_instances.get(alias, [{}])[0]
            return {
                "status": "success",
                "message": f"Model {alias} registered with {started} instance(s)",
                "routing": routing,
            }
        except Exception as exc:
            logger.error("Failed to register model %s: %s", alias, exc)
            async with self._lock:
                self.model_instances.pop(alias, None)
                self.model_configs.pop(alias, None)
            return {"status": "error", "message": f"Failed to register model: {exc}"}

    async def unregister_model(self, alias: str) -> Dict:
        instances = self.model_instances.get(alias)
        if not instances:
            return {"status": "error", "message": f"Model {alias} not found"}

        for instance in list(instances):
            worker_id = instance.get("worker_id")
            worker = self.workers.get(worker_id) if worker_id else None
            control_url = worker["worker_url"] if worker else instance.get("worker_url")

            if control_url:
                try:
                    async with httpx.AsyncClient(timeout=30.0) as client:
                        await client.post(f"{control_url}/instances/{instance['instance_alias']}/stop")
                except Exception as exc:
                    logger.warning("Failed to stop instance %s: %s", instance["instance_alias"], exc)

        async with self._lock:
            self.model_instances.pop(alias, None)
            self.model_configs.pop(alias, None)
            # per-worker replica counters are kept; no per-model cleanup needed

        return {"status": "success", "message": f"Model {alias} unregistered"}

    async def health(self) -> Dict:
        await self._prune_orphan_instances()
        return {
            "status": "ok",
            "service": "serve",
            "workers": len(self.workers),
            "models": len(self.model_instances),
            "autoscaler_enabled": self.autoscaler_enabled,
        }

    async def _prune_orphan_instances(self) -> None:
        async with self._lock:
            if not self.model_instances:
                return
            worker_ids = set(self.workers.keys())
            aliases_to_remove = []
            for alias, instances in list(self.model_instances.items()):
                kept = [inst for inst in instances if inst.get("worker_id") in worker_ids]
                if kept:
                    if len(kept) != len(instances):
                        self.model_instances[alias] = kept
                else:
                    aliases_to_remove.append(alias)
            for alias in aliases_to_remove:
                self.model_instances.pop(alias, None)
                self.model_configs.pop(alias, None)
            # per-worker replica counters are kept; no per-model cleanup needed

    def _unregister_worker_locked(self, worker_id: str) -> Dict:
        if worker_id not in self.workers:
            return {"status": "error", "message": f"Worker {worker_id} not found"}

        aliases_to_remove = [
            alias for alias, instances in self.model_instances.items()
            if any(inst.get("worker_id") == worker_id for inst in instances)
        ]
        for alias in aliases_to_remove:
            instances = [
                inst for inst in self.model_instances.get(alias, [])
                if inst.get("worker_id") != worker_id
            ]
            if instances:
                self.model_instances[alias] = instances
            else:
                self.model_instances.pop(alias, None)
                self.model_configs.pop(alias, None)
            # per-worker replica counters are kept; no per-model cleanup needed

        self.workers.pop(worker_id, None)
        self.failure_counts.pop(worker_id, None)
        return {"status": "success", "message": f"Worker {worker_id} unregistered"}

    async def _ping_worker(self, worker_url: str) -> Optional[Dict]:
        async with httpx.AsyncClient(timeout=self.healthcheck_timeout) as client:
            response = await client.get(f"{worker_url}/health")
            if response.status_code == 200:
                return response.json()
            return None

    async def _health_loop(self):
        while True:
            await asyncio.sleep(self.healthcheck_interval)
            if not self.workers:
                continue
            worker_ids = list(self.workers.keys())
            for worker_id in worker_ids:
                worker = self.workers.get(worker_id)
                if not worker:
                    continue
                worker_url = worker.get("worker_url")
                if not worker_url:
                    continue
                ok = False
                health = None
                try:
                    health = await self._ping_worker(worker_url)
                    ok = health is not None
                except Exception as exc:
                    logger.warning("Worker health check failed: %s (%s)", worker_id, exc)
                async with self._lock:
                    if worker_id not in self.workers:
                        continue
                    if ok:
                        self.failure_counts[worker_id] = 0
                        if health and "gpu_info" in health:
                            self.workers[worker_id]["gpu_info"] = health["gpu_info"]
                        continue
                    self.failure_counts[worker_id] = self.failure_counts.get(worker_id, 0) + 1
                    failures = self.failure_counts[worker_id]
                    if failures >= self.max_failures:
                        logger.warning(
                            "Worker %s unhealthy (%s/%s). Unregistering.",
                            worker_id,
                            failures,
                            self.max_failures,
                        )
                        self._unregister_worker_locked(worker_id)


def _filter_request_headers(headers: Dict) -> Dict:
    return {k: v for k, v in headers.items() if k.lower() not in ["host", "content-length"]}


def _filter_response_headers(headers: httpx.Headers) -> Dict:
    filtered = {}
    for k, v in headers.items():
        if k.lower() in ["content-length", "transfer-encoding", "connection"]:
            continue
        filtered[k] = v
    return filtered


def _model_not_ready_response(model: str, status: str = "starting", retry_after: int = 5) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "message": f"Model {model} is starting or sleeping, please retry later.",
                "type": "model_loading",
                "code": "model_not_ready",
                "status": status,
                "retry_after": retry_after,
            }
        },
        headers={"Retry-After": str(retry_after)},
    )

fastapi_app = FastAPI(title="LLM Router+Manager (Ray Serve)")
_MAX_ONGOING = int(os.getenv("SERVE_MAX_ONGOING_REQUESTS", "100"))


@serve.deployment(max_ongoing_requests=_MAX_ONGOING)
@serve.ingress(fastapi_app)
class RouterManagerServe(WorkerRegistryCore):
    def __init__(
        self,
        healthcheck_interval: float = 15.0,
        healthcheck_timeout: float = 3.0,
        max_failures: int = 3,
        instance_capacity: int = 4,
        min_replicas: int = 1,
        max_replicas: int = 8,
        scale_interval: float = 10.0,
    ):
        super().__init__(
            healthcheck_interval=healthcheck_interval,
            healthcheck_timeout=healthcheck_timeout,
            max_failures=max_failures,
            instance_capacity=instance_capacity,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            scale_interval=scale_interval,
        )
        self.request_timeout = float(os.getenv("ROUTER_REQUEST_TIMEOUT", "300"))

    @fastapi_app.get("/health")
    async def http_health(self):
        return await self.health()

    @fastapi_app.post("/v1/chat/completions")
    async def chat_completions(self, request: Request):
        body = await request.json()
        model = body.get("model")
        if not model:
            raise HTTPException(status_code=400, detail="model is required")

        routing, status = await self.resolve_instance_for_request(model)
        if status == "not_found":
            raise HTTPException(status_code=404, detail=f"Model {model} not found")
        if status != "ready":
            return _model_not_ready_response(model)

        target_url = f"{routing['worker_url']}/proxy/{routing['instance_alias']}/v1/chat/completions"
        headers = _filter_request_headers(dict(request.headers))
        is_stream = body.get("stream", False)

        try:
            if is_stream:
                client = httpx.AsyncClient(timeout=self.request_timeout)
                request = client.build_request(
                    "POST",
                    target_url,
                    json=body,
                    headers=headers,
                )
                response = await client.send(request, stream=True)
                if response.status_code >= 400:
                    detail = await response.aread()
                    await response.aclose()
                    await client.aclose()
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=detail.decode("utf-8", errors="ignore"),
                    )

                async def stream_generator():
                    try:
                        async for chunk in response.aiter_bytes():
                            yield chunk
                    except asyncio.CancelledError:
                        logger.info("Client disconnected during stream")
                    except Exception as exc:
                        logger.warning("Upstream stream interrupted: %s", exc)
                    finally:
                        await response.aclose()
                        await client.aclose()

                return StreamingResponse(
                    stream_generator(),
                    status_code=response.status_code,
                    media_type="text/event-stream",
                    headers=_filter_response_headers(response.headers),
                )

            async with httpx.AsyncClient(timeout=self.request_timeout) as client:
                response = await client.post(target_url, json=body, headers=headers)
                if response.headers.get("content-type", "").startswith("application/json"):
                    return JSONResponse(content=response.json(), status_code=response.status_code)
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=_filter_response_headers(response.headers),
                )
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Request timed out")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Error forwarding request: %s", exc)
            raise HTTPException(status_code=502, detail=f"Error forwarding request: {exc}")

    @fastapi_app.get("/v1/models")
    async def openai_list_models(self):
        result = await self.list_models()
        routes = result.get("models", [])
        data = []
        for route in routes:
            instances = route.get("instances", [])
            created_at = None
            owned_by = "external"
            if instances:
                created_at = instances[0].get("created_at")
                owned_by = instances[0].get("worker_id", owned_by)
            created = int(created_at) if created_at else int(time.time())
            data.append(
                {
                    "id": route.get("alias"),
                    "object": "model",
                    "created": created,
                    "owned_by": owned_by,
                }
            )
        return {"object": "list", "data": data}

    @fastapi_app.post("/admin/models/register")
    async def admin_register_model(self, request: Request):
        body = await request.json()
        return await self.register_model(
            alias=body.get("alias"),
            backends=body.get("backends"),
            routing_state=body.get("routing_state", ROUTING_FAST_ONLY),
            mix_weight=body.get("mix_weight", 50),
            worker_id=body.get("worker_id"),
        )

    @fastapi_app.delete("/admin/models/{alias}")
    async def admin_unregister_model(self, alias: str):
        return await self.unregister_model(alias)

    @fastapi_app.get("/admin/models")
    async def admin_list_models(self):
        return await self.list_models()

    @fastapi_app.post("/admin/models/{alias}/wake")
    async def admin_wake_model(self, alias: str):
        instances = self.model_instances.get(alias, [])
        if not instances:
            raise HTTPException(status_code=404, detail=f"Model {alias} not found")
        sleeping = [i for i in instances if i.get("sleep_level_value", 0) > 0]
        if not sleeping:
            return {"status": "success", "message": "no sleeping instances", "woken": False}
        candidate = min(sleeping, key=lambda i: i.get("sleep_level_value", 3))
        await self._wake_instance(candidate)
        return {
            "status": "success",
            "message": f"woke instance {candidate.get('instance_alias')}",
            "instance_alias": candidate.get("instance_alias"),
            "woken": True,
        }

    @fastapi_app.get("/admin/workers")
    async def admin_list_workers(self):
        return await self.list_workers()

    @fastapi_app.get("/admin/status")
    async def admin_status(self):
        return {
            "health": await self.health(),
            "models": await self.list_models(),
            "workers": await self.list_workers(),
            "autoscaler_enabled": self.autoscaler_enabled,
        }

    @fastapi_app.get("/workers")
    async def public_list_workers(self):
        return await self.list_workers()

    @fastapi_app.post("/workers/register")
    async def http_register_worker(self, request: Request):
        body = await request.json()
        return await self.register_worker(
            worker_id=body.get("worker_id"),
            worker_url=body.get("worker_url"),
            public_worker_url=body.get("public_worker_url"),
            gpu_info=body.get("gpu_info"),
        )

    @fastapi_app.delete("/workers/{worker_id}/unregister")
    async def http_unregister_worker(self, worker_id: str):
        return await self.unregister_worker(worker_id)


def _init_ray():
    if ray.is_initialized():
        return
    address = os.getenv("RAY_ADDRESS")
    working_dir = os.getenv("RAY_WORKING_DIR")
    if working_dir is None:
        working_dir = str(Path(__file__).resolve().parents[2])
    runtime_env = None
    if working_dir and working_dir.lower() != "none":
        runtime_env = {"working_dir": working_dir}

    if address:
        ray.init(address=address, runtime_env=runtime_env)
    else:
        ray.init(runtime_env=runtime_env)


def main():
    serve_host = os.getenv("SERVE_HOST", os.getenv("ROUTER_HOST", "0.0.0.0"))
    serve_port = int(os.getenv("SERVE_PORT", os.getenv("ROUTER_PORT", "8000")))

    _init_ray()

    try:
        serve.start(http_options={"host": serve_host, "port": serve_port})
    except Exception as exc:
        if "already" not in str(exc).lower():
            raise
        logger.info("Ray Serve already started: %s", exc)

    healthcheck_interval = float(os.getenv("WORKER_HEALTHCHECK_INTERVAL", "15"))
    healthcheck_timeout = float(os.getenv("WORKER_HEALTHCHECK_TIMEOUT", "3"))
    max_failures = int(os.getenv("WORKER_HEALTHCHECK_MAX_FAILURES", "3"))
    instance_capacity = int(os.getenv("MODEL_INSTANCE_CAPACITY", "4"))
    min_replicas = int(os.getenv("MODEL_MIN_REPLICAS", "1"))
    max_replicas = int(os.getenv("MODEL_MAX_REPLICAS", "8"))
    scale_interval = float(os.getenv("MODEL_SCALE_INTERVAL", "10"))

    serve.run(
        RouterManagerServe.bind(
            healthcheck_interval=healthcheck_interval,
            healthcheck_timeout=healthcheck_timeout,
            max_failures=max_failures,
            instance_capacity=instance_capacity,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            scale_interval=scale_interval,
        ),
        route_prefix="/",
    )

    logger.info("Ray Serve running on %s:%s", serve_host, serve_port)


if __name__ == "__main__":
    main()
