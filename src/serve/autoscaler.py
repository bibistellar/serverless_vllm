"""Latency-based autoscaler for vLLM instances."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional
from collections import Counter

import httpx

logger = logging.getLogger(__name__)

BACKEND_VLLM = "vllm"
BACKEND_LLAMA_CPP = "llama.cpp"

ROUTING_COLD = "COLD"
ROUTING_FAST_ONLY = "FAST_ONLY"
ROUTING_WARMING_VLLM = "WARMING_VLLM"
ROUTING_MIXED = "MIXED"
ROUTING_VLLM_PRIMARY = "VLLM_PRIMARY"
ROUTING_DEGRADED_FAST = "DEGRADED_FAST"
_ROUTING_STATES = {
    ROUTING_COLD,
    ROUTING_FAST_ONLY,
    ROUTING_WARMING_VLLM,
    ROUTING_MIXED,
    ROUTING_VLLM_PRIMARY,
    ROUTING_DEGRADED_FAST,
}


def _print(msg: str) -> None:
    print(msg, flush=True)


class LoadBasedAutoscaler:
    """Auto-scale policy based on average inference latency.

    Maintains an ACTIVE instance list for routing. Scaling decisions are made
    from recent average latency, not per-instance load.
    """

    def __init__(
        self,
        instance_capacity: int = 4,
        min_replicas: int = 1,
        max_replicas: int = 8,
        check_interval: float = 10.0,
        health_timeout: float = 3.0,
        scale_up_latency_threshold: float = 5.0,
        scale_down_latency_threshold: float = 2.0,
        scale_up_cooldown_s: float = 30.0,
        scale_down_cooldown_s: float = 30.0,
        latency_sample_window_s: float = 30.0,
        baseline_latency_multiplier: float = 2.0,
        baseline_max_tokens: int = 16,
        baseline_timeout_s: float = 30.0,
    ) -> None:
        self.instance_capacity = max(1, instance_capacity)
        self.min_replicas = max(1, min_replicas)
        self.max_replicas = max(self.min_replicas, max_replicas)
        self.check_interval = check_interval
        self.health_timeout = health_timeout
        self.scale_up_latency_threshold = scale_up_latency_threshold
        self.scale_down_latency_threshold = scale_down_latency_threshold
        self.scale_up_cooldown_s = scale_up_cooldown_s
        self.scale_down_cooldown_s = scale_down_cooldown_s
        self.latency_sample_window_s = latency_sample_window_s
        self.baseline_latency_multiplier = baseline_latency_multiplier
        self.baseline_max_tokens = baseline_max_tokens
        self.baseline_timeout_s = baseline_timeout_s
        self._last_scale_up: Dict[str, float] = {}
        self._last_scale_down: Dict[str, float] = {}
        self._scale_up_thresholds: Dict[str, float] = {}
        self._baseline_pending: set[str] = set()
        self.hybrid_prepare_conc = max(1, int(float(os.getenv("HYBRID_PREPARE_CONC", "3"))))
        self.hybrid_up_consecutive = max(1, int(float(os.getenv("HYBRID_UP_CONSECUTIVE", "2"))))
        self.hybrid_down_hold_s = float(os.getenv("HYBRID_DOWN_HOLD_S", "180"))
        self.hybrid_degraded_retry_s = float(os.getenv("HYBRID_DEGRADED_RETRY_S", "60"))
        self.hybrid_mix_step_interval_s = float(os.getenv("HYBRID_MIX_STEP_INTERVAL_S", "20"))
        mix_weights_raw = os.getenv("HYBRID_MIX_WEIGHTS", "20,50,80,100")
        self.hybrid_mix_weights = self._parse_mix_weights(mix_weights_raw)
        self._high_load_streak: Dict[str, int] = {}
        self._low_load_since: Dict[str, float] = {}
        self._degraded_since: Dict[str, float] = {}
        self._last_mix_step_ts: Dict[str, float] = {}
        self._task: Optional[asyncio.Task] = None

    def start(self, registry) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(registry))

    @staticmethod
    def _parse_mix_weights(value: str) -> List[int]:
        weights: List[int] = []
        for part in str(value).split(","):
            token = part.strip()
            if not token:
                continue
            try:
                parsed = int(token)
            except ValueError:
                continue
            parsed = max(0, min(100, parsed))
            if parsed not in weights:
                weights.append(parsed)
        if not weights:
            return [20, 50, 80, 100]
        weights = sorted(weights)
        if weights[-1] != 100:
            weights.append(100)
        return weights

    @staticmethod
    def _normalize_routing_state(routing_state: str) -> str:
        state = (routing_state or ROUTING_FAST_ONLY).upper()
        if state not in _ROUTING_STATES:
            return ROUTING_FAST_ONLY
        return state

    @staticmethod
    def _primary_backend_for_state(routing_state: str) -> str:
        state = LoadBasedAutoscaler._normalize_routing_state(routing_state)
        if state in {ROUTING_VLLM_PRIMARY, ROUTING_MIXED}:
            return BACKEND_VLLM
        return BACKEND_LLAMA_CPP

    @staticmethod
    def _scale_backend_for_state(routing_state: str) -> str:
        state = LoadBasedAutoscaler._normalize_routing_state(routing_state)
        if state in {ROUTING_WARMING_VLLM, ROUTING_MIXED, ROUTING_VLLM_PRIMARY}:
            return BACKEND_VLLM
        return BACKEND_LLAMA_CPP

    def _resolve_min_replicas(self, config: Dict) -> int:
        backends = config.get("backends")
        if isinstance(backends, dict) and backends:
            primary = self._primary_backend_for_state(config.get("routing_state", ROUTING_FAST_ONLY))
            primary_cfg = backends.get(primary)
            if not isinstance(primary_cfg, dict):
                primary_cfg = next(
                    (cfg for cfg in backends.values() if isinstance(cfg, dict)),
                    {},
                )
            value = primary_cfg.get("min_replicas", self.min_replicas)
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return self.min_replicas
        return self.min_replicas

    @staticmethod
    def _has_backend(config: Dict[str, Any], backend_type: str) -> bool:
        backends = config.get("backends")
        return isinstance(backends, dict) and backend_type in backends

    @staticmethod
    def _inflight_sum(instances: List[Dict[str, Any]], backend_type: Optional[str] = None) -> int:
        selected = instances
        if backend_type is not None:
            selected = [i for i in instances if i.get("backend_type") == backend_type]
        return sum(int(i.get("inflight_requests", 0)) for i in selected)

    def _load_ratio(self, instances: List[Dict[str, Any]]) -> float:
        inflight = self._inflight_sum(instances)
        cap_sum, _, _ = self._capacity_stats(instances)
        if cap_sum <= 0:
            return 0.0
        return inflight / cap_sum

    def _next_mix_weight(self, current: int) -> int:
        current = max(0, min(100, int(current)))
        for weight in self.hybrid_mix_weights:
            if weight > current:
                return weight
        return 100

    async def _set_routing_state(
        self,
        registry,
        model_alias: str,
        *,
        new_state: str,
        reason: str,
        now: float,
        mix_weight: Optional[int] = None,
    ) -> None:
        normalized_state = self._normalize_routing_state(new_state)
        async with registry._lock:
            config = registry.model_configs.get(model_alias)
            if not isinstance(config, dict):
                return
            old_state = self._normalize_routing_state(config.get("routing_state", ROUTING_FAST_ONLY))
            config["routing_state"] = normalized_state
            if mix_weight is not None:
                config["mix_weight"] = max(0, min(100, int(mix_weight)))
            config["switch_reason"] = reason
            config["last_switch_ts"] = now

        if old_state != normalized_state:
            if normalized_state == ROUTING_DEGRADED_FAST:
                self._degraded_since[model_alias] = now
            else:
                self._degraded_since.pop(model_alias, None)
            _print(
                f"autoscaler: model={model_alias} routing {old_state} -> {normalized_state} "
                f"reason={reason}"
            )

    def _apply_routing_activity(self, registry, state: str, instances: List[Dict[str, Any]]) -> None:
        ready_fast = [
            i for i in instances
            if i.get("backend_type") == BACKEND_LLAMA_CPP and registry._is_instance_ready(i)
        ]
        ready_slow = [
            i for i in instances
            if i.get("backend_type") == BACKEND_VLLM and registry._is_instance_ready(i)
        ]

        for inst in instances:
            inst["active"] = False
            if registry._is_instance_ready(inst):
                inst["pending_active"] = False

        if state in {ROUTING_FAST_ONLY, ROUTING_WARMING_VLLM, ROUTING_DEGRADED_FAST}:
            selected = ready_fast or ready_slow
            for inst in selected:
                inst["active"] = True
            return

        if state == ROUTING_MIXED:
            for inst in ready_fast + ready_slow:
                inst["active"] = True
            return

        if state == ROUTING_VLLM_PRIMARY:
            selected = ready_slow or ready_fast
            for inst in selected:
                inst["active"] = True
            return

    async def _decide_routing_state(
        self,
        registry,
        model_alias: str,
        config: Dict[str, Any],
        instances: List[Dict[str, Any]],
        now: float,
    ) -> tuple[str, int]:
        state = self._normalize_routing_state(config.get("routing_state", ROUTING_FAST_ONLY))
        mix_weight = max(0, min(100, int(config.get("mix_weight", 50) or 50)))
        has_fast = self._has_backend(config, BACKEND_LLAMA_CPP)
        has_slow = self._has_backend(config, BACKEND_VLLM)

        ready_fast = [
            i for i in instances
            if i.get("backend_type") == BACKEND_LLAMA_CPP and registry._is_instance_ready(i)
        ]
        ready_slow = [
            i for i in instances
            if i.get("backend_type") == BACKEND_VLLM and registry._is_instance_ready(i)
        ]
        pending_slow = [
            i for i in instances
            if i.get("backend_type") == BACKEND_VLLM
            and (
                i.get("pending_active")
                or str(i.get("status", "")).lower() == "starting"
            )
        ]

        total_load = self._load_ratio(instances)
        total_inflight = self._inflight_sum(instances)
        high_load = (
            total_inflight >= self.hybrid_prepare_conc
            or total_load >= float(registry.scale_up_load_threshold)
        )
        if high_load:
            self._high_load_streak[model_alias] = self._high_load_streak.get(model_alias, 0) + 1
        else:
            self._high_load_streak[model_alias] = 0

        if state == ROUTING_VLLM_PRIMARY:
            if total_load < float(registry.scale_down_load_threshold) and self._inflight_sum(instances, BACKEND_VLLM) == 0:
                self._low_load_since.setdefault(model_alias, now)
            else:
                self._low_load_since.pop(model_alias, None)
        else:
            self._low_load_since.pop(model_alias, None)

        if not has_fast and has_slow and state in {
            ROUTING_FAST_ONLY,
            ROUTING_WARMING_VLLM,
            ROUTING_DEGRADED_FAST,
            ROUTING_COLD,
        }:
            await self._set_routing_state(
                registry,
                model_alias,
                new_state=ROUTING_VLLM_PRIMARY,
                reason="only_slow_backend_available",
                now=now,
                mix_weight=100,
            )
            return ROUTING_VLLM_PRIMARY, 100

        if has_fast and not has_slow and state in {
            ROUTING_WARMING_VLLM,
            ROUTING_MIXED,
            ROUTING_VLLM_PRIMARY,
        }:
            await self._set_routing_state(
                registry,
                model_alias,
                new_state=ROUTING_FAST_ONLY,
                reason="only_fast_backend_available",
                now=now,
            )
            return ROUTING_FAST_ONLY, 50

        if state == ROUTING_COLD:
            if ready_fast:
                await self._set_routing_state(
                    registry,
                    model_alias,
                    new_state=ROUTING_FAST_ONLY,
                    reason="fast_ready",
                    now=now,
                )
                return ROUTING_FAST_ONLY, 50
            if ready_slow:
                await self._set_routing_state(
                    registry,
                    model_alias,
                    new_state=ROUTING_VLLM_PRIMARY,
                    reason="slow_ready",
                    now=now,
                    mix_weight=100,
                )
                return ROUTING_VLLM_PRIMARY, 100
            return state, mix_weight

        if state in {ROUTING_FAST_ONLY, ROUTING_DEGRADED_FAST}:
            if has_slow:
                if state == ROUTING_DEGRADED_FAST:
                    degraded_since = self._degraded_since.get(model_alias, now)
                    if now - degraded_since >= self.hybrid_degraded_retry_s:
                        await self._set_routing_state(
                            registry,
                            model_alias,
                            new_state=ROUTING_WARMING_VLLM,
                            reason="degraded_retry_window",
                            now=now,
                        )
                        return ROUTING_WARMING_VLLM, mix_weight
                elif self._high_load_streak.get(model_alias, 0) >= self.hybrid_up_consecutive:
                    await self._set_routing_state(
                        registry,
                        model_alias,
                        new_state=ROUTING_WARMING_VLLM,
                        reason="high_load_prepare_slow",
                        now=now,
                    )
                    return ROUTING_WARMING_VLLM, mix_weight
            return state, mix_weight

        if state == ROUTING_WARMING_VLLM:
            if ready_slow:
                first_weight = self.hybrid_mix_weights[0]
                self._last_mix_step_ts[model_alias] = now
                await self._set_routing_state(
                    registry,
                    model_alias,
                    new_state=ROUTING_MIXED,
                    reason="slow_ready_for_mixed",
                    now=now,
                    mix_weight=first_weight,
                )
                return ROUTING_MIXED, first_weight
            if not has_slow:
                await self._set_routing_state(
                    registry,
                    model_alias,
                    new_state=ROUTING_FAST_ONLY,
                    reason="slow_backend_missing",
                    now=now,
                )
                return ROUTING_FAST_ONLY, mix_weight
            if not pending_slow and self._high_load_streak.get(model_alias, 0) == 0:
                await self._set_routing_state(
                    registry,
                    model_alias,
                    new_state=ROUTING_FAST_ONLY,
                    reason="load_recovered_before_slow_ready",
                    now=now,
                )
                return ROUTING_FAST_ONLY, mix_weight
            return state, mix_weight

        if state == ROUTING_MIXED:
            if has_slow and not ready_slow:
                self._degraded_since[model_alias] = now
                await self._set_routing_state(
                    registry,
                    model_alias,
                    new_state=ROUTING_DEGRADED_FAST,
                    reason="slow_not_ready_in_mixed",
                    now=now,
                )
                return ROUTING_DEGRADED_FAST, mix_weight

            last_step_ts = self._last_mix_step_ts.get(model_alias, 0.0)
            if now - last_step_ts >= self.hybrid_mix_step_interval_s:
                next_weight = self._next_mix_weight(mix_weight)
                self._last_mix_step_ts[model_alias] = now
                if next_weight >= 100:
                    await self._set_routing_state(
                        registry,
                        model_alias,
                        new_state=ROUTING_VLLM_PRIMARY,
                        reason="mixed_promoted_to_vllm_primary",
                        now=now,
                        mix_weight=100,
                    )
                    return ROUTING_VLLM_PRIMARY, 100
                await self._set_routing_state(
                    registry,
                    model_alias,
                    new_state=ROUTING_MIXED,
                    reason="mixed_weight_step",
                    now=now,
                    mix_weight=next_weight,
                )
                return ROUTING_MIXED, next_weight
            return state, mix_weight

        if state == ROUTING_VLLM_PRIMARY:
            if has_slow and not ready_slow:
                self._degraded_since[model_alias] = now
                await self._set_routing_state(
                    registry,
                    model_alias,
                    new_state=ROUTING_DEGRADED_FAST,
                    reason="slow_not_ready_in_vllm_primary",
                    now=now,
                )
                return ROUTING_DEGRADED_FAST, mix_weight

            low_since = self._low_load_since.get(model_alias)
            if has_fast and low_since is not None and now - low_since >= self.hybrid_down_hold_s:
                await self._set_routing_state(
                    registry,
                    model_alias,
                    new_state=ROUTING_FAST_ONLY,
                    reason="low_load_back_to_fast",
                    now=now,
                )
                return ROUTING_FAST_ONLY, 50
            return state, mix_weight

        return state, mix_weight

    def _capacity_stats(self, instances: List[Dict]) -> tuple[int, float, int]:
        caps: List[int] = []
        unknown = 0
        for inst in instances:
            cap = inst.get("capacity")
            try:
                cap = int(cap)
            except (TypeError, ValueError):
                cap = None
            if cap is None or cap <= 0:
                unknown += 1
                cap = self.instance_capacity
            caps.append(cap)
        total = sum(caps)
        avg = total / len(caps) if caps else 0.0
        return total, avg, unknown

    def _format_stats(
        self,
        instances: List[Dict],
        active: List[Dict],
        pending: List[Dict],
        min_replicas: int,
        avg_latency: float,
        sample_count: int,
        scale_up_threshold: float,
        load_active: float,
        scale_up_load_threshold: float,
        scale_down_load_threshold: float,
    ) -> str:
        status_counts = Counter(str(i.get("status", "unknown")).lower() for i in instances)
        sleep_counts = Counter(int(i.get("sleep_level_value", -1)) for i in instances)
        inflight_all = sum(int(i.get("inflight_requests", 0)) for i in instances)
        inflight_active = sum(int(i.get("inflight_requests", 0)) for i in active)
        load_avg = 0.0
        if active:
            load_avg = sum(float(i.get("load", 0.0)) for i in active) / len(active)
        cap_all_sum, cap_all_avg, cap_all_unknown = self._capacity_stats(instances)
        cap_active_sum, cap_active_avg, cap_active_unknown = self._capacity_stats(active)

        per_instance = []
        for inst in instances:
            alias = inst.get("instance_alias") or inst.get("alias") or "unknown"
            status = str(inst.get("status", "unknown")).lower()
            sleep = inst.get("sleep_level_value")
            inflight = int(inst.get("inflight_requests", 0))
            load = inst.get("load")
            try:
                load = float(load) if load is not None else 0.0
            except (TypeError, ValueError):
                load = 0.0
            cap = inst.get("capacity")
            try:
                cap = int(cap)
            except (TypeError, ValueError):
                cap = self.instance_capacity
            per_instance.append(
                f"{alias}(status={status},sleep={sleep},cap={cap},inflight={inflight},load={load:.2f},"
                f"active={int(bool(inst.get('active')))},pending={int(bool(inst.get('pending_active')))}"
                ")"
            )
        per_instance_str = "; ".join(per_instance)

        status_str = ",".join(f"{k}={v}" for k, v in sorted(status_counts.items()))
        sleep_str = ",".join(f"{k}={v}" for k, v in sorted(sleep_counts.items()))
        return (
            f"min={min_replicas} "
            f"lat={avg_latency:.3f}s samples={sample_count} th={scale_up_threshold:.3f}s "
            f"load_active={load_active:.2f} th_up={scale_up_load_threshold:.2f} "
            f"th_down={scale_down_load_threshold:.2f} "
            f"cap_active={cap_active_sum} avg={cap_active_avg:.1f} unknown={cap_active_unknown} "
            f"cap_all={cap_all_sum} avg={cap_all_avg:.1f} unknown={cap_all_unknown} "
            f"inflight_active={inflight_active} inflight_all={inflight_all} "
            f"load_avg={load_avg:.2f} status[{status_str}] sleep[{sleep_str}] "
            f"pending={len(pending)} instances[{per_instance_str}]"
        )

    async def select_instance(self, registry, model_alias: str) -> Optional[Dict]:
        """Deprecated: routing handled by ACTIVE list in Serve."""
        async with registry._lock:
            instances = list(registry.model_instances.get(model_alias, []))
        if not instances:
            return None
        active = [i for i in instances if i.get("active")]
        if not active:
            return None
        return active[0]

    async def _loop(self, registry) -> None:
        while True:
            await asyncio.sleep(self.check_interval)
            try:
                await self._refresh_metrics(registry)
                await self._scale(registry)
            except Exception as exc:
                logger.warning("Autoscaler loop error: %s", exc)

    async def _refresh_metrics(self, registry) -> None:
        async with registry._lock:
            instances_snapshot = {
                alias: list(instances)
                for alias, instances in registry.model_instances.items()
            }

        now = time.time()
        async with httpx.AsyncClient(timeout=self.health_timeout) as client:
            for instances in instances_snapshot.values():
                for instance in instances:
                    worker_url = instance.get("control_url") or instance.get("worker_url")
                    inst_alias = instance.get("instance_alias")
                    if not worker_url or not inst_alias:
                        continue
                    try:
                        resp = await client.get(f"{worker_url}/instances/{inst_alias}/status")
                        if resp.status_code != 200:
                            continue
                        status = resp.json()
                        status_value = str(status.get("status", "")).lower()
                        inflight = int(status.get("inflight_requests", 0))
                        sleep_level = int(status.get("sleep_level_value", 0))
                        e2e_avg = status.get("e2e_avg")
                        e2e_last = status.get("e2e_last")
                        request_count = int(status.get("request_count", 0))
                        capacity = status.get("capacity") or self.instance_capacity
                        capacity = max(1, int(capacity))
                        load = min(inflight / capacity, 1.0)
                        async with registry._lock:
                            prev_count = int(instance.get("request_count", 0))
                            if request_count > prev_count:
                                instance["last_request_ts"] = now
                            instance["inflight_requests"] = inflight
                            instance["sleep_level_value"] = sleep_level
                            instance["load"] = load
                            instance["capacity"] = capacity
                            instance["e2e_avg"] = e2e_avg
                            instance["e2e_last"] = e2e_last
                            instance["request_count"] = request_count
                            if "status" in status:
                                instance["status"] = status["status"]
                            if status_value in {"error", "stopped"}:
                                instance["active"] = False
                                instance["pending_active"] = False
                            ready = registry._is_instance_ready(instance)
                            if instance.get("active") and not ready:
                                instance["active"] = False
                            if instance.get("pending_active") and ready:
                                instance["active"] = True
                                instance["pending_active"] = False
                    except Exception as exc:
                        logger.debug("Failed to refresh instance %s: %s", inst_alias, exc)

    async def _scale(self, registry) -> None:
        async with registry._lock:
            model_aliases = list(registry.model_instances.keys())

        for model_alias in model_aliases:
            async with registry._lock:
                instances = list(registry.model_instances.get(model_alias, []))
                config = registry.model_configs.get(model_alias, {})

            if not instances:
                continue

            now = time.time()
            routing_state, mix_weight = await self._decide_routing_state(
                registry, model_alias, config, instances, now
            )
            config["routing_state"] = routing_state
            config["mix_weight"] = mix_weight
            self._apply_routing_activity(registry, routing_state, instances)

            active = [i for i in instances if i.get("active")]
            pending = [
                i for i in instances
                if (i.get("pending_active") or str(i.get("status", "")).lower() == "starting")
                and str(i.get("status", "")).lower() not in {"error", "stopped"}
            ]
            min_replicas = self._resolve_min_replicas(config)
            target_backend = self._scale_backend_for_state(routing_state)
            if not self._has_backend(config, target_backend):
                target_backend = (
                    BACKEND_LLAMA_CPP
                    if self._has_backend(config, BACKEND_LLAMA_CPP)
                    else BACKEND_VLLM
                )
            if len(active) < min_replicas:
                detail = self._format_stats(
                    instances,
                    active,
                    pending,
                    min_replicas,
                    0.0,
                    0,
                    0.0,
                    0.0,
                    float(registry.scale_up_load_threshold),
                    float(registry.scale_down_load_threshold),
                )
                if pending:
                    _print(
                        f"autoscaler: model={model_alias} active={len(active)} "
                        f"pending={len(pending)} min_replicas={min_replicas} {detail} -> wait pending"
                    )
                    continue
                _print(
                    f"autoscaler: model={model_alias} active={len(active)} "
                    f"pending={len(pending)} min_replicas={min_replicas} {detail} -> scale up"
                )
                await self._scale_up(
                    registry,
                    model_alias,
                    instances,
                    config,
                    now,
                    target_backend=target_backend,
                    routing_state=routing_state,
                )
                continue

            scale_up_threshold = self._scale_up_thresholds.get(
                model_alias, self.scale_up_latency_threshold
            )
            avg_latency, sample_count = self._avg_latency(active, now)
            inflight_active = sum(int(i.get("inflight_requests", 0)) for i in active)
            cap_active_sum, _, _ = self._capacity_stats(active)
            load_active = (inflight_active / cap_active_sum) if cap_active_sum > 0 else 0.0
            scale_up_load_threshold = float(registry.scale_up_load_threshold)
            scale_down_load_threshold = float(registry.scale_down_load_threshold)
            detail = self._format_stats(
                instances,
                active,
                pending,
                min_replicas,
                avg_latency,
                sample_count,
                scale_up_threshold,
                load_active,
                scale_up_load_threshold,
                scale_down_load_threshold,
            )

            if not active:
                if pending:
                    _print(
                        f"autoscaler: model={model_alias} active=0 "
                        f"pending={len(pending)} avg_latency={avg_latency:.3f}s "
                        f"samples={sample_count} {detail} -> wait pending"
                    )
                    continue
                _print(
                    f"autoscaler: model={model_alias} active=0 pending=0 "
                    f"avg_latency={avg_latency:.3f}s samples={sample_count} {detail} -> scale up"
                )
                await self._scale_up(
                    registry,
                    model_alias,
                    instances,
                    config,
                    now,
                    target_backend=target_backend,
                    routing_state=routing_state,
                )
                continue

            if load_active > scale_up_load_threshold:
                if pending:
                    _print(
                        f"autoscaler: model={model_alias} active={len(active)} "
                        f"pending={len(pending)} avg_latency={avg_latency:.3f}s "
                        f"samples={sample_count} {detail} "
                        "-> skip scale up (pending)"
                    )
                    continue
                _print(
                    f"autoscaler: model={model_alias} active={len(active)} "
                    f"pending={len(pending)} avg_latency={avg_latency:.3f}s "
                    f"samples={sample_count} {detail} "
                    f"load_active>{scale_up_load_threshold:.2f} -> scale up"
                )
                await self._scale_up(
                    registry,
                    model_alias,
                    instances,
                    config,
                    now,
                    target_backend=target_backend,
                    routing_state=routing_state,
                )
            elif load_active < scale_down_load_threshold:
                _print(
                    f"autoscaler: model={model_alias} active={len(active)} "
                    f"pending={len(pending)} avg_latency={avg_latency:.3f}s "
                    f"samples={sample_count} {detail} "
                    f"load_active<{scale_down_load_threshold:.2f} -> scale down"
                )
                await self._scale_down(
                    registry,
                    model_alias,
                    instances,
                    config,
                    now,
                    min_replicas,
                    routing_state=routing_state,
                )

    def _avg_latency(self, active: List[Dict], now: float) -> tuple[float, int]:
        latencies = []
        for inst in active:
            last_ts = inst.get("last_request_ts")
            if last_ts is None or now - float(last_ts) > self.latency_sample_window_s:
                continue
            value = inst.get("e2e_avg")
            if value is None:
                value = inst.get("e2e_last")
            if value is None:
                continue
            try:
                latencies.append(float(value))
            except (TypeError, ValueError):
                continue
        if not latencies:
            return 0.0, 0
        return sum(latencies) / len(latencies), len(latencies)

    async def _ensure_latency_baseline(self, registry, model_alias: str, active: List[Dict]) -> None:
        if model_alias in self._scale_up_thresholds or model_alias in self._baseline_pending:
            return
        if not active:
            return
        self._baseline_pending.add(model_alias)
        try:
            candidate = active[0]
            latency = await self._measure_baseline_latency(registry, candidate, model_alias)
            if latency is None:
                _print(f"autoscaler: model={model_alias} baseline measure failed")
                return
            threshold = latency * self.baseline_latency_multiplier
            self._scale_up_thresholds[model_alias] = threshold
            _print(
                f"autoscaler: model={model_alias} baseline={latency:.3f}s "
                f"-> scale_up_threshold={threshold:.3f}s"
            )
        finally:
            self._baseline_pending.discard(model_alias)

    async def _measure_baseline_latency(
        self,
        registry,
        instance: Dict,
        model_alias: str,
    ) -> Optional[float]:
        worker_url = instance.get("worker_url") or instance.get("control_url")
        inst_alias = instance.get("instance_alias")
        if not worker_url or not inst_alias:
            return None
        payload = {
            "model": model_alias,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": self.baseline_max_tokens,
            "stream": False,
        }
        try:
            start = time.perf_counter()
            async with httpx.AsyncClient(timeout=self.baseline_timeout_s) as client:
                resp = await client.post(
                    f"{worker_url}/proxy/{inst_alias}/v1/chat/completions",
                    json=payload,
                )
            if resp.status_code >= 400:
                return None
            return time.perf_counter() - start
        except Exception:
            return None

    async def _scale_up(
        self,
        registry,
        model_alias: str,
        instances: List[Dict],
        config: Dict,
        now: float,
        *,
        target_backend: str,
        routing_state: str,
    ) -> None:
        last = self._last_scale_up.get(model_alias, 0.0)
        if now - last < self.scale_up_cooldown_s:
            _print(
                f"autoscaler: model={model_alias} scale-up cooldown "
                f"({self.scale_up_cooldown_s - (now - last):.1f}s left)"
            )
            return

        sleeping = [
            i for i in instances
            if i.get("backend_type") == target_backend
            and registry._sleep_level_value(i) > 0
            and not i.get("pending_active")
        ]
        if sleeping:
            candidate = min(sleeping, key=lambda i: registry._sleep_level_value(i))
            await registry._wake_instance(candidate)
            candidate["pending_active"] = True
            self._last_scale_up[model_alias] = now
            _print(
                f"autoscaler: model={model_alias} wake instance={candidate.get('instance_alias')} "
                f"level={registry._sleep_level_value(candidate)} state={routing_state}"
            )
            return

        backend_cfg = (config.get("backends") or {}).get(target_backend) or {}
        backend_max_replicas = backend_cfg.get("max_replicas", self.max_replicas)
        try:
            backend_max_replicas = max(0, int(backend_max_replicas))
        except (TypeError, ValueError):
            backend_max_replicas = self.max_replicas
        effective_instances = [
            i for i in instances
            if i.get("backend_type") == target_backend
            and str(i.get("status", "")).lower() not in {"error", "stopped"}
        ]
        if len(effective_instances) >= backend_max_replicas:
            _print(
                f"autoscaler: model={model_alias} backend={target_backend} "
                f"reached max_replicas={backend_max_replicas}"
            )
            return

        await registry._start_new_instance(
            model_alias,
            config,
            backend_type=target_backend,
        )
        self._last_scale_up[model_alias] = now
        _print(
            f"autoscaler: model={model_alias} start new instance backend={target_backend} "
            f"state={routing_state}"
        )

    async def _scale_down(
        self,
        registry,
        model_alias: str,
        instances: List[Dict],
        config: Dict,
        now: float,
        min_replicas: int,
        *,
        routing_state: str,
    ) -> None:
        last = self._last_scale_down.get(model_alias, 0.0)
        if now - last < self.scale_down_cooldown_s:
            _print(
                f"autoscaler: model={model_alias} scale-down cooldown "
                f"({self.scale_down_cooldown_s - (now - last):.1f}s left)"
            )
            return

        active = [i for i in instances if i.get("active")]
        if len(active) <= min_replicas:
            _print(f"autoscaler: model={model_alias} at min_replicas={min_replicas}")
            return

        idle = [i for i in active if int(i.get("inflight_requests", 0)) == 0]
        if not idle:
            _print(f"autoscaler: model={model_alias} no idle active instances")
            return

        state = self._normalize_routing_state(routing_state)
        if state in {ROUTING_VLLM_PRIMARY, ROUTING_MIXED}:
            preferred_backend_order = [BACKEND_LLAMA_CPP, BACKEND_VLLM]
        else:
            preferred_backend_order = [BACKEND_VLLM, BACKEND_LLAMA_CPP]

        candidate = None
        for backend_type in preferred_backend_order:
            pool = [i for i in idle if i.get("backend_type") == backend_type]
            if pool:
                candidate = sorted(pool, key=lambda i: i.get("created_at", 0))[-1]
                break
        if candidate is None:
            candidate = sorted(idle, key=lambda i: i.get("created_at", 0))[-1]
        candidate["active"] = False
        candidate["pending_active"] = False
        self._last_scale_down[model_alias] = now
        _print(
            f"autoscaler: model={model_alias} remove active instance={candidate.get('instance_alias')} "
            f"backend={candidate.get('backend_type')} state={state}"
        )
