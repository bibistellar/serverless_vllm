"""Latency-based autoscaler for vLLM instances."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


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
        self._task: Optional[asyncio.Task] = None

    def start(self, registry) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(registry))

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

            active = [i for i in instances if i.get("active")]
            ready_not_active = [i for i in instances if (not i.get("active")) and registry._is_instance_ready(i)]
            pending = [
                i for i in instances
                if i.get("pending_active") or str(i.get("status", "")).lower() == "starting"
            ]
            if len(active) < self.min_replicas and ready_not_active:
                for inst in ready_not_active[: self.min_replicas - len(active)]:
                    inst["active"] = True
            active = [i for i in instances if i.get("active")]

            now = time.time()
            await self._ensure_latency_baseline(registry, model_alias, active)
            scale_up_threshold = self._scale_up_thresholds.get(
                model_alias, self.scale_up_latency_threshold
            )
            avg_latency, sample_count = self._avg_latency(active, now)

            if not active:
                if pending:
                    _print(
                        f"autoscaler: model={model_alias} active=0 "
                        f"pending={len(pending)} avg_latency={avg_latency:.3f}s "
                        f"samples={sample_count} -> wait pending"
                    )
                    continue
                _print(
                    f"autoscaler: model={model_alias} active=0 pending=0 "
                    f"avg_latency={avg_latency:.3f}s samples={sample_count} -> scale up"
                )
                await self._scale_up(registry, model_alias, instances, config, now)
                continue

            if avg_latency > scale_up_threshold:
                if sample_count == 0:
                    _print(
                        f"autoscaler: model={model_alias} active={len(active)} "
                        f"pending={len(pending)} avg_latency={avg_latency:.3f}s "
                        f"samples=0 -> skip scale up (no recent samples)"
                    )
                    continue
                if pending:
                    _print(
                        f"autoscaler: model={model_alias} active={len(active)} "
                        f"pending={len(pending)} avg_latency={avg_latency:.3f}s "
                        f"samples={sample_count} "
                        "-> skip scale up (pending)"
                    )
                    continue
                _print(
                    f"autoscaler: model={model_alias} active={len(active)} "
                    f"pending={len(pending)} avg_latency={avg_latency:.3f}s "
                    f"samples={sample_count} "
                    f"> {scale_up_threshold:.3f}s -> scale up"
                )
                await self._scale_up(registry, model_alias, instances, config, now)
            elif avg_latency < self.scale_down_latency_threshold:
                _print(
                    f"autoscaler: model={model_alias} active={len(active)} "
                    f"pending={len(pending)} avg_latency={avg_latency:.3f}s "
                    f"samples={sample_count} "
                    f"< {self.scale_down_latency_threshold:.3f}s -> scale down"
                )
                await self._scale_down(registry, model_alias, instances, config, now)

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
            if registry._sleep_level_value(i) > 0 and not i.get("pending_active")
        ]
        if sleeping:
            candidate = min(sleeping, key=lambda i: registry._sleep_level_value(i))
            await registry._wake_instance(candidate)
            candidate["pending_active"] = True
            self._last_scale_up[model_alias] = now
            _print(
                f"autoscaler: model={model_alias} wake instance={candidate.get('instance_alias')} "
                f"level={registry._sleep_level_value(candidate)}"
            )
            return

        if len(instances) >= self.max_replicas:
            _print(f"autoscaler: model={model_alias} reached max_replicas={self.max_replicas}")
            return

        await registry._start_new_instance(model_alias, config)
        self._last_scale_up[model_alias] = now
        _print(f"autoscaler: model={model_alias} start new instance")

    async def _scale_down(
        self,
        registry,
        model_alias: str,
        instances: List[Dict],
        config: Dict,
        now: float,
    ) -> None:
        last = self._last_scale_down.get(model_alias, 0.0)
        if now - last < self.scale_down_cooldown_s:
            _print(
                f"autoscaler: model={model_alias} scale-down cooldown "
                f"({self.scale_down_cooldown_s - (now - last):.1f}s left)"
            )
            return

        active = [i for i in instances if i.get("active")]
        if len(active) <= self.min_replicas:
            _print(f"autoscaler: model={model_alias} at min_replicas={self.min_replicas}")
            return

        idle = [i for i in active if int(i.get("inflight_requests", 0)) == 0]
        if not idle:
            _print(f"autoscaler: model={model_alias} no idle active instances")
            return

        candidate = sorted(idle, key=lambda i: i.get("created_at", 0))[-1]
        candidate["active"] = False
        candidate["pending_active"] = False
        self._last_scale_down[model_alias] = now
        _print(
            f"autoscaler: model={model_alias} remove active instance={candidate.get('instance_alias')}"
        )
