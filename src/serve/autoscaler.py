"""Load-based autoscaler and routing policy for vLLM instances."""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class LoadBasedAutoscaler:
    """Auto-scale and routing policy based on instance load.

    Exposes a single selection interface for routing. Runs a background
    loop to refresh load metrics and scale instances up/down.
    """

    def __init__(
        self,
        load_high: float = 0.7,
        load_low: float = 0.3,
        instance_capacity: int = 4,
        min_replicas: int = 1,
        max_replicas: int = 8,
        check_interval: float = 10.0,
        health_timeout: float = 3.0,
    ) -> None:
        self.load_high = load_high
        self.load_low = load_low
        self.instance_capacity = max(1, instance_capacity)
        self.min_replicas = max(1, min_replicas)
        self.max_replicas = max(self.min_replicas, max_replicas)
        self.check_interval = check_interval
        self.health_timeout = health_timeout
        self._task: Optional[asyncio.Task] = None

    def start(self, registry) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(registry))

    async def select_instance(self, registry, model_alias: str) -> Optional[Dict]:
        """Pick the lowest-load active instance; wake one if needed."""
        async with registry._lock:
            instances = list(registry.model_instances.get(model_alias, []))

        if not instances:
            return None

        active = [i for i in instances if i.get("sleep_level_value", 0) == 0]
        if active:
            return min(active, key=lambda i: i.get("load", 0.0))

        # No active instance, pick the lightest sleep level and wake it
        candidate = min(instances, key=lambda i: i.get("sleep_level_value", 3))
        await registry._wake_instance(candidate)
        return candidate

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
                        capacity = status.get("capacity") or self.instance_capacity
                        capacity = max(1, int(capacity))
                        load = min(inflight / capacity, 1.0)
                        async with registry._lock:
                            instance["inflight_requests"] = inflight
                            instance["sleep_level_value"] = sleep_level
                            instance["load"] = load
                            instance["capacity"] = capacity
                            if "status" in status:
                                instance["status"] = status["status"]
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

            active = [i for i in instances if i.get("sleep_level_value", 0) == 0]
            avg_load = 0.0
            if active:
                avg_load = sum(i.get("load", 0.0) for i in active) / len(active)

            if avg_load > self.load_high:
                await self._scale_up(registry, model_alias, instances, config)
            elif avg_load < self.load_low:
                await self._scale_down(registry, model_alias, instances, config)

    async def _scale_up(self, registry, model_alias: str, instances: List[Dict], config: Dict) -> None:
        # Prefer waking sleeping instances
        sleeping = [i for i in instances if i.get("sleep_level_value", 0) > 0]
        if sleeping:
            candidate = min(sleeping, key=lambda i: i.get("sleep_level_value", 3))
            await registry._wake_instance(candidate)
            return

        if len(instances) >= self.max_replicas:
            return

        await registry._start_new_instance(model_alias, config)

    async def _scale_down(self, registry, model_alias: str, instances: List[Dict], config: Dict) -> None:
        active = [i for i in instances if i.get("sleep_level_value", 0) == 0]
        if len(active) > self.min_replicas:
            candidate = min(active, key=lambda i: i.get("load", 0.0))
            await registry._sleep_instance(candidate, level=1)
            return

        sleep1 = [i for i in instances if i.get("sleep_level_value", 0) == 1]
        if sleep1:
            await registry._sleep_instance(sleep1[0], level=2)
            return

        sleep2 = [i for i in instances if i.get("sleep_level_value", 0) == 2]
        if sleep2:
            await registry._sleep_instance(sleep2[0], level=3)
