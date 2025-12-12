1. 背景与目标

本模块是 VAD 项目中的专用 LLM 解释服务，目标：

在多台服务器组成的 Ray 集群上，预先创建多组 vLLM engine，全部进入 sleep mode level=2，降低显存占用；

对外通过一个统一的 HTTP 接口 POST /v1/chat/completions 提供 LLM 推理能力；

后端通过 Pool Manager 统一管理所有 engine 实例，选择一个可用 engine 执行本次请求；

所有需要初始化的模型由一个统一配置文件定义。

不考虑对外多租户场景，不实现复杂观测与日志，仅实现最基本的功能链路。

2. 总体架构
2.1 组件划分

Ray 集群层

多台服务器，组成一个 Ray 集群（1 个 head + N 个 worker）。

所有 Ray actor（Pool Manager、Engine Worker）运行其上。

路由层（Router, Ray Serve 部署）

使用 Ray Serve 实现 HTTP 服务。

对外暴露单一接口 POST /v1/chat/completions。

接收请求后，调用 Pool Manager 选取一个 engine，转发请求，返回结果。

后端层（Pool Manager + Engine Worker）

Pool Manager（Ray actor，单实例）

启动时读取统一配置文件，得到需要初始化的模型列表及参数。

获取 Ray 集群的节点信息，在每台服务器上为每个模型创建若干 Engine Worker。

维护 engine 列表（模型 → 多个 engine）。

对外暴露接口：get_engine_for_request(model_name)，返回一个可用 Engine Worker 句柄。

Engine Worker（Ray actor，多实例）

封装一个 vLLM LLM 对象。

初始化完成后立即进入 sleep mode level=2。

对外暴露接口：generate(request)，执行一次推理并返回结果。

2.2 调用链路（简化描述）

客户端（VAD 系统）调用 POST /v1/chat/completions。

Router 解析请求，确定 model 与对话消息。

Router 调用 PoolManager.get_engine_for_request(model)，获取一个 Engine Worker。

Router 调用该 Engine Worker 的 generate(request)。

Engine Worker 唤醒 vLLM（如当前处于 sleep 状态），执行推理，返回结果。

Router 将结果包装为 v1/chat/completions 格式，返回给客户端。

3. 对外 API 设计
3.1 Endpoint

Method: POST

Path: /v1/chat/completions

Content-Type: application/json

3.2 请求体（参考 OpenAI Chat Completion 简化版）
{
  "model": "qwen3-vl-2b-vad",
  "messages": [
    { "role": "system", "content": "你是一个监控视频异常分析专家。" },
    { "role": "user", "content": "请解释当前视频中的异常行为。" }
  ],
  "max_tokens": 128,
  "temperature": 0.2
}


说明：

model：字符串，必须与配置文件中的某个模型名一致，如 "qwen3-vl-2b-vad"。

messages：chat 对话的消息列表，至少包含一条 user 消息。

当前实现可以先简单处理成一个 prompt 串联（由 Router 负责拼接）。

max_tokens、temperature：可选；若缺省则使用模型配置中的默认值。

3.3 响应体（简化版）
{
  "id": "chatcmpl-xxx",
  "model": "qwen3-vl-2b-vad",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "监控画面中出现多人在收银台附近争执并有推搡动作，属于暴力冲突异常。"
      },
      "finish_reason": "stop"
    }
  ]
}

4. 组件详细设计
4.1 路由层（VadLLMRouter）
4.1.1 职责

接收 HTTP 请求 /v1/chat/completions。

校验参数、拼接 prompt。

调用 PoolManager.get_engine_for_request(model)。

调用 Engine Worker 的 generate 方法。

将结果转换为符合 Chat Completion 格式的 JSON 返回。

4.1.2 伪代码示意
@serve.deployment(route_prefix="/v1")
class VadLLMRouter:
    def __init__(self, pool_handle):
        self.pool = pool_handle  # Ray actor: PoolManager

    async def chat_completions(self, request: Request):
        body = await request.json()
        model = body["model"]
        messages = body["messages"]

        # 1. 将 messages 拼成一个 prompt（简单策略）
        prompt = self._build_prompt(messages)
        max_tokens = body.get("max_tokens")
        temperature = body.get("temperature")

        # 2. 请求 PoolManager 选择一个 engine
        engine_ref = await self.pool.get_engine_for_request.remote(model)

        # 3. 构造内部请求
        internal_req = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # 4. 调用 engine.generate
        output = await engine_ref.generate.remote(internal_req)
        text = output["text"]

        # 5. 组装 Chat Completion 响应
        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text
                    },
                    "finish_reason": "stop"
                }
            ]
        }
        return JSONResponse(resp)

    def _build_prompt(self, messages):
        # 简单拼接，可以后续换成更复杂的模板
        parts = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            parts.append(f"{role.upper()}: {content}")
        return "\n".join(parts)

    # Ray Serve 入口：映射到 /v1/chat/completions 路由
    async def __call__(self, request: Request):
        # 简易路由匹配
        path = request.url.path
        if path.endswith("/chat/completions"):
            return await self.chat_completions(request)
        else:
            return JSONResponse({"error": "not found"}, status_code=404)

4.2 Pool Manager
4.2.1 职责

启动时读取配置文件，知道有哪些模型、每模型在每个节点需要多少个 engine。

获取 Ray 节点列表，在每个节点上创建对应数量的 Engine Worker。

为每个模型维护一个 engine 列表。

对外提供 get_engine_for_request(model_name)，返回一个可用的 Engine Worker actor handle。

4.2.2 简化选择策略

目前先实现最简单的策略：

在 model_name 对应的 engine 列表中 随机选择一个。

不考虑负载，仅保证所有 engine 都可用。

后续可在 Engine Worker 中增加状态上报，在 Pool Manager 中实现按负载或按节点选择。

4.2.3 伪代码示意
@ray.remote
class PoolManager:
    def __init__(self, config_path: str):
        self.engines_by_model = {}  # model_name -> List[ActorHandle]
        self._load_config(config_path)
        self._init_engines_on_all_nodes()

    def _load_config(self, path):
        # 读取 YAML，得到 self.model_configs: dict[model_name] -> ModelConfig
        ...

    def _init_engines_on_all_nodes(self):
        nodes = ray.nodes()
        for node in nodes:
            if not node["Alive"]:
                continue
            node_id = node["NodeID"]
            for model_name, cfg in self.model_configs.items():
                for _ in range(cfg.num_engines_per_node):
                    worker = LLMEngineWorker.options(
                        scheduling_strategy=NodeAffinitySchedulingStrategy(
                            node_id=node_id,
                            soft=True,
                        )
                    ).remote(model_name, cfg)
                    self.engines_by_model.setdefault(model_name, []).append(worker)

    def get_engine_for_request(self, model_name: str):
        engines = self.engines_by_model.get(model_name, [])
        if not engines:
            raise RuntimeError(f"No engines for model {model_name}")
        # 简单随机挑一个
        return random.choice(engines)

4.3 Engine Worker（封装 vLLM）
4.3.1 职责

初始化 vLLM LLM 实例。

初始化完成后立即设置 sleep mode level=2。

接收单请求，必要时唤醒 engine，执行生成，返回文本。

4.3.2 伪代码示意
@ray.remote(num_gpus=0.2)  # 具体资源按实际调整
class LLMEngineWorker:
    def __init__(self, model_name: str, cfg: ModelConfig):
        self.model_name = model_name
        self.cfg = cfg
        self.llm = LLM(
            model=cfg.repo,
            max_model_len=cfg.max_model_len,
            tensor_parallel_size=cfg.tensor_parallel_size,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            enable_sleep_mode=True,
            enforce_eager=True,
        )
        self.sleep_level = 2
        self._set_sleep_level(2)

    def _set_sleep_level(self, level: int):
        # 封装真实的 vLLM sleep mode 调用
        # self.llm.llm_engine.set_sleep_mode(level)
        self.sleep_level = level

    def _wake_up_if_needed(self):
        if self.sleep_level > 0:
            # self.llm.llm_engine.wake_up()
            self.sleep_level = 0

    def generate(self, req: dict):
        """
        req: {
          "prompt": str,
          "max_tokens": Optional[int],
          "temperature": Optional[float]
        }
        """
        prompt = req["prompt"]
        max_tokens = req.get("max_tokens") or self.cfg.default_max_tokens
        temperature = req.get("temperature") or self.cfg.default_temperature

        self._wake_up_if_needed()

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )
        outputs = self.llm.generate(
            prompts=[prompt],
            sampling_params=sampling_params,
        )
        text = outputs[0].outputs[0].text

        return {
            "text": text
        }

5. 当前版本总结

当前版本的 serverless LLM pool 具备的能力：

多机 Ray 集群，统一通过 Pool Manager 创建并管理 vLLM engine。

每台机器为每个模型启动相同数量的 engine，初始全部进入 sleep mode level=2，显存驻留较低。

对外暴露一个简单、统一的 HTTP 接口 POST /v1/chat/completions，与 OpenAI 风格兼容。

Router 的处理流程：
HTTP 请求 → Router 解析 → PoolManager.get_engine_for_request(model) → engine.generate() → 返回结果