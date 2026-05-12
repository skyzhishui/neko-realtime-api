# neko-realtime-api

基于 WebSocket 的实时语音对话服务，兼容 OpenAI Realtime API 协议风格。支持语音输入 → LLM 推理 → TTS 语音合成的全双工实时流式交互，内置 VAD 语音活动检测和智能打断。

## 工作模式

| 模式 | 名称 | 流程 | 说明 |
|------|------|------|------|
| A 模式 | `asr_llm` | 音频 → ASR 转文字 → LLM → TTS | 兼容性更好 |
| B 模式 | `omni_audio` | 音频 → Omni 多模态模型 → TTS | 延迟更低，需部署 Omni 模型 |

服务启动后默认使用 `config.yaml` 中 `default_mode` 指定的模式。B 模式连续出错超过阈值时会自动降级到 A 模式，降级持续时间后自动恢复。

## 快速部署

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 主要依赖：FastAPI、uvicorn、aiohttp、PyTorch、Silero VAD、ONNX Runtime、Transformers

### 2. 准备模型

VAD 和 Smart Turn 模型默认从 ModelScope 自动下载到 `./models/` 目录，也可手动指定路径：

```
./models/silero-vad/       # Silero VAD 模型
./models/smart-turn-v3/    # Smart Turn v3.2 模型
./models/SenseVoiceSmall/  # 本地 ASR 模型（local_asr=true 时需要）
```

### 3. 配置

```bash
cp config.yaml.example config.yaml
```

编辑 `config.yaml`，重点修改以下配置：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `realtime_server.host` | 监听地址 | `0.0.0.0` |
| `realtime_server.port` | 监听端口 | `8765` |
| `realtime_server.default_mode` | 默认工作模式 | `asr_llm` |
| `realtime_server.auth_enabled` | 是否启用认证 | `false` |
| `services.omni.base_url` | Omni 模型 API 地址 | `http://localhost:8000` |
| `services.asr.local_asr` | 是否使用本地 ASR | `true` |
| `services.tts.base_url` | TTS HTTP API 地址 | `http://localhost:8091` |
| `services.tts.mode` | TTS 调用模式 (`http`/`ws`) | `http` |
| `services.gsv_tts.enabled` | 是否启用 GSV-TTS-Lite 语音合成 | `false` |

> 详细配置说明见 `config.yaml.example`，每项均有中文注释。

### 4. 启动服务

```bash
python main.py
```

启动时会自动预加载 VAD、Smart Turn 和本地 ASR 模型（如果启用），首次启动模型下载可能需要几分钟。

## API 使用

### HTTP 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 服务信息 |
| `/health` | GET | 健康检查 |
| `/v1/realtime` | WebSocket | 实时语音对话 |

### WebSocket 连接

```
ws://<host>:<port>/v1/realtime?model=local-qwen-omni
```

启用认证时需在连接头中携带：
```
Authorization: Bearer <auth_token>
```

### 客户端事件（Client → Server）

| 事件类型 | 说明 |
|----------|------|
| `session.update` | 更新会话配置（voice、temperature、turn_detection 等） |
| `input_audio_buffer.append` | 发送音频数据（base64 编码的 PCM16） |
| `input_audio_buffer.clear` | 清空音频缓冲区 |
| `input_image_buffer.append` | 发送图片数据（多模态输入） |
| `conversation.item.create` | 添加对话历史 |
| `response.create` | 主动触发响应 |
| `response.cancel` | 取消当前响应 |

### 服务端事件（Server → Client）

| 事件类型 | 说明 |
|----------|------|
| `input_audio_buffer.speech_started` | 检测到用户开始说话 |
| `input_audio_buffer.speech_stopped` | 检测到用户说完 |
| `conversation.item.input_audio_transcription.completed` | ASR 转写结果 |
| `response.created` | 响应开始 |
| `response.audio_transcript.delta` | LLM 文本增量 |
| `response.audio.delta` | TTS 音频增量（base64 编码 PCM16） |
| `response.audio_transcript.done` | 文本输出完成 |
| `response.done` | 响应结束 |
| `error` | 错误信息 |

### 音频格式

- **输入**：16kHz 单声道 PCM16（base64 编码）
- **输出**：24kHz 单声道 PCM16（base64 编码）

### 示例：发送音频

```python
import asyncio
import base64
import json
import websockets

async def main():
    async with websockets.connect("ws://localhost:8765/v1/realtime") as ws:
        # 读取 PCM 文件并发送
        with open("audio.pcm", "rb") as f:
            pcm_data = f.read()
        
        audio_b64 = base64.b64encode(pcm_data).decode()
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": audio_b64
        }))
        
        # 接收响应
        while True:
            msg = await ws.recv()
            event = json.loads(msg)
            print(event["type"])
            if event["type"] == "response.done":
                break

asyncio.run(main())
```

## 外部服务依赖

本服务需要以下后端服务配合运行：

| 服务 | 用途 | 默认地址 |
|------|------|----------|
| Omni 多模态模型 | LLM 推理（A/B 模式通用） | `http://localhost:8000` |
| TTS 服务 | 语音合成（Qwen3-TTS） | `http://localhost:8091` |
| GSV-TTS-Lite | 语音克隆 TTS（可选） | `http://localhost:8001` |
| 远程 ASR | 语音识别（local_asr=false 时） | 可配置 |

Omni API 兼容 OpenAI Chat Completions 接口格式，TTS API 兼容 OpenAI Speech 接口格式

## License

MIT License
