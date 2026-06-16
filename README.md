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

> 主要依赖：FastAPI、uvicorn、aiohttp、sherpa-onnx、Silero VAD (ONNX)、ONNX Runtime、Transformers

### 2. 准备模型

VAD、Smart Turn 和 ASR 模型默认从 ModelScope 自动下载到 `./models/` 目录，也可手动指定路径：

```
./models/silero-vad/                        # Silero VAD 模型 (ONNX)
./models/smart-turn-v3/                     # Smart Turn v3.2 模型 (ONNX)
./models/sherpa-onnx-sense-voice-small/     # 本地 ASR 模型 (sherpa-onnx, local_asr=true 时需要)
```

ASR 模型目录需包含以下文件（首次启动时自动从 ModelScope 下载 `xiaowangge/sherpa-onnx-sense-voice-small`）：

```
model_q8.onnx   # Q8 量化模型（优先使用，228MB，推理更快）
model.onnx      # FP32 模型（备选，894MB）
tokens.txt      # 词表文件
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
| `services.omni.base_url` | Omni 模型 API 地址 | `http://localhost:8000/v1` |
| `services.omni.api_key` | Omni API Key（可选，null 则不发送） | `null` |
| `services.asr.local_asr` | 是否使用本地 ASR | `true` |
| `services.asr.asr_model_path` | 本地 ASR 模型路径 | `./models/sherpa-onnx-sense-voice-small` |
| `services.tts.base_url` | TTS API 地址（需含 /v1 前缀） | `http://localhost:8091/v1` |
| `services.tts.model` | TTS 后端 (`Qwen3-TTS`/`voxcpm2`/`gsv-tts-lite`) | `Qwen3-TTS` |
| `services.tts.mode` | TTS 调用模式 (`http`/`ws`，Qwen3-TTS 和 VoxCPM2 均支持） | `http` |
| `services.tts.api_key` | TTS API Key（可选，null 则不发送，HTTP/WS 共用） | `null` |
| `services.tts.voice` | 默认音色（Qwen3-TTS 预设音色 / VoxCPM2 填 default） | `vivian` |
| `services.tts.ref_audio` | 声音克隆参考音频路径或 URL（可选） | `null` |
| `services.tts.ref_text` | 声音克隆参考音频转录文本（可选） | `null` |

> 详细配置说明见 `config.yaml.example`，每项均有中文注释。

### 4. 启动服务

```bash
python main.py
```

启动时会自动预加载 VAD、Smart Turn 和本地 ASR 模型（如果启用），首次启动模型下载可能需要几分钟。

## ASR 引擎

本地 ASR 使用 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 的 SenseVoice-Small 模型进行语音识别，支持 ONNX 推理，无需 PyTorch 依赖。

| 特性 | 说明 |
|------|------|
| 推理框架 | sherpa-onnx (ONNX Runtime) |
| 模型 | SenseVoice-Small (Q8 量化) |
| ModelScope 源 | `xiaowangge/sherpa-onnx-sense-voice-small` |
| 输入格式 | 16kHz 单声道 PCM16 |
| 语言支持 | 中文 / 英文 / 日文 / 韩文 / 自动检测 |
| 逆文本正则化 | 启用 (ITN)，数字/日期等转为自然语言 |
| 典型 RTF | 0.006~0.018 (CPU, Q8) |


## VAD 引擎

VAD 使用 Silero VAD 模型的 ONNX 推理后端，无需 PyTorch 依赖。Smart Turn v3.2 话轮检测同样使用 ONNX 推理。

| 组件 | 推理后端 | 模型 |
|------|----------|------|
| Silero VAD | ONNX Runtime (CPU) | `silero_vad.onnx` |
| Smart Turn v3.2 | ONNX Runtime (CPU) | `smart-turn-v3.2-gpu.onnx` |

## API 使用

### HTTP 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 服务信息 |
| `/health` | GET | 健康检查 |
| `/v1/audio/transcriptions` | POST | OpenAI 兼容语音转写（multipart 上传） |
| `/v1/realtime` | WebSocket | 实时语音对话 |

### 语音转写端点（POST /v1/audio/transcriptions）

OpenAI Whisper 协议兼容的独立 ASR 端点，与 WebSocket 端点共享同一份 ASR 引擎（`local_asr=true` 时复用启动预加载的本地模型，否则转发到 `services.asr.base_url`）。

**请求**：

```bash
curl -X POST http://localhost:8765/v1/audio/transcriptions \
  -H "Authorization: Bearer <auth_token>" \
  -F "file=@audio.wav" \
  -F "model=SenseVoiceSmall" \
  -F "language=zh"
```

| Form 字段 | 说明 | 默认值 |
|-----------|------|--------|
| `file`    | 音频文件，支持 WAV/FLAC/OGG/MP3 等 soundfile 可解码格式，或原始 int16 PCM | 必填 |
| `model`   | 兼容字段，忽略（实际使用本地或远程配置） | `SenseVoiceSmall` |
| `language` | 语言代码 `zh` / `en` / `ja` / `ko` / `auto` | `auto` |

**响应**：

```json
{"text": "识别得到的文本"}
```

**认证**：与 WebSocket 端点共用 `security.auth_token`，启用认证时缺失或错误的 Bearer Token 返回 401。

**ASR 引擎选择**：
- `services.asr.local_asr = true`（默认）：使用启动时预加载的 sherpa-onnx LocalASREngine
- `services.asr.local_asr = false` 且 `services.asr.base_url` 已配置：转发到远程 ASR
- 两者皆不可用时返回 503

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
| `session.created` | 会话创建成功 |
| `session.updated` | 会话配置更新成功 |
| `input_audio_buffer.speech_started` | 检测到用户开始说话 |
| `input_audio_buffer.speech_stopped` | 检测到用户说完 |
| `conversation.item.input_audio_transcription.completed` | ASR 转写结果 |
| `response.created` | 响应开始 |
| `response.output_audio_transcript.delta` | LLM 文本增量 |
| `response.output_audio.delta` | TTS 音频增量（base64 编码 PCM16） |
| `response.output_audio_transcript.done` | 文本输出完成 |
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
| Omni 多模态模型 | LLM 推理（A/B 模式通用） | `http://localhost:8000/v1` |
| Qwen3-TTS | 语音合成（预设音色 + 声音克隆，24kHz） | `http://localhost:8091/v1` |
| VoxCPM2 | 语音合成（音色克隆 + 情感指令，48kHz→24kHz） | `http://localhost:8093/v1` |
| GSV-TTS-Lite | 语音合成（音色风格解耦，32kHz，可选） | `http://localhost:8001` |
| 远程 ASR | 语音识别（local_asr=false 时） | `http://localhost:8082/v1` |

Omni API 兼容 OpenAI Chat Completions 接口格式，TTS API 兼容 OpenAI Speech 接口格式。三种 TTS 后端通过 `services.tts.model` 切换，均支持 HTTP 分句流式和 WebSocket 逐字流式两种模式。

## 更新记录

### 2026-06-16

- **新增 HTTP ASR 端点**：`POST /v1/audio/transcriptions`，OpenAI Whisper 协议兼容
  - 支持 WAV/FLAC/OGG/MP3 等 soundfile 可解码格式，或原始 int16 PCM
  - `local_asr=true` 时复用启动预加载的 LocalASREngine 单例（无重复模型加载）
  - `local_asr=false` 时转发到 `services.asr.base_url`，可对接 funasr/SenseVoice 等远程服务
  - 共用 `security.auth_token` Bearer 认证
- 新增依赖：`soundfile>=0.12.0`、`librosa>=0.10.0`

### 2026-06-11

- **声音克隆**：Qwen3-TTS 和 VoxCPM2 均支持 `ref_audio`/`ref_text` 声音克隆配置
  - 配置统一：`services.tts.ref_audio` + `services.tts.ref_text`
  - Qwen3-TTS：`ref_audio`/`ref_text` 直接传递给 API
  - VoxCPM2：`ref_audio`/`ref_text` 直接传递给 API（传 `ref_text` 启用终极克隆模式）
- **增加TTS支持**: 新增VoxCPM2 TTS支持 基于vllm-omni部署的标准openai接口
  - 流式推理：输出 48kHz PCM16 自动 2:1 decimation 到 24kHz

### 2026-05-30

- 新增 Tool Calling（工具调用）支持，兼容 Qwen 工具调用协议

### 2026-05-29

- **ASR 引擎迁移**：FunASR → sherpa-onnx，本地 ASR 推理从 PyTorch 迁移至 ONNX Runtime
  - 模型源从 `iic/SenseVoiceSmall` 更换为 `xiaowangge/sherpa-onnx-sense-voice-small`
  - 优先使用 Q8 量化模型（228MB），RTF 从 0.009~0.049（FunASR GPU）优化至 0.006~0.018（sherpa-onnx CPU Q8）
- **依赖优化**：`requirements.txt` 移除 pytorch/funasr/transformers等难搞的依赖，降低部署难度

### 2026-05-23

- 修正 GSV-TTS-Lite SSE 音频解析，简化 ASR 模型路径解析

### 2026-05-22

- 新增本地 ASR `device` 配置项（cuda/cpu）
- 修复 FunASR/ModelScope 本地 ASR 模型路径解析兼容性

### 2026-05-21

- 新增 Omni 和 TTS 服务的 `api_key` 配置，支持 OpenAI 兼容接口认证

### 2026-05-16

- 优化端到端延迟
- 修复打断时残留图片数据未清理

### 2026-05-12

- 初始版本发布，支持 A/B 双模式实时语音对话

## License

MIT License
