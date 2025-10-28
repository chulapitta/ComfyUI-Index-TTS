# ComfyUI-Index-TTS

> **🍴 This is a Fork** - Enhanced ComfyUI implementation of IndexTTS with advanced multi-character support and emotion control features.
> 
> **Fork Chain**: [IndexTTS](https://github.com/index-tts/index-tts) → [chenpipi0807/ComfyUI-Index-TTS](https://github.com/chenpipi0807/ComfyUI-Index-TTS) → **This Fork**  
> **This Fork's Enhancements**: Multi-character voice synthesis, emotion control system, automatic emotion analysis, structured text parsing, and advanced IndexTTS Pro node.

---

## English Documentation

### 📢 Disclaimer

This project is a fork of [chenpipi0807/ComfyUI-Index-TTS](https://github.com/chenpipi0807/ComfyUI-Index-TTS), which itself is based on the open-source IndexTTS project from Bilibili. This fork adds advanced multi-character synthesis and emotion control features. However, it must be emphasized that **this project is strictly prohibited for any illegal purposes or any behavior related to copyright infringement!** This project is only for communication and learning within the open-source community to promote technology sharing and innovation, aiming to provide beneficial references and learning resources for developers.

It is hereby solemnly declared that all personal use behaviors are not associated with the developer or this project itself. The developer does not assume any responsibility for users' behaviors, and users should bear all risks and legal responsibilities that may arise during use. Please use this project reasonably and legally under the premise of complying with laws, regulations, and related provisions.

### 🎯 Overview

ComfyUI-Index-TTS is an enhanced ComfyUI custom node that enables high-quality text-to-speech conversion using IndexTTS models. This fork builds upon chenpipi0807's excellent ComfyUI implementation and adds powerful multi-character voice synthesis, emotion control, and novel reading capabilities with advanced features like automatic emotion analysis and structured text processing.

### 🎭 Key Features Added in This Fork

#### **IndexTTS Pro Node** - Multi-Character Voice Synthesis
- **Multi-Character Support**: Handle up to 5 different characters + narrator with individual voice settings
- **Advanced Emotion Control**: Three emotion modes for sophisticated expression control
- **Structured Text Processing**: Parse and process novel-style text with character tags
- **Automatic Audio Concatenation**: Seamless stitching of multi-character dialogues
- **Subtitle Generation**: Time-aligned subtitles in JSON and simplified formats

#### **Enhanced Emotion System** (IndexTTS-2)
1. **Explicit Emotion Tags**: `<Character1 emo="excited and joyful">Dialogue</Character1>`
2. **Automatic Emotion Analysis**: AI-powered emotion detection using Qwen model
3. **Emotion Suppression**: `<Character1 emo="">Dialogue</Character1>` to disable emotion
4. **Per-Line Emotion Weight**: `<Character1 emo="sad" ew="0.8">Dialogue</Character1>` for fine-tuned intensity
5. **Per-Line Pause Control**: `<Character1 pause="2.0">Dialogue</Character1>` for custom timing

#### **Supported Emotions**
- **angry** - Angry, furious tone
- **happy** - Joyful, cheerful tone  
- **afraid** - Fearful, nervous tone
- **disgusted** - Disgusted, dissatisfied tone
- **sad** - Sorrowful, melancholic tone
- **melancholic** - Depressed, gloomy tone
- **surprised** - Surprised, shocked tone
- **calm** - Natural, peaceful tone

#### **Advanced Text Format Support**
```xml
<Narrator>Narrative content here</Narrator>
<Character1 emo="excited">Character 1 dialogue with emotion</Character1>
<Character2 emo="sad" ew="0.8">Character 2 with custom emotion intensity</Character2>
<Character3 pause="2.0">Character 3 with custom pause duration</Character3>
<Character4 emo="angry" ew="0.9" pause="1.5">Full control: emotion + intensity + pause</Character4>
<Character5 emo="" pause="0">Character 5 dialogue (emotion suppressed, no pause)</Character5>
```

#### **Attribute Reference**
- **emo="text"**: Emotion description (e.g., "excited", "sad and disappointed")
- **emo=""**: Explicitly disable emotion for this line
- **ew="0.0-1.0"**: Per-line emotion weight (overrides global emotion_weight)
- **pause="0.0-10.0"**: Per-line pause duration in seconds (overrides global pause_between_lines)

### Basic Usage
   - Add **Index TTS Pro** node in ComfyUI
   - Connect narrator audio and optional character audio files
   - Input structured text with character tags
   - Configure emotion settings and run workflow

### 🔧 Key Parameters

- **emotion_weight** (0.0-1.0): Controls emotion expression intensity
- **auto_emotion** (True/False): Enable AI-powered emotion analysis  
- **pause_between_lines** (0.0-5.0): Pause duration between lines in seconds
- **model_version**: Index-TTS / IndexTTS-1.5 / IndexTTS-2 (recommended)

### 💡 Optimization Tips

#### Voice Quality
- Use high-quality reference audio (clear, 10-30 seconds, no noise)
- Adjust `temperature` (0.7-0.9) for better results
- Increase `repetition_penalty` (10.0-12.0) for voice consistency

#### Emotion Control
- **Daily conversation**: `emotion_weight = 0.4-0.6`
- **Dramatic performance**: `emotion_weight = 0.7-0.9`  
- **Formal speech**: `emotion_weight = 0.2-0.4`
- Use descriptive emotion text: "excited and joyful" > "happy"

### 📁 Example Workflows
- `workflow/TTS2.json` - IndexTTS-2 modular workflow
- `workflow/读小说用这个.json` - Novel reading workflow with IndexTTS Pro

---

## 中文文档 / Chinese Documentation

## 免责声明

本项目基于B站开源项目进行二次开发，由本人对项目进行了ComfyUI的实现，并进行了部分功能优化与调整与进阶功能的开发。然而，需要强调的是，本项目严禁用于任何非法目的以及与侵犯版权相关的任何行为！本项目仅用于开源社区内的交流与学习，以促进技术共享与创新，旨在为开发者提供有益的参考和学习资源。

在此郑重声明，本项目所有个人使用行为与开发者本人及本项目本身均无任何关联。开发者对于项目使用者的行为不承担任何责任，使用者应自行承担使用过程中可能产生的所有风险和法律责任。请广大使用者在遵守法律法规及相关规定的前提下，合理、合法地使用本项目，维护开源社区的良好秩序与健康发展。

感谢您的理解与支持！

使用IndexTTS模型在ComfyUI中实现高质量文本到语音转换的自定义节点。支持中文和英文文本，可以基于参考音频复刻声音特征。

![示例截图1](https://github.com/user-attachments/assets/41960425-f739-4496-9520-8f9cae34ff51)
![示例截图2](https://github.com/user-attachments/assets/1ff0d1d0-7a04-4d91-9d53-cd119250ed67)
![微信截图_20250605215845](https://github.com/user-attachments/assets/d5eb22f6-2ca2-40cf-a619-d709746f83e3)



## 最新更新（重要）

### 🎭 IndexTTS Pro 节点（推荐小说朗读）
专为小说阅读设计的一体化解决方案，支持：
- **多角色语音合成**：一次处理整篇小说，自动分配不同角色声音
- **高级情感控制系统**：支持显式情感标签、自动情感分析、情感抑制三种模式
- **行级精细控制**：每行可独立设置情感强度(`ew`)和停顿时长(`pause`)
- **智能文本解析**：自动识别角色对话和旁白，支持复杂结构化文本格式
- **音频拼接优化**：智能行间停顿控制，生成流畅的长篇音频
- **字幕生成**：自动生成时间轴对齐的字幕文件（JSON + 简化格式）

### 🔧 分离式TTS2节点（推荐高级用户）
功能拆分为四个核心节点，方便在 ComfyUI 中按需组合：
- Index TTS 2 - Base（基础合成）
- Index TTS 2 - Emotion Audio（基于参考音频情绪复刻）
- Index TTS 2 - Emotion Vector（基于情绪向量复刻）
- Index TTS 2 - Emotion Text（基于情绪文本复刻）

基础工作流已更新，详见./workflow/TTS2.json.

<img width="3090" height="1389" alt="image" src="https://github.com/user-attachments/assets/b12dae62-0ae3-49a7-99a9-f153218328fa" />


TTS2 模型下载与放置位置（全部放到 `./ComfyUI/models/IndexTTS-2/`）：

1) 基础模型
   - 页面：[TTS2](https://huggingface.co/IndexTeam/IndexTTS-2/tree/main)
   - 放置：`.\ComfyUI\models\IndexTTS-2`

2) qwen 模型（情绪分类）
   - 页面：[IndexTTS-2/qwen0.6bemo4-merge](https://huggingface.co/IndexTeam/IndexTTS-2/tree/main/qwen0.6bemo4-merge)
   - 放置：`.\ComfyUI\models\IndexTTS-2\qwen0.6bemo4-merge\`

3) semantic codec（MaskGCT 语义编码器）
   - 页面：[https://huggingface.co/amphion/MaskGCT/tree/main/semantic_codec](https://huggingface.co/amphion/MaskGCT/tree/main/semantic_codec)
   - 直链：[https://huggingface.co/amphion/MaskGCT/resolve/main/semantic_codec/model.safetensors](https://huggingface.co/amphion/MaskGCT/resolve/main/semantic_codec/model.safetensors)
   - 放置：`.\ComfyUI\models\IndexTTS-2\semantic_codec\model.safetensors`

4) CampPlus 说话人嵌入
   - 页面：[https://huggingface.co/funasr/campplus](https://huggingface.co/funasr/campplus)
   - 直链：[https://huggingface.co/funasr/campplus/resolve/main/campplus_cn_common.bin](https://huggingface.co/funasr/campplus/resolve/main/campplus_cn_common.bin)
   - 放置：`.\ComfyUI\models\IndexTTS-2\campplus_cn_common.bin`

5) Wav2Vec2Bert 特征提取器（facebook/w2v-bert-2.0）
   - 页面：[https://huggingface.co/facebook/w2v-bert-2.0/tree/main](https://huggingface.co/facebook/w2v-bert-2.0/tree/main)
   - 放置（离线优先）：`.\ComfyUI\models\IndexTTS-2\w2v-bert-2.0\`（整个仓库文件夹，包含 `config.json`、`model.safetensors`、`preprocessor_config.json` 等）
   - 若未放置本地文件夹，将自动下载到 HF 缓存：`.\ComfyUI\models\IndexTTS-2\hf_cache\`

6) BigVGAN 声码器
   - 名称读取自 `config.yaml` 的 `vocoder.name`（示例：`nvidia/bigvgan_v2_22khz_80band_256x`）
   - 建议：提前将对应模型完整缓存到 `.\ComfyUI\models\IndexTTS-2\bigvgan\` 内

7) 其他本地直读文件（需与 `config.yaml` 一致）：
   - `gpt.pth`（`cfg.gpt_checkpoint`）
   - `s2mel.pth`（`cfg.s2mel_checkpoint`）
   - `bpe.model`（`cfg.dataset.bpe_model`）
   - `wav2vec2bert_stats.pt`（`cfg.w2v_stat`）
   - 语义编码配置（如 `repcodec.json`，若需要，`cfg.semantic_codec`）
   - `emo_matrix`（例如 `feat2.pt`）
   - `spk_matrix`（例如 `feat1.pt`）
   - `qwen0.6bemo4-merge\`（`cfg.qwen_emo_path` 指定目录）

示例目录结构（部分）：

```text
ComfyUI/models/IndexTTS-2/
│  .gitattributes
│  bpe.model
│  campplus_cn_common.bin
│  config.yaml
│  feat1.pt
│  feat2.pt
│  gpt.pth
│  README.md
│  s2mel.pth
│  wav2vec2bert_stats.pt
│
├─bigvgan
│  └─bigvgan_v2_22khz_80band_256x
│          .gitattributes
│          .gitignore
│          activations.py
│          bigvgan.py
│          bigvgan_discriminator_optimizer.pt
│          bigvgan_discriminator_optimizer_3msteps.pt
│          bigvgan_generator.pt
│          bigvgan_generator_3msteps.pt
│          config.json
│          env.py
│          LICENSE
│          meldataset.py
│          README.md
│          utils.py
│
├─hf_cache
├─qwen0.6bemo4-merge
│      added_tokens.json
│      chat_template.jinja
│      config.json
│      generation_config.json
│      merges.txt
│      model.safetensors
│      Modelfile
│      special_tokens_map.json
│      tokenizer.json
│      tokenizer_config.json
│      vocab.json
│
├─semantic_codec
│      model.safetensors
│
└─w2v-bert-2.0
        .gitattributes
        config.json
        conformer_shaw.pt
        model.safetensors
        preprocessor_config.json
        README.md
```

> 提示：若你只使用旧版 IndexTTS/IndexTTS-1.5，可忽略上述 TTS2 模型放置步骤。

### 一键下载脚本（推荐）

- 脚本位置：`ComfyUI/custom_nodes/ComfyUI-Index-TTS/TTS2_download.py`
- 作用：自动下载并放置上述所有 TTS2 所需模型文件，支持断点续传、国内镜像（HF_ENDPOINT=hf-mirror.com）、本地缓存（HF_HOME=./ComfyUI/models/IndexTTS-2/hf_cache）。

使用方法（Windows/PowerShell）：

```powershell
python .\ComfyUI\custom_nodes\ComfyUI-Index-TTS\TTS2_download.py
```

- 运行后根据提示选择 2 使用国内镜像（默认）或 1 使用官方源。
- 依赖：`huggingface_hub`（必须）；可选加速：`hf_transfer`、`hf_xet`。

```powershell
python -m pip install -U huggingface_hub
# 可选加速：
python -m pip install -U hf_transfer
python -m pip install -U "huggingface_hub[hf_xet]"
```

### 显存/缓存控制（新功能）

- 新增节点：`Index TTS 2 - Cache Control`
  - 输出：`cache_control`（类型：DICT），包含 `{"keep_cached": true/false}`。
  - 用法：将该输出连到以下任一/多个节点的 `cache_control` 输入上：
    - `Index TTS 2 - Base`
    - `Index TTS 2 - Emotion Audio`
    - `Index TTS 2 - Emotion Vector`
    - `Index TTS 2 - Emotion Text`

- 行为说明：
  - 关闭（默认）：本次推理结束后自动卸载 TTS2 模型并清理 CUDA 缓存，降低显存驻留峰值，适合 12GB 显卡日常使用。
  - 开启：保留已加载的权重（尽量驻留，视环境/模式），连续多次生成更快，但显存占用更高。调参批量测试时可临时打开，用完关闭。

## 功能特点

### 🎯 核心功能
- 支持中文和英文文本合成
- 基于参考音频复刻声音特征（变声功能）
- 支持调节语速（原版不支持后处理实现效果会有一点折损）
- 多种音频合成参数控制
- Windows兼容（无需额外依赖）

### 🎭 IndexTTS Pro 高级功能
- **多角色支持**：最多支持5个不同角色 + 旁白，每个角色可配置独立的参考音频
- **智能情感控制**：
  - 显式情感标签：`<Character1 emo="开心而兴奋">对话内容</Character1>`
  - 自动情感分析：基于Qwen模型自动从对话内容分析情感
  - 情感抑制模式：`<Character1 emo="">对话内容</Character1>` 明确禁用情感
  - 行级情感强度：`<Character1 emo="悲伤" ew="0.8">对话</Character1>` 精确控制每行情感强度
- **精细停顿控制**：
  - 全局停顿设置：统一控制所有行间停顿
  - 行级停顿控制：`<Character1 pause="2.0">对话</Character1>` 自定义每行后的停顿时长
  - 零停顿支持：`<Character1 pause="0">对话</Character1>` 实现快速对话
- **结构化文本支持**：
  ```xml
  <Narrator>旁白文本</Narrator>
  <Character1 emo="excited" ew="0.9" pause="1.0">兴奋对话</Character1>
  <Character2 emo="sad" ew="0.4" pause="2.5">悲伤对话（长停顿）</Character2>
  <Character3 pause="0.1">快速回应</Character3>
  <Character4 emo="angry" ew="0.8">愤怒对话</Character4>
  <Character5 emo="">平静叙述（无情感）</Character5>
  ```
- **音频优化**：
  - 智能音频拼接和归一化
  - 自动生成时间轴字幕（JSON + 简化格式）
  - 支持长篇文本的内存优化处理

### 📊 支持的情感类型
IndexTTS-2模式下支持以下情感：
- 愤怒 (angry) - 生气、愤怒的语调
- 高兴 (happy) - 开心、愉悦的语调  
- 恐惧 (afraid) - 害怕、紧张的语调
- 反感 (disgusted) - 厌恶、不满的语调
- 悲伤 (sad) - 伤心、难过的语调
- 低落 (melancholic) - 沮丧、忧郁的语调
- 惊讶 (surprised) - 惊讶、震惊的语调
- 自然 (calm) - 平静、自然的语调


## 废话两句

- 生成的很快，真的很快！而且竟然也很像！！！
- 效果很好，感谢小破站的开源哈哈哈哈哈 
- 如果你想体验一下效果 附赠道友B站的传送阵[demo](https://huggingface.co/spaces/IndexTeam/IndexTTS)
- 如果你不知道去哪找音频，那我建议你去[隔壁](https://drive.google.com/drive/folders/1AyB3egmr0hAKp0CScI0eXJaUdVccArGB)偷哈哈哈哈哈

## 演示案例

以下是一些实际使用效果演示：

| 参考音频 | 输入文本 | 推理结果 |
|---------|---------|---------|
| <video src="https://github.com/user-attachments/assets/5e8cb570-242f-4a16-8472-8a64a23183fb"></video> | 我想把钉钉的自动回复设置成"服务器繁忙，请稍后再试"，仅对老板可见。  我想把钉钉的自动回复设置成"服务器繁忙，请稍后再试"，仅对老板可见。 | <video src="https://github.com/user-attachments/assets/d8b89db3-5cf5-406f-b930-fa75d13ff0bd"></video> |
| <video src="https://github.com/user-attachments/assets/8e774223-e0f7-410b-ae4e-e46215e47e96"></video> | 我想把钉钉的自动回复设置成"服务器繁忙，请稍后再试"，仅对老板可见。 | <video src="https://github.com/user-attachments/assets/6e3e63ed-2d3d-4d5a-bc2e-b42530748fa0"></video> |

- 长文本测试：

<video src="https://github.com/user-attachments/assets/6bfa35dc-1a30-4da0-a4dc-ac3def25452b"></video>

- 多角色小说测试：

<video src="https://github.com/user-attachments/assets/6d4737f4-9d75-431e-bb11-fe3e86a4ab0e"></video>



## 更新日志

### 2025-10-28（最新 - 本Fork新增功能）

#### 🎯 行级精细控制系统
- **行级情感强度控制**：新增 `ew` 属性，支持每行独立设置情感强度
  - 使用示例：`<Character1 emo="愤怒" ew="0.9">非常生气的对话</Character1>`
  - 优先级：行级设置 > 全局 emotion_weight 参数
  - 取值范围：0.0-1.0，自动验证和限制范围
- **行级停顿控制**：新增 `pause` 属性，支持每行独立设置停顿时长
  - 使用示例：`<Character1 pause="2.5">需要长停顿的对话</Character1>`
  - 优先级：行级设置 > 全局 pause_between_lines 参数
  - 取值范围：0.0-10.0 秒，支持零停顿实现快速对话
- **组合使用**：支持多属性同时使用
  - 完整示例：`<Character1 emo="惊讶而震惊" ew="0.8" pause="1.5">什么？！</Character1>`
- **智能日志**：详细显示每行使用的参数来源（行级 vs 全局设置）

#### 🎭 情感控制增强
- **三重情感模式完善**：显式情感、自动情感分析、情感抑制三种模式协同工作
- **情感强度微调**：0.0-1.0 无级调节，支持从微妙表达到强烈情感的全范围控制
- **自动情感分析优化**：基于Qwen模型的智能情感识别，配合行级强度控制

#### 📖 实用示例
```xml
<Narrator>故事开始了...</Narrator>
<Character1 emo="兴奋而期待" ew="0.9" pause="0.5">太好了！</Character1>
<Character2 emo="怀疑" ew="0.6" pause="2.0">真的吗？</Character2>
<Character1 pause="0.1">当然！</Character1>
<Character3 emo="" pause="1.0">请保持安静。</Character3>
```

### 2025-06-24

- pro节点新增了对于字幕的json输出，感谢@qy8502提供的玩法思路

![image](https://github.com/user-attachments/assets/e7f5e92a-7f76-48a1-ba01-86143d10d359)


### 2025-06-05

- 改进了小说文本解析器（Novel Text Parser）的功能
  - 增加了对预格式化文本的检测和处理
  - 优化了对话检测和角色识别算法
  - 改进了中文角色名称的识别
  - 支持引号中的对话自动识别

## 多角色小说文本解析

本项目包含一个专门用于解析小说文本的节点（Novel Text Structure Node），可以将普通小说文本解析为多角色对话结构，以便生成更加自然的多声音TTS效果。

### 使用说明

- 节点会尝试自动识别小说中的角色对话和旁白部分
- 对话部分会标记为`<CharacterX>`形式（X为数字，最多支持5个角色）
- 旁白部分会标记为`<Narrator>`
- 解析后的文本可直接用于多声音TTS生成

### 局限性

- 当前解析算法并不完美，复杂的小说结构可能导致错误的角色识别
- 对于重要文本，建议使用LLM（如GPT等）手动拆分文本为以下格式：

```
<Narrator>少女此时就站在院墙那边，她有一双杏眼，怯怯弱弱。</Narrator>
<Narrator>院门那边，有个嗓音说：</Narrator>
<Character1>"你这婢女卖不卖？"</Character1>
<Narrator>宋集薪愣了愣，循着声音转头望去，是个眉眼含笑的锦衣少年，站在院外，一张全然陌生的面孔。</Narrator>
<Narrator>锦衣少年身边站着一位身材高大的老者，面容白皙，脸色和蔼，轻轻眯眼打量着两座毗邻院落的少年少女。</Narrator>
<Narrator>老者的视线在陈平安一扫而过，并无停滞，但是在宋集薪和婢女身上，多有停留，笑意渐渐浓郁。</Narrator>
<Narrator>宋集薪斜眼道：</Narrator>
<Character2>"卖！怎么不卖！"</Character2>
<Narrator>那少年微笑道：</Narrator>
<Character1>"那你说个价。"</Character1>
<Narrator>少女瞪大眼眸，满脸匪夷所思，像一头惊慌失措的年幼麋鹿。</Narrator>
<Narrator>宋集薪翻了个白眼，伸出一根手指，晃了晃，</Narrator>
<Character2>"白银一万两！"</Character2>
<Narrator>锦衣少年脸色如常，点头道：</Narrator>
<Character1>"好。"</Character1>
<Narrator>宋集薪见那少年不像是开玩笑的样子，连忙改口道：</Narrator>
<Character2>"是黄金万两！"</Character2>
<Narrator>锦衣少年嘴角翘起，道：</Narrator>
<Character1>"逗你玩的。"</Character1>
<Narrator>宋集薪脸色阴沉。</Narrator>
```

### 示例用法

1. 将小说文本输入到 Novel Text Structure 节点
2. 连接输出到 Index TTS Pro 节点
3. 设置不同角色的语音
4. 运行工作流生成多声音小说朗读
5. 实在不会看我最新增加的工作流
6. 如果你想在comfyui中一站式完成这个，我推荐你使用各类的llm节点，比如[kimichat](https://github.com/chenpipi0807/PIP_KIMI2comfyui)
7. 我也提供了一段llm提示词模板，你可以在llm_prompt模板.txt中看到他


### 2025-05-18

- 优化了长期以来transformers库4.50+版本的API变化与原始IndexTTS模型代码不兼容导致的生成报错问题


### 2025-05-16

- 新增对**IndexTTS-1.5**模型的支持
  - 现在可以在UI中通过下拉菜单切换不同版本的模型
  - 支持原始的Index-TTS和新的IndexTTS-1.5模型
  - 切换模型时会自动加载相应版本，无需重启ComfyUI
 
  ![微信截图_20250516182957](https://github.com/user-attachments/assets/ce13f02c-9834-43b8-82e9-5567bb226280)
  

### 2025-05-11
- 增加了seed功能，现在linux也可以重复执行抽卡了
- 增加了对 Apple Silicon MPS 设备的检测（仍需测试反馈~）


### 2025-04-23

![微信截图_20250423175608](https://github.com/user-attachments/assets/f2b15d8a-3453-4c88-b609-167b372aab74)


- 新增 **Audio Cleaner** 节点，用于处理TTS输出音频中的混响和杂音问题
  - 该节点可以连接在 Index TTS 节点之后，优化生成音频的质量
  - 主要功能：去除混响、降噪、频率滤波和音频归一化
  - 适用于处理有杂音或混响问题的TTS输出

- 修复了对于transformers版本强依赖的问题

#### Audio Cleaner 参数说明

**必需参数**：：
- **audio**: 输入音频（通常为 Index TTS 节点的输出）
- **denoise_strength**: 降噪强度（0.1-1.0，默认0.5）
  - 值越大，降噪效果越强，但可能影响语音自然度
- **dereverb_strength**: 去混响强度（0.0-1.0，默认0.7）
  - 值越大，去混响效果越强，适合处理在回声环境下录制的参考音频

**可选参数**：：
- **high_pass_freq**: 高通滤波器频率（20-500Hz，默认100Hz）
  - 用于过滤低频噪音，如环境嗡嗡声
- **low_pass_freq**: 低通滤波器频率（1000-16000Hz，默认8000Hz）
  - 用于过滤高频噪音
- **normalize**: 是否归一化音频（"true"或"false"，默认"true"）
  - 开启可使音量更均衡

#### 使用建议

- 对于有明显混响的音频，将 `dereverb_strength` 设置为 0.7-0.9
- 对于有背景噪音的音频，将 `denoise_strength` 设置为 0.5-0.8
- 如果处理后音频听起来不自然，尝试减小 `dereverb_strength` 和 `denoise_strength`
- 高通和低通滤波器可以微调以获得最佳人声效果


### 2025-04-25
- 优化了阿拉伯数字的发音判断问题；可以参考这个case使用：“4 0 9 0”会发音四零九零，“4090”会发音四千零九十； 


### 2025-04-26
- 优化英文逗号导致吞字的问题；


### 2025-04-29
- 修正了语言模式切换en的时候4090依然读中文的问题，auto现在会按照中英文占比确定阿拉伯数字读法
- 新增了从列表读取音频的方法，同时新增了一些音色音频供大家玩耍；你可以将自己喜欢的音频放入 ComfyUI-Index-TTS\TimbreModel 里，当然也很鼓励你能把好玩的声音分享出来。
- 示例用法如图：

![微信截图_20250429112255](https://github.com/user-attachments/assets/a0af9a5b-7609-4c34-adf5-e14321b379a7)


## 安装

### 安装节点

1. 将此代码库克隆或下载到ComfyUI的`custom_nodes`目录：

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/chenpipi0807/ComfyUI-Index-TTS.git
   ```

2. 安装依赖： 安装依赖：

   ```bash
   cd ComfyUI-Index-TTS
   .\python_embeded\python.exe -m pip install -r requirements.txt

   git pull # 更新很频繁你可能需要
   ```

### 下载模型

#### 原始版本 (Index-TTS)

1. 从[Hugging Face](https://huggingface.co/IndexTeam/Index-TTS/tree/main)或者[魔搭](https://modelscope.cn/models/IndexTeam/Index-TTS)下载IndexTTS模型文件
2. 将模型文件放置在`ComfyUI/models/Index-TTS`目录中（如果目录不存在，请创建）
3. 模型文件夹结构：
   
   ```
   ComfyUI/models/Index-TTS/
   ├── .gitattributes
   ├── bigvgan_discriminator.pth
   ├── bigvgan_generator.pth
   ├── bpe.model
   ├── config.yaml
   ├── configuration.json
   ├── dvae.pth
   ├── gpt.pth
   ├── README.md
   └── unigram_12000.vocab
   ```
   
   确保所有文件都已完整下载，特别是较大的模型文件如`bigvgan_discriminator.pth`(1.6GB)和`gpt.pth`(696MB)。

#### 新版本 (IndexTTS-1.5)

1. 从[Hugging Face](https://huggingface.co/IndexTeam/IndexTTS-1.5/tree/main)下载IndexTTS-1.5模型文件
2. 将模型文件放置在`ComfyUI/models/IndexTTS-1.5`目录中（如果目录不存在，请创建）
3. 模型文件夹结构与Index-TTS基本相同，但文件大小和内容会有所不同：
   
   ```
   ComfyUI/models/IndexTTS-1.5/
   ├── .gitattributes
   ├── bigvgan_discriminator.pth
   ├── bigvgan_generator.pth
   ├── bpe.model
   ├── config.yaml
   ├── configuration.json
   ├── dvae.pth
   ├── gpt.pth
   ├── README.md
   └── unigram_12000.vocab
   ```

## 使用方法

1. 在ComfyUI中，找到并添加`Index TTS`节点
2. 连接参考音频输入（AUDIO类型）
3. 输入要转换为语音的文本
4. 调整参数（语言、语速等）
5. 运行工作流获取生成的语音输出

### 示例工作流

项目包含一个基础工作流示例，位于`workflow/workflow.json`，您可以在ComfyUI中通过导入此文件来快速开始使用。

## 参数说明

### 必需参数

- **text**: 要转换为语音的文本（支持中英文）
- **reference_audio**: 参考音频，模型会复刻其声音特征
- **model_version**: 模型版本选择，可选项：
  - `Index-TTS`: 原始模型版本（默认）
  - `IndexTTS-1.5`: 新版本模型
- **language**: 文本语言选择，可选项：
  - `auto`: 自动检测语言（默认）
  - `zh`: 强制使用中文模式
  - `en`: 强制使用英文模式
- **speed**: 语速因子（0.5~2.0，默认1.0）

### 可选参数

以下参数适用于高级用户，用于调整语音生成质量和特性：

- **temperature** (默认1.0): 控制生成随机性，较高的值增加多样性但可能降低稳定性
- **top_p** (默认0.8): 采样时考虑的概率质量，降低可获得更准确但可能不够自然的发音
- **top_k** (默认30): 采样时考虑的候选项数量
- **repetition_penalty** (默认10.0): 重复内容的惩罚系数
- **length_penalty** (默认0.0): 生成内容长度的调节因子
- **num_beams** (默认3): 束搜索的宽度，增加可提高质量但降低速度
- **max_mel_tokens** (默认600): 最大音频token数量
- **sentence_split** (默认auto): 句子拆分方式

## 音色优化建议

要提高音色相似度：

- 使用高质量的参考音频（清晰、无噪音）
- 尝试调整`temperature`参数（0.7-0.9范围内效果较好）
- 增加`repetition_penalty`（10.0-12.0）可以提高音色一致性
- 对于长文本，确保`max_mel_tokens`足够大

## 故障排除


- 如果出现“模型加载失败”，检查模型文件是否完整且放置在正确目录
- 对于Windows用户，无需额外安装特殊依赖，节点已优化
- 如果显示CUDA错误，尝试重启ComfyUI或减少`num_beams`值
- 如果你是pytorch2.7运行报错，短期无法适配，请尝试降级方案(.\python_embeded\python.exe -m pip install transformers==4.48.3)



## 鸣谢

- 基于原始[IndexTTS](https://github.com/index-tts/index-tts)模型
- 感谢ComfyUI社区的支持
- 感谢使用！
- 

## 许可证

请参考原始IndexTTS项目许可证。



