import os
import sys
import numpy as np
import torch
import tempfile
import soundfile as sf
import time
import re
import json
from pathlib import Path
import random

# 确保当前目录在导入路径中 / Ensure current directory is in import path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入TTS模型 / Import TTS models
from .tts_models import IndexTTSModel
from .indextts2 import IndexTTS2Loader, IndexTTS2Engine


class IndexTTSProNode:
    """
    ComfyUI的IndexTTS Pro节点，专用于小说阅读，支持多角色语音合成和情感控制
    ComfyUI IndexTTS Pro node for novel reading with multi-character voice synthesis and emotion control
    
    支持格式 / Supported format:
    <Narrator>旁白文本</Narrator>
    <Character1 emo="开心而兴奋">角色对话</Character1>
    <Character2 emo="悲伤而难过">悲伤的对话</Character2>
    
    情感控制仅在IndexTTS-2模型中可用 / Emotion control only available with IndexTTS-2 model
    
    情感模式 / Emotion modes:
    1. 显式情感 / Explicit emotion: 使用emo属性指定 / Use emo attribute
    2. 自动情感 / Automatic emotion: 启用auto_emotion从对话内容自动分析 / Enable auto_emotion for auto analysis
    3. 抑制情感 / Suppress emotion: 使用emo=""明确禁用情感 / Use emo="" to explicitly disable emotion
    
    支持的情感类型 / Supported emotion types:
    愤怒(angry), 高兴(happy), 恐惧(afraid), 反感(disgusted), 
    悲伤(sad), 低落(melancholic), 惊讶(surprised), 自然(calm)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "structured_text": ("STRING", {"multiline": True, "default": "<Narrator>This is a sample narrative text.<Character1 emo=\"excited\">Hello!<Narrator>He said cheerfully."}),
                "narrator_audio": ("AUDIO", {"description": "正文/旁白的参考音频 / Narrator reference audio"}),
                "model_version": (["Index-TTS", "IndexTTS-1.5", "IndexTTS-2"], {"default": "IndexTTS-2"}),
                "language": (["auto", "zh", "en"], {"default": "auto"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "character1_audio": ("AUDIO", {"description": "角色1的参考音频 / Character 1 reference audio"}),
                "character2_audio": ("AUDIO", {"description": "角色2的参考音频 / Character 2 reference audio"}),
                "character3_audio": ("AUDIO", {"description": "角色3的参考音频 / Character 3 reference audio"}),
                "character4_audio": ("AUDIO", {"description": "角色4的参考音频 / Character 4 reference audio"}),
                "character5_audio": ("AUDIO", {"description": "角色5的参考音频 / Character 5 reference audio"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 30, "min": 0, "max": 100, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 15.0, "step": 0.1}),
                "length_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "max_mel_tokens": ("INT", {"default": 1500, "min": 100, "max": 3000, "step": 50}),
                "do_sample": ("BOOLEAN", {"default": False}),
                "mode": (["Auto", "Duration", "Tokens"], {"default": "Auto"}),
                "emotion_weight": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "description": "情感强度控制 / Emotion intensity control (现已修复，支持情感文本模式 / Now fixed, supports emotion text mode)"}),
                "auto_emotion": ("BOOLEAN", {"default": False, "description": "自动情感分析 / Automatic emotion analysis from dialogue text (IndexTTS-2 only)"}),
                "pause_between_lines": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 5.0, "step": 0.01, "description": "行间停顿时长(秒) / Pause duration between lines (seconds)"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "INT", "STRING", "STRING",)
    RETURN_NAMES = ("audio", "seed", "Subtitle", "SimplifiedSubtitle",)
    FUNCTION = "generate_multi_voice_speech"
    CATEGORY = "audio"
    
    def __init__(self):
        # 根路径 / Root path
        self.models_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
        # 可用模型版本 / Available model versions
        self.model_versions = {
            "Index-TTS": os.path.join(self.models_root, "Index-TTS"),
            "IndexTTS-1.5": os.path.join(self.models_root, "IndexTTS-1.5"),
            "IndexTTS-2": os.path.join(self.models_root, "IndexTTS-2")
        }
        # 默认使用 IndexTTS-2 版本 / Default to IndexTTS-2
        self.current_version = "IndexTTS-2"
        self.model_dir = self.model_versions[self.current_version]
        # V1/V1.5 模型实例 / V1/V1.5 model instance
        self.tts_model = None
        # V2 模型实例 / V2 model instances
        self.tts2_loader = None
        self.tts2_engine = None
        
        print(f"[IndexTTS Pro] 初始化节点，可用模型版本 / Initializing node, available versions: {list(self.model_versions.keys())}")
        print(f"[IndexTTS Pro] 默认模型目录 / Default model directory: {self.model_dir}")
    
    def _init_model(self, model_version="Index-TTS"):
        """初始化TTS模型（延迟加载） / Initialize TTS model (lazy loading)
        
        Args:
            model_version: 模型版本，默认为 "Index-TTS" / Model version, defaults to "Index-TTS"
        """
        # 如果版本发生变化或模型未加载，重新加载模型 / Reload model if version changed or not loaded
        if self.current_version != model_version or (self.tts_model is None and self.tts2_engine is None):
            # 更新当前版本和模型目录 / Update current version and model directory
            if model_version in self.model_versions:
                self.current_version = model_version
                self.model_dir = self.model_versions[model_version]
                print(f"[IndexTTS Pro] 切换到模型版本 / Switching to model version: {model_version}, 目录 / directory: {self.model_dir}")
            else:
                print(f"[IndexTTS Pro] 警告 / Warning: 未知模型版本 / Unknown model version {model_version}，使用默认版本 / using default version {self.current_version}")
            
            # 如果已有模型，先释放资源 / Release existing model resources
            if self.tts_model is not None:
                print(f"[IndexTTS Pro] 卸载现有V1/V1.5模型 / Unloading existing V1/V1.5 model...")
                self.tts_model = None
            if self.tts2_engine is not None:
                print(f"[IndexTTS Pro] 卸载现有V2模型 / Unloading existing V2 model...")
                self.tts2_loader = None
                self.tts2_engine = None
            
            # 强制垃圾回收 / Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"[IndexTTS Pro] 开始加载模型版本 / Starting to load model version: {self.current_version}...")
            
            try:
                # 记录开始加载时间 / Record start time
                start_time = time.time()
                
                if model_version == "IndexTTS-2":
                    # 加载V2模型 / Load V2 model
                    print(f"[IndexTTS Pro] 加载IndexTTS-2模型 / Loading IndexTTS-2 model...")
                    self.tts2_loader = IndexTTS2Loader()
                    self.tts2_engine = IndexTTS2Engine(self.tts2_loader)
                    print(f"[IndexTTS Pro] IndexTTS-2模型加载完成 / IndexTTS-2 model loaded successfully")
                else:
                    # 加载V1/V1.5模型 / Load V1/V1.5 model
                    # 检查必要的模型文件 / Check required model files
                    required_files = ["gpt.pth", "config.yaml"]
                    missing_files = []
                    for file in required_files:
                        file_path = os.path.join(self.model_dir, file)
                        if not os.path.exists(file_path):
                            missing_files.append(file)
                        else:
                            file_size = os.path.getsize(file_path) / (1024*1024)  # 转换为MB / Convert to MB
                            print(f"[IndexTTS Pro] 找到模型文件 / Found model file: {file} ({file_size:.2f}MB)")
                    
                    if missing_files:
                        error_msg = f"模型 / Model {self.current_version} 缺少必要的文件 / missing required files: {', '.join(missing_files)}"
                        print(f"[IndexTTS Pro] 错误 / Error: {error_msg}")
                        raise FileNotFoundError(error_msg)
                    
                    # 使用tts_models.py中的IndexTTSModel实现 / Use IndexTTSModel from tts_models.py
                    self.tts_model = IndexTTSModel(model_dir=self.model_dir)
                
                # 记录加载完成时间 / Record completion time
                load_time = time.time() - start_time
                print(f"[IndexTTS Pro] 模型 / Model {self.current_version} 已成功加载 / loaded successfully，耗时 / time taken: {load_time:.2f}秒 / seconds")
            except Exception as e:
                import traceback
                print(f"[IndexTTS Pro] 初始化模型 / Model initialization {self.current_version} 失败 / failed: {e}")
                print(f"[IndexTTS Pro] 错误详情 / Error details:")
                traceback.print_exc()
                raise RuntimeError(f"初始化IndexTTS模型 / Initialize IndexTTS model {self.current_version} 失败 / failed: {e}")
    
    def _process_audio_input(self, audio_input):
        """处理ComfyUI的音频格式 / Process ComfyUI audio format
        
        Args:
            audio_input: ComfyUI的音频格式 / ComfyUI audio format
            
        Returns:
            tuple: (waveform, sample_rate) 元组 / tuple
        """
        if audio_input is None:
            return None
            
        if isinstance(audio_input, dict) and "waveform" in audio_input and "sample_rate" in audio_input:
            waveform = audio_input["waveform"]
            sample_rate = audio_input["sample_rate"]
            
            # 如果waveform是torch.Tensor，转换为numpy / If waveform is torch.Tensor, convert to numpy
            if isinstance(waveform, torch.Tensor):
                waveform_np = waveform.detach().cpu().numpy()
                # 处理不同维度 / Handle different dimensions
                if waveform_np.ndim == 3:
                    waveform_np = waveform_np[0, 0]  # [batch, channels, samples] -> [samples]
                elif waveform_np.ndim == 2:
                    waveform_np = waveform_np[0]  # [channels, samples] -> [samples]
                return (waveform_np.astype(np.float32), int(sample_rate))
                
            return (waveform, sample_rate)
            
        # 如果已经是元组格式 / If already tuple format
        elif isinstance(audio_input, tuple) and len(audio_input) == 2:
            return audio_input
            
        # 如果都不是，报错 / Otherwise raise error
        raise ValueError("参考音频格式不支持，应为 AUDIO 类型 / Reference audio format not supported, should be AUDIO type")
    
    def _parse_structured_text(self, structured_text):
        """解析结构化文本 / Parse structured text
        
        Args:
            structured_text: 结构化文本，支持情感属性 / Structured text, supports emotion attributes
                           e.g. "<Narrator>This is narrative text<Character1 emo="excited">Hello!</Character1>"
            
        Returns:
            list: 解析后的文本段落列表，每个元素为 (role, text, emotion) / List of parsed text segments, each element is (role, text, emotion)
        """
        segments = []
        # 简单匹配模式，捕获整个标签和内容 / Simple matching pattern to capture entire tag and content
        pattern = re.compile(r'<(Narrator|Character\d+)([^>]*)>(.*?)(?=<|$)', re.DOTALL)
        
        # 查找所有匹配 / Find all matches
        matches = pattern.findall(structured_text)
        
        # 如果找不到任何匹配，将整个文本作为旁白处理 / If no matches found, treat entire text as narrator
        if not matches:
            segments.append(("Narrator", structured_text.strip(), None))
        else:
            for role, attributes, text in matches:
                # 分析属性部分，查找emo="..." / Analyze attributes part, look for emo="..."
                emotion = None
                if attributes.strip():
                    emo_match = re.search(r'emo="([^"]*)"', attributes)
                    if emo_match:
                        emotion = emo_match.group(1).strip()  # 可能是空字符串或有内容 / Could be empty string or have content
                
                text = text.strip()
                if text:  # 只添加非空文本 / Only add non-empty text
                    segments.append((role, text, emotion))
                    
        return segments
    
    def _concatenate_audio(self, audio_segments):
        """连接多个音频段落
        
        Args:
            audio_segments: 音频段落列表，每个元素为 (waveform, sample_rate)
            
        Returns:
            tuple: 连接后的 (waveform, sample_rate)
        """
        if not audio_segments:
            return None
            
        # 确保所有段落的采样率相同
        sample_rate = audio_segments[0][1]
        
        # 过滤有效的音频段落
        valid_segments = []
        for idx, segment in enumerate(audio_segments):
            try:
                audio_data, seg_sample_rate = segment
                
                # 如果是第一个有效的音频段，设置采样率
                if not valid_segments:
                    sample_rate = seg_sample_rate
                    
                # 检查音频数据是否有效
                if audio_data is not None and isinstance(audio_data, np.ndarray):
                    # 确保是有效的numpy数组
                    if audio_data.size > 0:
                        valid_segments.append(audio_data)
                        print(f"[IndexTTS Pro] Added segment {idx+1}: shape={audio_data.shape}, dtype={audio_data.dtype}")
                    else:
                        print(f"[IndexTTS Pro] Warning: Skipping empty audio segment {idx+1} with shape: {audio_data.shape}")
                else:
                    # 打印数据类型信息以便调试
                    print(f"[IndexTTS Pro] Warning: Skipping invalid audio data of type: {type(audio_data)}")
                    if hasattr(audio_data, '__dict__'):
                        print(f"[IndexTTS Pro] Data attributes: {dir(audio_data)}")
                    print(f"[IndexTTS Pro] Data value: {str(audio_data)[:100]}...")
            except Exception as e:
                print(f"[IndexTTS Pro] Error processing segment {idx+1}: {e}")
        
        if not valid_segments:
            print("[IndexTTS Pro] Error: No valid audio segments to concatenate")
            return None
            
        # 连接所有有效的音频段落
        try:
            # 连接所有段落
            print(f"[IndexTTS Pro] Concatenating {len(valid_segments)} audio segments")
            concatenated = np.concatenate(valid_segments, axis=0)
            print(f"[IndexTTS Pro] Concatenated audio shape: {concatenated.shape}")
            
            # 确保音频数据是适当的格式
            if concatenated.ndim == 1:
                # 保持为1D格式，我们在返回前会转化为适当的维度
                print(f"[IndexTTS Pro] Audio is 1D array with {len(concatenated)} samples")
            elif concatenated.ndim > 2:
                # 如果维度过多，转为1D数组
                print(f"[IndexTTS Pro] Audio has too many dimensions: {concatenated.shape}, flattening")
                concatenated = concatenated.flatten()
                print(f"[IndexTTS Pro] Flattened to: {concatenated.shape}")
                
            return (concatenated, sample_rate)
        except Exception as e:
            print(f"[IndexTTS Pro] Error concatenating audio segments: {e}")
            import traceback
            traceback.print_exc()
            
            # 如果连接失败，返回第一个有效段落
            if valid_segments:
                print(f"[IndexTTS Pro] Falling back to first valid segment")
                first_segment = valid_segments[0]
                return (first_segment, sample_rate)
            
            print(f"[IndexTTS Pro] No valid segments found, returning None")
            return None
    
    def _seconds_to_time_format(self, seconds):
        """将秒数转换为分:秒.毫秒格式
        
        Args:
            seconds: 秒数(float)
            
        Returns:
            str: 格式化的时间字符串，如 "1:23.456"
        """
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        seconds_int = int(remaining_seconds)
        milliseconds = int((remaining_seconds - seconds_int) * 1000)
        return f"{minutes}:{seconds_int:02d}.{milliseconds:03d}"
        
    def _parse_time_format(self, time_str):
        """将时间字符串转换为秒数
        
        Args:
            time_str: 时间字符串，如 "1:23.456" 或 "1:23"
            
        Returns:
            float: 对应的秒数
        """
        # 支持带毫秒和不带毫秒的格式
        if "." in time_str:
            # 格式: mm:ss.sss
            time_part, ms_part = time_str.split(".")
            parts = time_part.split(":")
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
                milliseconds = int(ms_part[:3].ljust(3, '0'))  # 确保是3位毫秒
                return minutes * 60 + seconds + milliseconds / 1000.0
        else:
            # 格式: mm:ss (向后兼容)
            parts = time_str.split(":")
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes * 60 + seconds
        return 0.0
    
    def generate_multi_voice_speech(self, structured_text, narrator_audio, model_version="Index-TTS", 
                                   language="auto", speed=1.0, seed=0, 
                                   character1_audio=None, character2_audio=None, character3_audio=None, 
                                   character4_audio=None, character5_audio=None,
                                   temperature=0.8, top_p=0.9, top_k=30, 
                                   repetition_penalty=10.0, length_penalty=0.0, 
                                   num_beams=3, max_mel_tokens=1500,
                                   do_sample=False, mode="Auto", emotion_weight=0.8,
                                   auto_emotion=False, pause_between_lines=0.2):
        """
        生成多角色语音的主函数 / Main function for generating multi-character speech
        
        参数 / Parameters:
            structured_text: 结构化文本，包含角色标签 / Structured text with character tags
            narrator_audio: 旁白/正文的参考音频 / Narrator reference audio
            model_version: 模型版本 / Model version
            language: 语言设置 / Language setting
            speed: 语音速度 / Speech speed
            seed: 随机种子 / Random seed
            character1_audio~character5_audio: 角色参考音频 / Character reference audios
            temperature: 温度参数 / Temperature parameter
            top_p: top_p参数 / top_p parameter
            top_k: top_k参数 / top_k parameter
            repetition_penalty: 重复惩罚 / Repetition penalty
            length_penalty: 长度惩罚 / Length penalty
            num_beams: beam数量 / Number of beams
            max_mel_tokens: 最大mel token数 / Max mel tokens
            do_sample: 是否使用采样 / Whether to use sampling (V2 only)
            mode: 生成模式 / Generation mode (V2 only)
            emotion_weight: 情感强度控制 / Emotion intensity control (0.0-1.6, V2 only)
            auto_emotion: 自动情感分析 / Automatic emotion analysis from dialogue text (V2 only)
            pause_between_lines: 行间停顿时长(秒) / Pause duration between lines (seconds)
        """
        try:
            print(f"[IndexTTS Pro] Starting multi-voice generation with structured_text: {structured_text[:100]}...")
            print(f"[IndexTTS Pro] 使用模型版本 / Using model version: {model_version}")
            
            # 使用固定种子或随机种子 / Use fixed seed or random seed
            if seed == 0:
                seed = int(time.time() * 1000) % (2**32 - 1)
            
            # 初始化模型 / Initialize model
            self._init_model(model_version)
            
            # 解析结构化文本 / Parse structured text
            parsed_text = self._parse_structured_text(structured_text)
            print(f"[IndexTTS Pro] Parsed text segments: {len(parsed_text)}")
            
            # 构建角色音频映射 / Build character audio mapping
            character_audios = {}
            for i, char_audio in enumerate([character1_audio, character2_audio, character3_audio, 
                                           character4_audio, character5_audio], 1):
                if char_audio is not None:
                    character_audios[f"Character{i}"] = char_audio
            
            # 生成音频片段 / Generate audio segments
            audio_segments = []
            current_time = 0.0  # 当前时间位置 / Current time position
            subtitle_data = []  # Subtitle数据列表 / Subtitle data list
            
            # 判断是否使用V2模型 / Check if using V2 model
            is_v2 = (model_version == "IndexTTS-2")
            
            for segment_idx, (role, text, emotion) in enumerate(parsed_text):
                emotion_text = f" (emotion: {emotion})" if emotion else ""
                print(f"\n[IndexTTS Pro] 🎭 Processing: {role}{emotion_text}")
                print(f"[IndexTTS Pro] 📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
                
                # 选择参考音频 / Select reference audio
                if role == "Narrator":
                    ref_audio = narrator_audio
                elif role in character_audios:
                    ref_audio = character_audios[role]
                else:
                    # 使用旁白音频作为默认参考 / Use narrator audio as default
                    ref_audio = narrator_audio
                    print(f"[IndexTTS Pro] Warning: No specific audio for {role}, using narrator audio")
                
                try:
                    if is_v2:
                        # 使用V2 API / Use V2 API
                        # 注意: V2不支持speed参数 / Note: V2 does not support speed parameter
                        
                        # 确定是否使用情感分析 / Determine whether to use emotion analysis
                        use_emotion_analysis = False
                        emotion_text_input = None
                        
                        if emotion is not None:
                            # 检查是否显式抑制情感 / Check if emotion is explicitly suppressed
                            if emotion == "":
                                # 显式抑制情感：emo="" / Explicitly suppress emotion: emo=""
                                use_emotion_analysis = False
                                emotion_text_input = None
                                print(f"[IndexTTS Pro] Emotion explicitly suppressed for {role}")
                            else:
                                # 显式指定了情感文本 / Explicit emotion text specified
                                emotion_text_input = emotion
                                use_emotion_analysis = True
                                print(f"[IndexTTS Pro] Explicit Emotion Text: '{emotion}' (weight: {emotion_weight})")
                        elif auto_emotion:
                            # 自动情感分析模式：使用对话文本本身 / Automatic emotion mode: use dialogue text itself
                            emotion_text_input = None  # 让引擎自动使用text参数 / Let engine auto-use text parameter
                            use_emotion_analysis = True
                            print(f"[IndexTTS Pro] Automatic Emotion Analysis enabled for: '{text[:50]}...'" if len(text) > 50 else f"[IndexTTS Pro] Automatic Emotion Analysis enabled for: '{text}'")
                        else:
                            print("[IndexTTS Pro] No emotion processing")
                        
                        ref_processed = self._process_audio_input(ref_audio)
                        sr, wave, sub = self.tts2_engine.generate(
                            text=text,
                            reference_audio=ref_processed,
                            mode=mode,
                            do_sample=do_sample,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            num_beams=num_beams,
                            repetition_penalty=repetition_penalty,
                            length_penalty=length_penalty,
                            max_mel_tokens=max_mel_tokens,
                            emo_text=emotion_text_input,  # None会让引擎自动使用text / None lets engine auto-use text
                            emo_ref_audio=None,
                            emo_vector=None,
                            emo_weight=emotion_weight,  # 使用用户设定的情感强度 / Use user-defined emotion intensity
                            use_qwen=use_emotion_analysis,  # 根据情况启用Qwen分析 / Enable Qwen analysis based on conditions
                            verbose=True,  # 启用详细日志显示情感向量 / Enable verbose logging to show emotion vectors
                            seed=seed,
                            return_subtitles=True,
                        )
                        sample_rate = sr
                        audio_data = wave
                        print(f"[IndexTTS Pro] ✅ Generated audio for {role} with emotion: '{emotion}'")
                    else:
                        # 使用V1/V1.5 API / Use V1/V1.5 API
                        # 注意: V1/V1.5不支持情感控制，emotion参数被忽略 / Note: V1/V1.5 don't support emotion control, emotion parameter is ignored
                        if emotion:
                            print(f"[IndexTTS Pro] Warning: Emotion '{emotion}' specified but V1/V1.5 models don't support emotion control")
                        result = self.tts_model.infer(
                            reference_audio=self._process_audio_input(ref_audio),
                            text=text,
                            output_path=None,
                            language=language,
                            speed=speed,
                            verbose=False,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            repetition_penalty=repetition_penalty,
                            length_penalty=length_penalty,
                            num_beams=num_beams,
                            max_mel_tokens=max_mel_tokens
                        )
                        
                        # 处理返回结果 / Process return result
                        if isinstance(result, tuple) and len(result) == 2:
                            sample_rate, audio_data = result
                        else:
                            print(f"[IndexTTS Pro] Warning: Unexpected return format from V1/V1.5 model")
                            continue
                    
                    # 计算音频长度 / Calculate audio length
                    if isinstance(audio_data, np.ndarray) and audio_data.size > 0:
                        if audio_data.ndim == 1:
                            audio_length = len(audio_data) / sample_rate
                        else:
                            audio_length = audio_data.shape[-1] / sample_rate
                        
                        # 添加字幕数据 / Add subtitle data
                        start_time = self._seconds_to_time_format(current_time)
                        end_time = self._seconds_to_time_format(current_time + audio_length)
                        subtitle_item = {
                            "id": role,
                            "字幕": text,
                            "start": start_time,
                            "end": end_time
                        }
                        # 如果有情感信息，添加到字幕数据中 / If emotion info exists, add to subtitle data
                        if emotion:
                            subtitle_item["emotion"] = emotion
                        subtitle_data.append(subtitle_item)
                        current_time += audio_length
                        
                        # 添加到段落列表 / Add to segment list
                        audio_segments.append((audio_data, sample_rate))
                        
                        # 添加行间停顿 (除了最后一段) / Add pause between lines (except for last segment)
                        if pause_between_lines > 0 and segment_idx < len(parsed_text) - 1:
                            silence_samples = int(pause_between_lines * sample_rate)
                            silence = np.zeros(silence_samples, dtype=np.float32)
                            audio_segments.append((silence, sample_rate))
                            current_time += pause_between_lines
                            print(f"[IndexTTS Pro] Added {pause_between_lines}s pause after {role}")
                    else:
                        print(f"[IndexTTS Pro] Warning: Invalid audio data for {role}")
                    
                except Exception as e:
                    print(f"[IndexTTS Pro] Error generating {role} voice: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 连接所有音频片段 / Concatenate all audio segments
            final_audio = self._concatenate_audio(audio_segments)
            if final_audio is None:
                raise ValueError("Failed to generate any audio segments / 未能生成任何音频片段")
            
            # 计算音频长度（考虑可能是2D格式） / Calculate audio length (considering possible 2D format)
            if final_audio[0].ndim > 1:
                audio_length = final_audio[0].shape[1] / final_audio[1]
            else:
                audio_length = len(final_audio[0]) / final_audio[1]
                
            print(f"[IndexTTS Pro] Multi-voice generation complete, total length: {audio_length:.2f} seconds")
            print(f"[IndexTTS Pro] Final audio shape before processing: {final_audio[0].shape}, sample rate: {final_audio[1]}")
            
            # 转为ComfyUI格式 - 需要是3D格式: [batch, channels, samples] / Convert to ComfyUI format - needs to be 3D: [batch, channels, samples]
            audio_numpy = final_audio[0]
            
            # 转换为PyTorch张量 / Convert to PyTorch tensor
            audio_tensor = torch.tensor(audio_numpy, dtype=torch.float32)
            print(f"[IndexTTS Pro] Audio tensor dimensions: {audio_tensor.dim()}")
            
            # 确保音频数据是3D张量 [batch, channels, samples] / Ensure audio data is 3D tensor [batch, channels, samples]
            if audio_tensor.dim() == 1:
                # [samples] -> [1, 1, samples]
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
                print(f"[IndexTTS Pro] 1D tensor reshaped to 3D: [1, 1, {audio_tensor.shape[-1]}]")
            elif audio_tensor.dim() == 2:
                # [channels, samples] -> [1, channels, samples]
                audio_tensor = audio_tensor.unsqueeze(0)
                print(f"[IndexTTS Pro] 2D tensor reshaped to 3D: [1, {audio_tensor.shape[1]}, {audio_tensor.shape[2]}]")
            
            print(f"[IndexTTS Pro] Final tensor shape: {audio_tensor.shape}")
            
            # 生成SubtitleJSON字符串 / Generate SubtitleJSON string
            import json
            subtitle_json = json.dumps(subtitle_data, ensure_ascii=False, indent=2)
            print(f"[IndexTTS Pro] Generated subtitle data with {len(subtitle_data)} items")
            
            # 生成简化字幕格式 (只包含时间和处理后实际分句的文本，不包含角色名) / Generate simplified subtitle format (only time and processed text, without role names)
            simplified_subtitles = []
            
            # 在这里，由于我们没有直接获取到TTS处理后的分句，需要从模型日志中获取或使用一个模拟处理
            # 这部分需要根据tts_models.py中的实际处理逻辑调整
            # Here, since we don't directly get sentence splits from TTS processing, we need to get it from model logs or use simulated processing
            # This part needs adjustment based on actual processing logic in tts_models.py
            
            # 我们使用带有冒号的时间格式 / We use time format with colons
            process_timepoints = []
            current_pos = 0.0
            
            # 为每个角色的每句话创建时间点 / Create timepoints for each character's lines
            for item in subtitle_data:
                # 使用原始的带冒号时间格式 / Use original time format with colons
                start_time = item["start"]
                end_time = item["end"]
                text = item["字幕"]
                
                # 模拟分句处理 - 实际应该从模型中获取 / Simulate sentence splitting - should actually be obtained from model
                # 这里简单地按标点符号拆分 / Simply split by punctuation here
                import re
                # 将文本拆分为句子 (中文标点和英文标点) / Split text into sentences (Chinese and English punctuation)
                sentences = re.split(r'([,，.。!！?？;；])', text)
                # 过滤空字符串并重组句子和标点 / Filter empty strings and recombine sentences with punctuation
                sentences = [s + next_s for s, next_s in zip(sentences[::2], sentences[1::2] + [""])] if len(sentences) > 1 else [text]
                sentences = [s for s in sentences if s.strip()]
                
                if not sentences:  # 如果没有成功分句，就使用原始文本 / If no successful split, use original text
                    sentences = [text]
                
                # 计算每个子句的时长 / Calculate duration for each sub-sentence
                total_duration = self._parse_time_format(end_time) - self._parse_time_format(start_time)
                sentence_duration = total_duration / len(sentences) if sentences else total_duration
                
                # 为每个子句生成时间点 / Generate timepoints for each sub-sentence
                for i, sentence in enumerate(sentences):
                    if not sentence.strip():  # 跳过空句 / Skip empty sentences
                        continue
                    
                    sub_start = self._parse_time_format(start_time) + i * sentence_duration
                    sub_end = sub_start + sentence_duration
                    
                    sub_start_formatted = self._seconds_to_time_format(sub_start)
                    sub_end_formatted = self._seconds_to_time_format(sub_end)
                    
                    time_line = f">> {sub_start_formatted}-{sub_end_formatted}"
                    text_line = f">> {sentence}"
                    
                    simplified_subtitles.append(time_line)
                    simplified_subtitles.append(text_line)
            
            # 连接为字符串 / Join as string
            simplified_subtitle_str = "\n".join(simplified_subtitles)
            print(f"[IndexTTS Pro] Generated simplified subtitle format with processed sentences")
            
            # 最终返回ComfyUI格式的音频数据、种子、JSON字幕和简化字幕 / Finally return ComfyUI format audio data, seed, JSON subtitles and simplified subtitles
            return ({"waveform": audio_tensor, "sample_rate": final_audio[1]}, seed, subtitle_json, simplified_subtitle_str)
            
        except Exception as e:
            import traceback
            print(f"[IndexTTS Pro] Generation failed / 生成失败: {e}")
            print(traceback.format_exc())
            raise RuntimeError(f"Multi-voice generation failed / 多角色语音生成失败: {e}")
