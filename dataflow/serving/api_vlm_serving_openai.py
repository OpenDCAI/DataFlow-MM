import os
import base64
import json
import re
import requests
from PIL import Image
import io
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataflow.core import LLMServingABC
from openai import OpenAI
from io import BytesIO
from tqdm import tqdm

from ..logger import get_logger


class APIVLMServing_openai(LLMServingABC):
    """
    Client for interacting with a Vision-Language Model (VLM) via OpenAI's API.

    Provides methods for single-image chat, batch image processing, and multi-image analysis,
    with support for concurrent requests.
    """

    def __init__(
        self,
        api_url: str = "https://api.openai.com/v1",
        key_name_of_api_key: str = "DF_API_KEY",
        model_name: str = "o4-mini",
        image_io = None,
        send_request_stream = False,
        max_workers: int = 10,
        timeout: int = 1800
    ):
        """
        Initialize the OpenAI client and settings.

        :param api_url: Base URL of the VLM API endpoint.
        :param key_name_of_api_key: Environment variable name for the API key.
        :param model_name: Default model name to use for requests.
        :param max_workers: Maximum number of threads for concurrent requests.
        """
        self.api_url = api_url
        self.model_name = model_name
        self.max_workers = max_workers
        self.logger = get_logger()
        self.timeout = timeout
        api_key = os.environ.get(key_name_of_api_key)
        if not api_key:
            self.logger.error(f"API key not found in environment variable '{key_name_of_api_key}'")
            raise EnvironmentError(f"Missing environment variable '{key_name_of_api_key}'")

        self.client = OpenAI(
            api_key=api_key,
            base_url=api_url
        )
        self.image_io = image_io      # utilize the image IO to judge whether solve the generated images
        self.send_request_stream = send_request_stream

    def _encode_image_to_base64(self, image_path: str) -> Tuple[str, str]:
        """
        Read an image file and convert it to a base64-encoded string, returning the image data and MIME format.

        :param image_path: Path to the image file.
        :return: Tuple of (base64-encoded string, image format, e.g. 'jpeg' or 'png').
        :raises ValueError: If the image format is unsupported.
        """
        with open(image_path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("utf-8")
        ext = image_path.rsplit('.', 1)[-1].lower()

        if ext == 'jpg':
            fmt = 'jpeg'
        elif ext == 'jpeg':
            fmt = 'jpeg'
        elif ext == 'png':
            fmt = 'png'
        else:
            raise ValueError(f"Unsupported image format: {ext}")

        return b64, fmt

    def combine_images_to_base64(self, image_paths: List[str], mode: str = "horizontal") -> str:
        """
        Combine multiple images into one and return the combined image as a Base64 string.

        :param image_paths: List of image file paths.
        :param mode: Combination mode ('horizontal', 'vertical', or 'grid').
        :return: Base64 string of the combined image.
        """
        # Load images
        images = [Image.open(path) for path in image_paths]

        # Calculate combined image size
        if mode == "horizontal":
            width = sum(img.width for img in images)
            height = max(img.height for img in images)
            combined_image = Image.new("RGB", (width, height))
            offset = 0
            for img in images:
                combined_image.paste(img, (offset, 0))
                offset += img.width

        elif mode == "vertical":
            width = max(img.width for img in images)
            height = sum(img.height for img in images)
            combined_image = Image.new("RGB", (width, height))
            offset = 0
            for img in images:
                combined_image.paste(img, (0, offset))
                offset += img.height

        elif mode == "grid":
            import math
    
            # 设置参数
            canvas_size = 1024  # 1:1 画布大小
            padding = 40        # 留白大小
            
            # 创建白色画布
            combined_image = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
            
            n = len(images)
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)
            
            # 计算每个单元格的尺寸（考虑留白）
            cell_width = (canvas_size - padding * (cols + 1)) // cols
            cell_height = (canvas_size - padding * (rows + 1)) // rows
            
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if idx >= n:
                        break
                    
                    # 计算单元格左上角坐标
                    x = padding + c * (cell_width + padding)
                    y = padding + r * (cell_height + padding)
                    
                    # 缩放图片以适应单元格（保持比例）
                    img = images[idx]
                    w, h = img.size
                    
                    # 计算缩放比例
                    scale = min(cell_width / w, cell_height / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    
                    # 缩放图片
                    resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    
                    # 计算居中位置
                    offset_x = x + (cell_width - new_w) // 2
                    offset_y = y + (cell_height - new_h) // 2
                    
                    # 粘贴到画布
                    if resized_img.mode == 'RGBA':
                        combined_image.paste(resized_img, (offset_x, offset_y), resized_img)
                    else:
                        combined_image.paste(resized_img, (offset_x, offset_y))
                    
                    idx += 1

        else:
            raise ValueError("Mode must be 'horizontal', 'vertical', or 'combine'.")

        # Convert to Base64
        buffer = BytesIO()
        # resize 
        original_width, original_height = combined_image.size
        combined_image = combined_image.resize(
            (original_width // 2, original_height // 2), 
            Image.Resampling.LANCZOS
        )
        combined_image.save(buffer, format="PNG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return base64_image


    def _create_messages(self, content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Wrap content items into the standard OpenAI messages structure.

        :param content: List of content dicts (text/image elements).
        :return: Messages payload for the API call.
        """
        return [{"role": "user", "content": content}]

    def _send_chat_request(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        timeout: int
    ) -> str:
        """
        Send a chat completion request to the OpenAI API and return the generated content.

        :param model: Model name for the request.
        :param messages: Messages payload constructed by `_create_messages`.
        :param timeout: Timeout in seconds for the API call.
        :return: Generated text response from the model.
        """
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=timeout,
            stream=self.send_request_stream
        )
        if self.model_name == "gemini-2.5-flash-image-preview":
            if self.send_request_stream:
                full_content = ""
                for chunk in resp:
                    if chunk.choices[0].delta.content is not None and chunk.choices[0].delta.content!="":
                        # content_piece = chunk.choices[0].delta.content
                        # full_content += content_piece
                        full_content = chunk.choices[0].delta.content  # utilize the final response
            else:
                full_content = resp.choices[0].message.content
            return full_content
        return resp.choices[0].message.content

    def chat_with_one_image(
        self,
        image_path: str,
        text_prompt: str,
        model: str = None,
        timeout: int = 1800
    ) -> str:
        """
        Perform a chat completion using a single image and a text prompt.

        :param image_path: Path to the image file.
        :param text_prompt: Text prompt to accompany the image.
        :param model: (Optional) Model override; defaults to instance `model_name`.
        :param timeout: Timeout in seconds for the API call.
        :return: Model's response as a string.
        """
        model = model or self.model_name
        if isinstance(image_path, list):
            if len(image_path) > 1:
                b64 = self.combine_images_to_base64(image_paths=image_path, mode="grid")
                fmt = "png"
            elif len(image_path) == 1:
                b64, fmt = self._encode_image_to_base64(image_path[0])
        else: b64, fmt = self._encode_image_to_base64(image_path)
        content = [
            {"type": "text", "text": text_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/{fmt};base64,{b64}"}}
        ]
        messages = self._create_messages(content)
        return self._send_chat_request(model, messages, timeout)

    def chat_with_one_image_with_id(
        self,
        request_id: Any,
        image_path: str,
        text_prompt: str,
        model: str = None,
        timeout: int = 1800
    ) -> Tuple[Any, str]:
        """
        Same as `chat_with_one_image` but returns a tuple of (request_id, response).

        :param request_id: Arbitrary identifier for tracking the request.
        :param image_path: Path to the image file.
        :param text_prompt: Text prompt to accompany the image.
        :param model: (Optional) Model override; defaults to instance `model_name`.
        :param timeout: Timeout in seconds for the API call.
        :return: Tuple of (request_id, model response).
        """
        response = self.chat_with_one_image(image_path, text_prompt, model, timeout)
        if self.image_io != None:
            io_results = self._extract_and_save_image(response, f"sample_{request_id}")
            try:
                response = io_results[f"sample_{request_id}"]
            except Exception as e:
                print("Cannot save the image files: {str(e)}")
        return request_id, response

    def generate_from_input_one_image(
        self,
        image_paths: List[str],
        text_prompts: List[str],
        system_prompt: str = "",
        model: str = None,
        timeout: int = 1800
    ) -> List[str]:
        """
        Batch process single-image chat requests concurrently.

        :param image_paths: List of image file paths.
        :param text_prompts: List of text prompts (must match length of image_paths).
        :param system_prompt: Optional system-level prompt prefixed to each user prompt.
        :param model: (Optional) Model override; defaults to instance `model_name`.
        :param timeout: Timeout in seconds for each API call.
        :return: List of model responses preserving input order.
        :raises ValueError: If lengths of image_paths and text_prompts differ.
        """
        if len(image_paths) != len(text_prompts):
            raise ValueError("`image_paths` and `text_prompts` must have the same length")

        model = model or self.model_name
        prompts = [f"{system_prompt}\n{p}" for p in text_prompts]
        responses = [None] * len(image_paths)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.chat_with_one_image_with_id,
                    idx,
                    path,
                    prompt,
                    model,
                    timeout
                ): idx
                for idx, (path, prompt) in enumerate(zip(image_paths, prompts))
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating..."):
                idx, res = future.result()
                responses[idx] = res

        return responses

    def _is_base64(self, s):
        """check whether string is validate base64"""
        try:
            if isinstance(s, str):
                s = s.strip()
                if re.match(r'^[A-Za-z0-9+/]*={0,2}$', s):
                    base64.b64decode(s)
                    return True
            return False
        except:
            return False
    
    def _extract_and_save_image(self, content, prompt):
        """save generated image from API response"""
        try:
            # extract the base64 code
            images = []
            image_data = {}
            base64_pattern = r'!\[.*?\]\(data:image/(png|jpg|jpeg|gif|bmp);base64,([A-Za-z0-9+/=]+)\)'
            matches = re.findall(base64_pattern, content)
            for i, (image_type, base64_str) in enumerate(matches):
                try:
                    image_bytes = base64.b64decode(base64_str)
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    images.append(pil_image)
                    # return self.image_io.write(image_data)
                except:
                    continue
            
            # save the url images
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\])]+'
            urls = re.findall(url_pattern, content)
            for url in urls:
                url = url.replace('\/', '/')
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        pil_image = Image.open(io.BytesIO(response.content))
                        images.append(pil_image)
                except:
                    continue
            image_data[prompt] = images

            if image_data:
                return self.image_io.write(image_data)
            else:
                return {}

        except Exception as e:
            print(f"保存图片时出错: {str(e)}")
            return {}

    def analyze_images_with_gpt(
        self,
        image_paths: List[str],
        image_labels: List[str],
        system_prompt: str = "",
        model: str = None,
        timeout: int = 1800
    ) -> str:
        """
        Analyze multiple images in a single request with labels.

        :param image_paths: List of image file paths.
        :param image_labels: Corresponding labels for each image.
        :param system_prompt: Overall prompt before listing images.
        :param model: (Optional) Model override; defaults to instance `model_name`.
        :param timeout: Timeout in seconds for the API call.
        :return: Model's combined analysis as text.
        """
        if len(image_paths) != len(image_labels):
            raise ValueError("`image_paths` and `image_labels` must have the same length")

        model = model or self.model_name
        content: List[Dict[str, Any]] = []
        if system_prompt:
            content.append({"type": "text", "text": system_prompt})

        for label, path in zip(image_labels, image_paths):
            b64, fmt = self._encode_image_to_base64(path)
            content.append({"type": "text", "text": f"{label}:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{fmt};base64,{b64}"}
            })

        messages = self._create_messages(content)
        return self._send_chat_request(model, messages, timeout)

    def analyze_images_with_gpt_with_id(
        self,
        image_paths: List[str],
        image_labels: List[str],
        request_id: Any,
        system_prompt: str = "",
        model: str = None,
        timeout: int = 1800
    ) -> Tuple[Any, str]:
        """
        Batch-tracked version of `analyze_images_with_gpt`, returning (request_id, analysis).

        :param image_paths: List of image file paths.
        :param image_labels: Corresponding labels for each image.
        :param request_id: Identifier for tracking the request.
        :param system_prompt: Overall prompt before listing images.
        :param model: (Optional) Model override; defaults to instance `model_name`.
        :param timeout: Timeout in seconds for the API call.
        :return: Tuple of (request_id, model's analysis).
        """
        result = self.analyze_images_with_gpt(
            image_paths,
            image_labels,
            system_prompt,
            model,
            timeout
        )
        self.logger.info(f"Request {request_id} completed")
        return request_id, result

    def generate_from_input_multi_images(
        self,
        list_of_image_paths: List[List[str]],
        list_of_image_labels: List[List[str]],
        system_prompt: str = "",
        model: str = None,
        timeout: int = 1800
    ) -> List[str]:
        """
        Concurrently analyze multiple sets of images with labels.

        :param list_of_image_paths: List of image path lists.
        :param list_of_image_labels: Parallel list of label lists.
        :param system_prompt: Prompt prefixed to each batch.
        :param model: (Optional) Model override; defaults to instance `model_name`.
        :param timeout: Timeout in seconds for each API call.
        :return: List of analysis results in input order.
        :raises ValueError: If outer lists lengths differ.
        """
        if len(list_of_image_paths) != len(list_of_image_labels):
            raise ValueError(
                "`list_of_image_paths` and `list_of_image_labels` must have the same length"
            )

        model = model or self.model_name
        responses = [None] * len(list_of_image_paths)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.analyze_images_with_gpt_with_id,
                    paths,
                    labels,
                    idx,
                    system_prompt,
                    model,
                    timeout
                ): idx
                for idx, (paths, labels) in enumerate(
                    zip(list_of_image_paths, list_of_image_labels)
                )
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating..."):
                idx, res = future.result()
                responses[idx] = res

        return responses

    def cleanup(self) -> None:
        """
        Clean up any resources (e.g., close HTTP connections).
        """
        self.client.close()
    
    def _encode_image_to_base64_from_path(self, image_path: str) -> str:
        """
        将图像文件编码为 base64 字符串
        
        :param image_path: 图像文件路径
        :return: base64 编码的字符串
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _encode_video_to_base64_from_path(self, video_path: str) -> str:
        """
        将视频文件编码为 base64 字符串
        
        :param video_path: 视频文件路径
        :return: base64 编码的字符串
        """
        with open(video_path, "rb") as video_file:
            return base64.b64encode(video_file.read()).decode("utf-8")

    def _encode_audio_to_base64_from_path(self, audio_path: str) -> str:
        """
        将音频文件编码为 base64 字符串
        
        :param audio_path: 音频文件路径
        :return: base64 编码的字符串
        """
        with open(audio_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")

    def _guess_audio_format(self, audio_path: str) -> str:
        """
        给 Chat Completions 的 input_audio.format 用的格式字符串
        常见：wav / mp3 / m4a / ogg / webm / flac / aac
        """
        ext = audio_path.rsplit(".", 1)[-1].lower()

        # 与 docs/常见格式对齐，未知就默认 wav
        if ext in {"wav", "mp3", "m4a", "ogg", "webm", "flac", "aac"}:
            return ext

        # 一些边缘扩展名做个容错映射
        if ext in {"mpeg", "mpga"}:
            return "mp3"
        if ext == "mp4":
            return "mp4"

        return "wav"

    def _build_message_content(
        self, prompt: str, 
        image_paths: List[str] = None, 
        video_paths: List[str] = None,
        audio_paths: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        构建 OpenAI API 格式的消息内容
        
        :param prompt: 文本提示
        :param image_paths: 图像路径列表
        :param video_paths: 视频路径列表
        :param audio_paths: 音频路径列表
        :return: OpenAI API 格式的内容列表
        """
        content = []
        
        # 添加图像内容
        if image_paths:
            for image_path in image_paths:
                if image_path:
                    base64_image = self._encode_image_to_base64_from_path(image_path)
                    # 检测图像格式
                    ext = image_path.rsplit('.', 1)[-1].lower()
                    if ext == 'jpg':
                        fmt = 'jpeg'
                    elif ext == 'jpeg':
                        fmt = 'jpeg'
                    elif ext == 'png':
                        fmt = 'png'
                    elif ext == 'webp':
                        fmt = 'webp'
                    else:
                        fmt = 'jpeg'  # 默认使用 jpeg
                    
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{fmt};base64,{base64_image}"}
                    })
        
        # 添加视频内容
        if video_paths:
            for video_path in video_paths:
                if video_path:
                    base64_video = self._encode_video_to_base64_from_path(video_path)
                    content.append({
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{base64_video}"}
                    })

        # 添加音频内容
        if audio_paths:
            for audio_path in audio_paths:
                if audio_path:
                    base64_audio = self._encode_audio_to_base64_from_path(audio_path)
                    fmt = self._guess_audio_format(audio_path)
                    content.append({
                        "type": "audio_url",
                        "audio_url": {
                            "url": f"data:audio/{fmt};base64,{base64_audio}",
                        }
                    })
        
        # 添加文本内容
        content.append({"type": "text", "text": prompt})
        
        return content

    def _send_single_request_with_id(
        self,
        idx: int,
        prompt: str,
        image_paths: List[str] = None,
        video_paths: List[str] = None,
        audio_paths: List[str] = None,
        system_prompt: str = "You are a helpful assistant."
    ) -> Tuple[int, str]:
        """
        发送单个请求并返回结果
        
        :param idx: 请求索引
        :param prompt: 文本提示
        :param image_paths: 图像路径列表
        :param video_paths: 视频路径列表
        :param audio_paths: 音频路径列表
        :param system_prompt: 系统提示
        :return: (索引, 响应文本)
        """
        content = self._build_message_content(prompt, image_paths, video_paths, audio_paths)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ]
        
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                timeout=self.timeout,
                stream=self.send_request_stream
            )
            
            # if self.model_name == "gemini-2.5-flash-image-preview":
            #     if self.send_request_stream:
            #         full_content = ""
            #         for chunk in resp:
            #             if chunk.choices[0].delta.content is not None and chunk.choices[0].delta.content != "":
            #                 full_content = chunk.choices[0].delta.content
            #     else:
            #         full_content = resp.choices[0].message.content
            #     return idx, full_content
            
            return idx, resp.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Request {idx} failed: {str(e)}")
            return idx, f"Error: {str(e)}"

    def generate_from_input(
        self,
        user_inputs: List[str],
        system_prompt: str = "You are a helpful assistant.",
        image_inputs: List[List[str]] = None,
        video_inputs: List[List[str]] = None,
        audio_inputs: List[List[str]] = None
    ) -> List[str]:
        """
        批量生成文本，与 local_model_vlm_serving 接口对齐
        
        :param user_inputs: 文本提示列表
        :param system_prompt: 系统提示
        :param image_inputs: 图像路径列表的列表，每个元素是一个样本的图像路径列表
        :param video_inputs: 视频路径列表的列表，每个元素是一个样本的视频路径列表
        :param audio_inputs: 音频路径列表的列表（当前未使用）
        :return: 生成的文本列表
        """
        self.logger.info(f"API VLM Serving: Processing {len(user_inputs)} requests")
        
        if image_inputs is not None:
            self.logger.info(f"Image inputs: {len(image_inputs)}")
        if video_inputs is not None:
            self.logger.info(f"Video inputs: {len(video_inputs)}")
        if audio_inputs is not None:
            self.logger.info(f"Audio inputs: {len(audio_inputs)}")
        
        # 检查是否为纯文本模式
        is_pure_text = (image_inputs is None or all(img is None for img in image_inputs)) and \
                       (video_inputs is None or all(vid is None for vid in video_inputs)) and \
                       (audio_inputs is None or all(aud is None for aud in audio_inputs))
        
        if is_pure_text:
            self.logger.info("Pure text mode detected")
        
        results = [None] * len(user_inputs)
        futures = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for idx, prompt in enumerate(user_inputs):
                # 获取当前样本的图像和视频路径
                current_image_paths = None
                if image_inputs is not None and idx < len(image_inputs) and image_inputs[idx] is not None:
                    current_image_paths = image_inputs[idx] if isinstance(image_inputs[idx], list) else [image_inputs[idx]]
                
                current_video_paths = None
                if video_inputs is not None and idx < len(video_inputs) and video_inputs[idx] is not None:
                    current_video_paths = video_inputs[idx] if isinstance(video_inputs[idx], list) else [video_inputs[idx]]
                
                current_audio_paths = None
                if audio_inputs is not None and idx < len(audio_inputs) and audio_inputs[idx] is not None:
                    current_audio_paths = audio_inputs[idx] if isinstance(audio_inputs[idx], list) else [audio_inputs[idx]]


                future = executor.submit(
                    self._send_single_request_with_id,
                    idx,
                    prompt,
                    current_image_paths,
                    current_video_paths,
                    current_audio_paths,
                    system_prompt
                )
                futures.append(future)
            
            # 收集结果
            for future in tqdm(as_completed(futures), total=len(futures), desc="API Generating..."):
                idx, response = future.result()
                results[idx] = response
        
        return results
    
    def _build_messages_from_conversation(
        self,
        conversation: List[Dict[str, str]],
        image_paths: List[str] = None,
        video_paths: List[str] = None,
        audio_paths: List[str] = None,
        system_prompt: str = "You are a helpful assistant."
    ) -> List[Dict[str, Any]]:
        """
        从对话历史构建 OpenAI API 格式的消息列表
        
        :param conversation: 单个对话历史，格式为 [{"role": "user", "content": "..."}, ...]
        :param image_paths: 图像路径列表
        :param video_paths: 视频路径列表
        :param audio_paths: 音频路径列表
        :param system_prompt: 系统提示
        :return: OpenAI API 格式的消息列表
        """
        messages = []
        
        # 添加系统提示
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 处理对话历史
        for turn in conversation:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            
            # 只对最后一轮 user 消息添加多模态内容
            if role == "user" and turn == conversation[-1]:
                # 构建包含多模态内容的消息
                message_content = []
                
                # 添加图像
                if image_paths:
                    for image_path in image_paths:
                        if image_path:
                            base64_image = self._encode_image_to_base64_from_path(image_path)
                            ext = image_path.rsplit('.', 1)[-1].lower()
                            if ext == 'jpg':
                                fmt = 'jpeg'
                            elif ext == 'jpeg':
                                fmt = 'jpeg'
                            elif ext == 'png':
                                fmt = 'png'
                            elif ext == 'webp':
                                fmt = 'webp'
                            else:
                                fmt = 'jpeg'
                            
                            message_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/{fmt};base64,{base64_image}"}
                            })
                
                # 添加视频
                if video_paths:
                    for video_path in video_paths:
                        if video_path:
                            base64_video = self._encode_video_to_base64_from_path(video_path)
                            message_content.append({
                                "type": "video_url",
                                "video_url": {"url": f"data:video/mp4;base64,{base64_video}"}
                            })

                # 添加音频内容
                if audio_paths:
                    for audio_path in audio_paths:
                        if audio_path:
                            base64_audio = self._encode_audio_to_base64_from_path(audio_path)
                            fmt = self._guess_audio_format(audio_path)
                            content.append({
                                "type": "audio_url",
                                "audio_url": {
                                    "url": f"data:audio/{fmt};base64,{base64_audio}",
                                }
                            })
                
                # 添加文本内容
                message_content.append({"type": "text", "text": content})
                
                messages.append({"role": role, "content": message_content})
            else:
                # 其他消息直接添加文本
                messages.append({"role": role, "content": content})
        
        return messages
    
    def _send_conversation_request_with_id(
        self,
        idx: int,
        conversation: List[Dict[str, str]],
        image_paths: List[str] = None,
        video_paths: List[str] = None,
        audio_paths: List[str] = None,
        system_prompt: str = "You are a helpful assistant."
    ) -> Tuple[int, str]:
        """
        发送带有对话历史的单个请求
        
        :param idx: 请求索引
        :param conversation: 对话历史
        :param image_paths: 图像路径列表
        :param video_paths: 视频路径列表
        :param audio_paths: 音频路径列表
        :param system_prompt: 系统提示
        :return: (索引, 响应文本)
        """
        messages = self._build_messages_from_conversation(
            conversation,
            image_paths,
            video_paths,
            audio_paths,
            system_prompt
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                timeout=self.timeout,
                stream=self.send_request_stream
            )
            
            return idx, resp.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Request {idx} failed: {str(e)}")
            return idx, f"Error: {str(e)}"
    
    def generate_from_input_messages(
        self,
        conversations: List[List[Dict[str, str]]],
        image_list: List[List[str]] = None,
        video_list: List[List[str]] = None,
        audio_list: List[List[str]] = None,
        system_prompt: str = "You are a helpful assistant."
    ) -> List[str]:
        """
        批量生成文本，支持对话历史格式，与 local_model_vlm_serving 接口对齐
        
        :param conversations: 对话历史列表，每个元素是一个对话历史
                             格式：[[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...], ...]
        :param image_list: 图像路径列表的列表，每个元素对应一个样本的图像路径列表
        :param video_list: 视频路径列表的列表，每个元素对应一个样本的视频路径列表
        :param audio_list: 音频路径列表的列表（当前未实现）
        :param system_prompt: 系统提示
        :return: 生成的文本列表
        """
        self.logger.info(f"API VLM Serving: Processing {len(conversations)} conversation requests")
        
        if image_list is not None:
            self.logger.info(f"Image inputs: {len(image_list)}")
        if video_list is not None:
            self.logger.info(f"Video inputs: {len(video_list)}")
        if audio_list is not None:
            self.logger.warning("Audio inputs are not yet supported and will be ignored")
        
        results = [None] * len(conversations)
        futures = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for idx, conversation in enumerate(conversations):
                # 获取当前样本的图像和视频路径
                current_image_paths = None
                if image_list is not None and idx < len(image_list) and image_list[idx] is not None:
                    current_image_paths = image_list[idx] if isinstance(image_list[idx], list) else [image_list[idx]]
                
                current_video_paths = None
                if video_list is not None and idx < len(video_list) and video_list[idx] is not None:
                    current_video_paths = video_list[idx] if isinstance(video_list[idx], list) else [video_list[idx]]

                current_audio_paths = None
                if audio_list is not None and idx < len(audio_list) and audio_list[idx] is not None:
                    current_audio_paths = audio_list[idx] if isinstance(audio_list[idx], list) else [audio_list[idx]]
                
                future = executor.submit(
                    self._send_conversation_request_with_id,
                    idx,
                    conversation,
                    current_image_paths,
                    current_video_paths,
                    current_audio_paths,
                    system_prompt
                )
                futures.append(future)
            
            # 收集结果
            for future in tqdm(as_completed(futures), total=len(futures), desc="API Generating (Messages)..."):
                idx, response = future.result()
                results[idx] = response
        
        return results