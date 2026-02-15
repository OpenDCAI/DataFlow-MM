# dataflow/serving/api_image_gen_serving.py
import os
import base64
import requests
from typing import Any, List, Dict, Optional, Callable, Tuple
from PIL import Image
from io import BytesIO
from dataflow.core import VLMServingABC
from dataflow import get_logger

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class APIImageGenServing(VLMServingABC):    
    API_FORMAT_OPENAI = "openai"
    API_FORMAT_GEMINI = "gemini"
    
    def __init__(
        self,
        api_url: str,
        image_io,
        Image_gen_task: str = "text2image",
        batch_size: int = 4,
        timeout: int = 300,
        connect_timeout: int = 30,
        api_format: str = "openai",
        api_key: Optional[str] = None,
        model_name: str = "dall-e-3",
    ):
        """
        :param api_url: Base URL of the cloud API (e.g. "https://api.openai.com/v1")
        :param image_io: ImageIO instance, for saving generated images
        :param Image_gen_task: Task type, "text2image" or "imageedit"
        :param batch_size: Batch size
        :param timeout: Request timeout (seconds)
        :param api_format: API format type, "openai" or "gemini" (default is "openai")
        :param api_key: API key (directly from parameters, not from environment variables)
        :param model_name: Model name (OpenAI: "dall-e-3", Gemini: "gemini-2.5-flash-image", "gemini-3-pro-image", etc. (default is "dall-e-3"))
        """
        self.api_url = api_url.rstrip("/")
        self.image_io = image_io
        self.image_gen_task = Image_gen_task
        self.batch_size = batch_size
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.api_format = api_format
        self.model_name = model_name
        self.logger = get_logger()
        
        self.api_key = api_key
        
        if not self.api_key:
            self.logger.warning("API key not provided. Some APIs may require authentication.")
        
        if api_format == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError(
                    "google.genai library is required for Gemini API. "
                    "Please install it: pip install google-genai"
                )
            
            if not self.api_key:
                raise ValueError("Gemini API key is required! Please provide it via --api_key parameter.")
            
            if self.api_url:
                http_options = types.HttpOptions(
                    base_url=self.api_url,
                    timeout=None
                )
            else:
                http_options = types.HttpOptions(timeout=None)
            
            try:
                self.gemini_client = genai.Client(
                    api_key=self.api_key,
                    http_options=http_options
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize Gemini client: {e}")
                raise
        else:
            self.gemini_client = None
        
        if api_format not in ["openai", "gemini"]:
            raise ValueError(f"Unsupported api_format: {api_format}. Only 'openai' and 'gemini' are supported.")

    def _build_openai_request(self, user_input: Any) -> Dict[str, Any]:
        """
        Build a single OpenAI image generation request (supports DALL-E 2/3 and gpt-image-1)
        """
        if isinstance(user_input, dict):
            prompt = user_input.get("text_prompt", user_input.get("prompt", ""))
        else:
            prompt = str(user_input)
        
        request = {
            "prompt": prompt,
            "n": 1,  # number of images to generate (1-10)
        }
        
        # Set default size based on model
        # DALL-E 2: "256x256", "512x512", "1024x1024"
        # DALL-E 3: "1024x1024", "1024x1792", "1792x1024"
        # GPT Image: "1024x1024", "1024x1792", "1792x1024"
        if self.model_name and "dall-e-2" in self.model_name.lower():
            request["size"] = "1024x1024"  # default size for DALL-E 2
        else:
            request["size"] = "1024x1024"  # default size for DALL-E 3 and GPT Image
        
        if isinstance(user_input, dict):
            user_size = user_input.get("size")
            if user_size:
                request["size"] = user_size
        
        if self.model_name:
            request["model"] = self.model_name
        
        # GPT Image model supports quality parameter (low, standard, high)
        if self.model_name and "gpt-image" in self.model_name.lower():
            quality = user_input.get("quality", "standard") if isinstance(user_input, dict) else "standard"
            request["quality"] = quality
            
            # GPT Image supports transparent background
            if isinstance(user_input, dict) and user_input.get("background") == "transparent":
                request["background"] = "transparent"
                # transparent background is recommended to use PNG or WebP format
                request["response_format"] = "b64_json"
        
        return request

    def _parse_openai_response(self, response: Dict[str, Any], user_input: Any) -> Tuple[str, Optional[Image.Image]]:
        """Parse OpenAI image generation/editing response, return (key, image)"""
        if "data" in response and len(response["data"]) > 0:
            first_item = response["data"][0]
            
            if "url" in first_item:
                img_url = first_item["url"]
                if img_url:
                    img_resp = requests.get(img_url, timeout=30)
                    img = Image.open(BytesIO(img_resp.content)).convert("RGB")
                    
                    if isinstance(user_input, dict):
                        key = str(user_input.get("sample_id", user_input.get("idx", user_input.get("text_prompt", "default"))))
                    else:
                        key = str(user_input)
                    return key, img
            
            # Check if there is base64 encoded image (if response_format="b64_json")
            elif "b64_json" in first_item:
                import base64
                b64_data = first_item["b64_json"]
                img_data = base64.b64decode(b64_data)
                img = Image.open(BytesIO(img_data)).convert("RGB")
                
                # Determine key
                if isinstance(user_input, dict):
                    key = str(user_input.get("sample_id", user_input.get("idx", user_input.get("text_prompt", "default"))))
                else:
                    key = str(user_input)
                return key, img
        
        if isinstance(user_input, dict):
            key = str(user_input.get("sample_id", user_input.get("idx", user_input.get("text_prompt", "default"))))
        else:
            key = str(user_input)
        return key, None

    def _build_openai_edit_request(self, user_input: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Build OpenAI image editing request (supports DALL-E 2 and gpt-image-1)
        Returns:
            (files_dict, data_dict): files dictionary (for multipart/form-data) and data dictionary
        """
        if not isinstance(user_input, dict):
            raise ValueError("Image editing requires dict input with 'image_path' and 'prompt'")
        
        image_path = user_input.get("image_path") or user_input.get("image") or user_input.get("input_image")
        if not image_path:
            raise ValueError("Image editing requires 'image_path' in input")
        
        prompt = user_input.get("prompt") or user_input.get("text_prompt", "")
        if not prompt:
            raise ValueError("Image editing requires 'prompt' in input")
        
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
        except Exception as e:
            raise ValueError(f"Failed to read image file {image_path}: {e}")
        
        files = {
            "image": (os.path.basename(image_path), image_data, "image/png")
        }
        
        data = {
            "prompt": prompt,
            "n": 1,
        }
        
        # If there is mask (optional), add mask
        mask_path = user_input.get("mask") or user_input.get("mask_path")
        if mask_path:
            try:
                with open(mask_path, "rb") as f:
                    mask_data = f.read()
                files["mask"] = (os.path.basename(mask_path), mask_data, "image/png")
            except Exception as e:
                self.logger.warning(f"Failed to read mask file {mask_path}: {e}")
        
        if self.model_name and "dall-e-2" in self.model_name.lower():
            data["size"] = user_input.get("size", "1024x1024")
        else:
            data["size"] = user_input.get("size", "1024x1024")
        
        # For gpt-image-1, can add quality parameter
        if self.model_name and "gpt-image" in self.model_name.lower():
            quality = user_input.get("quality", "standard")
            data["quality"] = quality
        
        return files, data

    def _build_openai_variation_request(self, user_input: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Build OpenAI image variation request (only supports DALL-E 2)
        Returns:
            (files_dict, data_dict): files dictionary (for multipart/form-data) and data dictionary
        """
        if not isinstance(user_input, dict):
            raise ValueError("Image variation requires dict input with 'image_path'")
        
        image_path = user_input.get("image_path") or user_input.get("image") or user_input.get("input_image")
        if not image_path:
            raise ValueError("Image variation requires 'image_path' in input")
        
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
        except Exception as e:
            raise ValueError(f"Failed to read image file {image_path}: {e}")
        
        files = {
            "image": (os.path.basename(image_path), image_data, "image/png")
        }
        
        data = {
            "n": 1,
            "size": user_input.get("size", "1024x1024"),
        }
        
        return files, data

    def _call_openai_edit_api(self, files: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Call OpenAI image editing API (/images/edits)"""
        endpoint = f"{self.api_url}/v1/images/edits"
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        timeout_tuple = (self.connect_timeout, self.timeout)
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    endpoint,
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=timeout_tuple
                )
                
                if resp.status_code == 503 and attempt < max_retries - 1:
                    self.logger.warning(
                        f"503 Service Unavailable (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {retry_delay}s..."
                    )
                    import time
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                
                resp.raise_for_status()
                return resp.json()
                
            except requests.exceptions.HTTPError as e:
                if resp.status_code == 503 and attempt < max_retries - 1:
                    self.logger.warning(
                        f"503 Service Unavailable (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {retry_delay}s..."
                    )
                    import time
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    self.logger.error(
                        f"HTTP error {resp.status_code} for endpoint: {endpoint}\n"
                        f"Response: {resp.text[:500] if hasattr(resp, 'text') else 'N/A'}"
                    )
                    raise
            except Exception as e:
                self.logger.error(f"Error calling OpenAI edit API: {e}")
                raise

    def _call_openai_variation_api(self, files: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Call OpenAI image variation API (/images/variations)"""
        endpoint = f"{self.api_url}/v1/images/variations"
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        timeout_tuple = (self.connect_timeout, self.timeout)
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    endpoint,
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=timeout_tuple
                )
                
                if resp.status_code == 503 and attempt < max_retries - 1:
                    self.logger.warning(
                        f"503 Service Unavailable (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {retry_delay}s..."
                    )
                    import time
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                
                resp.raise_for_status()
                return resp.json()
                
            except requests.exceptions.HTTPError as e:
                if resp.status_code == 503 and attempt < max_retries - 1:
                    self.logger.warning(
                        f"503 Service Unavailable (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {retry_delay}s..."
                    )
                    import time
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    self.logger.error(
                        f"HTTP error {resp.status_code} for endpoint: {endpoint}\n"
                        f"Response: {resp.text[:500] if hasattr(resp, 'text') else 'N/A'}"
                    )
                    raise
            except Exception as e:
                self.logger.error(f"Error calling OpenAI variation API: {e}")
                raise

    def _build_gemini_prompt(self, user_input: Any) -> str:
        """Extract prompt text from user_input"""
        if isinstance(user_input, dict):
            prompt = user_input.get("text_prompt", user_input.get("prompt", ""))
        else:
            prompt = str(user_input)
        return prompt

    def _load_image_for_gemini(self, image_path: str) -> Image.Image:
        """Load image file, return PIL Image object (Gemini API can directly accept PIL Image)"""
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            raise

    def _load_images_for_gemini(self, image_paths: List[str]) -> List[Image.Image]:
        """Load multiple image files, return PIL Image object list"""
        images = []
        for image_path in image_paths:
            try:
                image = self._load_image_for_gemini(image_path)
                images.append(image)
            except Exception as e:
                self.logger.warning(f"Failed to load image {image_path}: {e}, skipping...")
        return images

    def _call_gemini_chat_api(
        self, 
        conversations: List[Dict[str, str]], 
        image_paths: Optional[List[str]] = None,
        aspect_ratio: Optional[str] = None,
        resolution: Optional[str] = None
    ) -> List[Image.Image]:
        """
        Use chat mode for multi-round image editing
        Returns:
            list of images generated in the last round
        """
        from google.genai import types
        
        config = types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE'],
        )
        
        chat = self.gemini_client.chats.create(
            model=self.model_name,
            config=config
        )
        
        all_images = []
        
        for turn_idx, turn in enumerate(conversations):
            message = turn.get("content", "")
            if not message:
                continue
            
            contents = []
            
            if turn_idx == 0 and image_paths:
                for image_path in image_paths:
                    if isinstance(image_path, str):
                        image = self._load_image_for_gemini(image_path)
                        contents.append(image)
                    elif isinstance(image_path, Image.Image):
                        contents.append(image_path)
            
            contents.append(message)
            
            if turn_idx > 0 and (aspect_ratio or resolution):
                image_config = types.ImageConfig()
                if aspect_ratio:
                    image_config.aspect_ratio = aspect_ratio
                if resolution:
                    image_config.image_size = resolution
                
                config = types.GenerateContentConfig(
                    image_config=image_config
                )
                response = chat.send_message(contents, config=config)
            else:
                response = chat.send_message(contents)
            
            turn_images = []
            for part in response.parts:
                if part.inline_data is not None:
                    inline_data = part.inline_data
                    if hasattr(inline_data, 'data'):
                        img_data = inline_data.data
                        img_bytes = BytesIO(img_data)
                        image = Image.open(img_bytes)
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                        turn_images.append(image)
            
            if turn_images:
                all_images.extend(turn_images)
        
        return all_images if all_images else None

    def _call_gemini_api(self, prompt: str, image_paths: Optional[List[str]] = None) -> List[Image.Image]:
        """
        Call Gemini API to generate images using google.genai library
        """
        contents = []
        
        if image_paths and self.image_gen_task == "imageedit":
            for image_path in image_paths:
                if isinstance(image_path, str):
                    image = self._load_image_for_gemini(image_path)
                    contents.append(image)
                elif isinstance(image_path, Image.Image):
                    contents.append(image_path)
        
        contents.append(prompt)
        
        response = self.gemini_client.models.generate_content(
            model=self.model_name,
            contents=contents,
        )
        
        images = []
        for part in response.parts:
            if part.inline_data is not None:
                inline_data = part.inline_data
                if hasattr(inline_data, 'data'):
                    img_data = inline_data.data
                    img_bytes = BytesIO(img_data)
                    image = Image.open(img_bytes)
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    images.append(image)
        
        if images:
            return images[0] if len(images) == 1 else images
        else:
            return None

    def _call_api(self, payload: Any, headers: Optional[Dict[str, str]] = None, endpoint_suffix: str = "generations") -> Dict[str, Any]:
        """
        Send API request
        """
        if headers is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            headers["Content-Type"] = "application/json"
        
        if self.api_format == "openai":
            endpoint = f"{self.api_url}/v1/images/{endpoint_suffix}"
        elif self.api_format == "gemini":
            raise ValueError("Gemini API should use _call_gemini_api() method, not _call_api()")
        else:
            raise ValueError(f"Unsupported api_format: {self.api_format}")
        
        timeout_tuple = (self.connect_timeout, self.timeout)
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    endpoint,
                    json=payload if isinstance(payload, (dict, list)) else payload,
                    headers=headers,
                    timeout=timeout_tuple
                )
                
                if resp.status_code == 503 and attempt < max_retries - 1:
                    self.logger.warning(
                        f"503 Service Unavailable (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {retry_delay}s..."
                    )
                    import time
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                
                resp.raise_for_status()
                return resp.json()
                
            except requests.exceptions.HTTPError as e:
                if resp.status_code == 503 and attempt < max_retries - 1:
                    self.logger.warning(
                        f"503 Service Unavailable (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {retry_delay}s..."
                    )
                    import time
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    self.logger.error(
                        f"HTTP error {resp.status_code} for endpoint: {endpoint}\n"
                        f"Response: {resp.text[:500] if hasattr(resp, 'text') else 'N/A'}"
                    )
                    raise

    def generate_from_input(self, user_inputs: List[Any]):
        """Batch generate images, interface aligned with LocalImageGenServing"""
        out_dict = {}
        for start in range(0, len(user_inputs), self.batch_size):
            batch_inputs = user_inputs[start:start + self.batch_size]
            if not batch_inputs:
                continue
            batch_result = self.generate_from_input_one_batch(batch_inputs)
            out_dict.update(batch_result)
        return out_dict

    def generate_from_input_one_batch(self, user_inputs: List[Any]) -> Dict[str, List[Image.Image]]:
        """
        Process a batch of inputs, return dictionary in {key: [Image, ...]} format.
        
        user_inputs format is aligned with LocalImageGenServing:
        - text2image: List[dict], each dict contains "text_prompt" and "sample_id"
        - imageedit: List[dict], each dict contains "image_path" and "prompt" etc.
        """
        images_dict = {}
        
        try:
            for user_input in user_inputs:
                try:
                    if self.api_format == "openai":
                        if self.image_gen_task == "imageedit":
                            # Image editing task
                            try:
                                use_variations = user_input.get("use_variations", False) if isinstance(user_input, dict) else False
                                
                                if use_variations and self.model_name and "dall-e-2" in self.model_name.lower():
                                    # Use variations endpoint (only supported by DALL-E 2)
                                    files, data = self._build_openai_variation_request(user_input)
                                    response = self._call_openai_variation_api(files, data)
                                else:
                                    # Use edits endpoint (supported by DALL-E 2 and gpt-image-1)
                                    files, data = self._build_openai_edit_request(user_input)
                                    response = self._call_openai_edit_api(files, data)
                                
                                key, img = self._parse_openai_response(response, user_input)
                                if img is not None:
                                    # Save single image immediately
                                    single_image_dict = {key: [img]}
                                    self.image_io(single_image_dict)
                                    images_dict.setdefault(key, []).append(img)
                            except Exception as e:
                                self.logger.error(f"Failed to process image editing request: {str(e)}")
                                continue
                        else:
                            # Image generation task (text2image)
                            payload = self._build_openai_request(user_input)
                            response = self._call_api(payload)
                            key, img = self._parse_openai_response(response, user_input)
                            if img is not None:
                                # Save single image immediately
                                single_image_dict = {key: [img]}
                                self.image_io(single_image_dict)
                                images_dict.setdefault(key, []).append(img)
                    
                    elif self.api_format == "gemini":
                        # If image editing task, check if using multi-turn chat mode
                        image_paths = None
                        conversations = None
                        use_chat_mode = False
                        prompt = None
                        
                        if self.image_gen_task == "imageedit":
                            if isinstance(user_input, dict):
                                # Check if there is conversation history (multi-turn editing)
                                conversations = user_input.get("conversations")
                                if conversations and isinstance(conversations, list) and len(conversations) > 1:
                                    # Use chat mode for multi-turn editing
                                    use_chat_mode = True
                                else:
                                    # Single-turn mode: extract prompt
                                    prompt = self._build_gemini_prompt(user_input)
                                
                                # Extract image paths (support multiple images)
                                image_path = user_input.get("image_path") or user_input.get("image") or user_input.get("input_image") or user_input.get("images")
                                
                                # Process image paths: maybe single path or path list
                                if image_path:
                                    if isinstance(image_path, list):
                                        image_paths = [path for path in image_path if isinstance(path, str)]
                                    elif isinstance(image_path, str):
                                        image_paths = [image_path]
                                    else:
                                        image_paths = []
                                    
                                    if not image_paths and not use_chat_mode:
                                        continue
                                elif not use_chat_mode:
                                    continue
                            else:
                                continue
                        else:
                            # Text-to-image generation task
                            prompt = self._build_gemini_prompt(user_input)
                        
                        # Choose calling method based on whether there are multiple conversation turns
                        if use_chat_mode:
                            # Multi-turn chat mode
                            aspect_ratio = user_input.get("aspect_ratio")
                            resolution = user_input.get("resolution")
                            img_result = self._call_gemini_chat_api(
                                conversations=conversations,
                                image_paths=image_paths,
                                aspect_ratio=aspect_ratio,
                                resolution=resolution
                            )
                        else:
                            # Single-turn mode
                            img_result = self._call_gemini_api(prompt, image_paths=image_paths)
                        
                        if img_result is not None:
                            if isinstance(user_input, dict):
                                if self.image_gen_task == "imageedit":
                                    idx = user_input.get("idx")
                                    if idx is not None:
                                        key = f"sample_{idx}"
                                    else:
                                        key = str(user_input.get("prompt", "default"))
                                else:
                                    key = str(user_input.get("sample_id", user_input.get("text_prompt", "default")))
                            else:
                                key = str(user_input)
                            
                            # Process returned images
                            if isinstance(img_result, list):
                                for img_idx, img in enumerate(img_result):
                                    img_key = f"{key}_{img_idx}" if len(img_result) > 1 else key
                                    single_image_dict = {img_key: [img]}
                                    self.image_io(single_image_dict)
                                    images_dict.setdefault(key, []).append(img)
                            else:
                                single_image_dict = {key: [img_result]}
                                self.image_io(single_image_dict)
                                images_dict.setdefault(key, []).append(img_result)
                    
                    else:
                        raise ValueError(f"Unsupported api_format: {self.api_format}. Only 'openai' and 'gemini' are supported.")
                
                except requests.exceptions.Timeout as e:
                    self.logger.error(
                        f"Request timeout for input. Connect timeout: {self.connect_timeout}s, "
                        f"Read timeout: {self.timeout}s. Error: {str(e)}"
                    )
                    self.logger.warning(
                        "If you're behind a firewall or proxy, you may need to configure it. "
                        "Alternatively, try using --serving_type local for local model."
                    )
                    continue
                except requests.exceptions.ConnectionError as e:
                    self.logger.error(
                        f"Connection error: Cannot connect to {self.api_url}. "
                        f"Please check your network connection or API endpoint. Error: {str(e)}"
                    )
                    continue
                except requests.exceptions.HTTPError as e:
                    if hasattr(e.response, 'status_code'):
                        if e.response.status_code == 503:
                            self.logger.error(
                                f"503 Service Unavailable: The API server is temporarily unavailable. "
                                f"Please check if the service is running or try again later. "
                                f"Endpoint: {getattr(e, 'request', {}).get('url', 'N/A')}"
                            )
                        else:
                            self.logger.error(
                                f"HTTP {e.response.status_code} error: {str(e)}. "
                                f"Response: {e.response.text[:200] if hasattr(e.response, 'text') else 'N/A'}"
                            )
                    else:
                        self.logger.error(f"HTTP error: {str(e)}")
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing single input: {str(e)}")
                    continue
            
            # Save images using image_io, keep the same return value as LocalImageGenServing
            return self.image_io(images_dict)
            
        except Exception as e:
            self.logger.error(f"Error in generate_from_input_one_batch: {str(e)}")
            # Return empty dictionary to avoid pipeline crash
            return {}

    def cleanup(self):
        self.logger.info("APIImageGenServing cleanup completed")