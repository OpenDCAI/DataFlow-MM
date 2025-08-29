import os
from openai import OpenAI
import base64
import requests
from PIL import Image
import io
import json
import re


class GeminiImageEditor:
    def __init__(self, base_url, api_key="sk-HVNugK6ODNXTg1tOImCu8FZ39GMRDlbu56EnWjQ8qUJJPZjU"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.api_url = base_url
        self.api_key = api_key
        self.model = "gemini-2.5-flash-image-preview"
    
    def generate_image(self, prompt, save_path="generated.png"):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": f"基于提示词：{prompt}，生成一张图片。不要其他任何文字描述。"}]
            )
            
            content = response.choices[0].message.content
            print(f"Response content type: {type(content)}")
            print(f"Content preview: {content[:200] if len(content) > 200 else content}")
            
            # 尝试从返回内容中提取图片
            image_saved = self._extract_and_save_image(content, save_path)
            
            if image_saved:
                print(f"图片已保存到: {save_path}")
                return save_path
            else:
                print("警告：无法从响应中提取图片")
                return None
                
        except Exception as e:
            print(f"生成图片时出错: {str(e)}")
            return None
    
    def _encode_image_to_base64(self, image_path: str):
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

    def edit_image(self, image_path, edit_prompt, save_path="edited.png"):
        try:
            # # 读取并编码图片
            # with open(image_path, "rb") as f:
            #     image_data = f.read()
            #     base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # # 检测图片格式
            # image_format = self._detect_image_format(image_data)

            base64_image, image_format = self._encode_image_to_base64(image_path)
            
            # response = self.client.chat.completions.create(
            #     model=self.model,
            #     messages=[{
            #         "role": "user",
            #         "content": [
            #             {"type": "text", "text": edit_prompt},
            #             {
            #                 "type": "image_url",
            #                 "image_url": {"url": f"data:image/{image_format};base64,{base64_image}"}
            #             }
            #         ]
            #     }]
            # )

            payload = json.dumps({
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": edit_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{image_format};base64,{base64_image}"}
                        }
                    ]
                }]
            })

            headers = {
                'Authorization': f"Bearer {self.api_key}",
                'Content-Type': 'application/json',
                'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
            }
            
            response = requests.post(self.api_url, headers=headers, data=payload, timeout=1800)

            if response.status_code == 200:
                response_data = response.json()
                content = response_data['choices'][0]['message']['content']
            # content = response.choices[0].message.content
            print(f"Edit response preview: {content[:200] if len(content) > 200 else content}")
            
            # 尝试从返回内容中提取图片
            image_saved = self._extract_and_save_image(content, save_path)
            
            if image_saved:
                print(f"编辑后的图片已保存到: {save_path}")
                return save_path
            else:
                print("警告：无法从响应中提取图片")
                return None
                
        except Exception as e:
            print(f"编辑图片时出错: {str(e)}")
            return None
    
    def _detect_image_format(self, image_data):
        """检测图片格式"""
        if image_data[:2] == b'\xff\xd8':
            return 'jpeg'
        elif image_data[:8] == b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a':
            return 'png'
        else:
            return 'jpeg'  # 默认
    
    def _extract_and_save_image(self, content, save_path):
        """从内容中提取并保存图片"""
        try:
            # 情况1: 内容是纯base64字符串
            if self._is_base64(content):
                image_data = base64.b64decode(content)
                with open(save_path, "wb") as f:
                    f.write(image_data)
                return True
            
            # 情况2: 内容是data URL格式 (data:image/xxx;base64,...)
            if content.startswith("data:image"):
                base64_str = content.split(",")[1]
                image_data = base64.b64decode(base64_str)
                with open(save_path, "wb") as f:
                    f.write(image_data)
                return True
            
            # 情况3: 内容是包含base64图片的JSON
            try:
                json_data = json.loads(content)
                if isinstance(json_data, dict):
                    # 查找可能的图片字段
                    for key in ['image', 'data', 'result', 'output', 'content']:
                        if key in json_data:
                            return self._extract_and_save_image(json_data[key], save_path)
            except json.JSONDecodeError:
                pass
            
            # 情况4: 内容中包含base64字符串（使用正则表达式提取）
            base64_pattern = r'([A-Za-z0-9+/]{100,}={0,2})'
            matches = re.findall(base64_pattern, content)
            for match in matches:
                if self._is_base64(match):
                    try:
                        image_data = base64.b64decode(match)
                        # 验证是否是有效的图片
                        Image.open(io.BytesIO(image_data))
                        with open(save_path, "wb") as f:
                            f.write(image_data)
                        return True
                    except:
                        continue
            
            # 情况5: 内容是图片URL
            if content.startswith(('http://', 'https://')):
                response = requests.get(content, timeout=30)
                if response.status_code == 200:
                    with open(save_path, "wb") as f:
                        f.write(response.content)
                    return True
            
            return False
            
        except Exception as e:
            print(f"保存图片时出错: {str(e)}")
            return False
    
    def _is_base64(self, s):
        """检查字符串是否是有效的base64"""
        try:
            if isinstance(s, str):
                # 去除空白字符
                s = s.strip()
                # 检查是否只包含base64字符
                if re.match(r'^[A-Za-z0-9+/]*={0,2}$', s):
                    # 尝试解码
                    base64.b64decode(s)
                    return True
            return False
        except:
            return False


# 使用示例
if __name__ == "__main__":
    # editor = GeminiImageEditor("http://35.220.164.252:3888/v1")
    editor = GeminiImageEditor("http://123.129.219.111:3000/v1/chat/completions")
    # editor = GeminiImageEditor("https://www.apihy.com/v1/chat/completions")
    # editor = GeminiImageEditor("https://www.apihy.com/v1")
    
    # # 生成图片
    # result = editor.generate_image("A horse playing in the garden")
    
    # if result:
    #     print(f"成功生成图片: {result}")
        
    # 编辑图片
    edited_result = editor.edit_image("human_inpaint.jpg", "把黑影部分补全，生成一张通用人像，只返回图片的base64编码，不要返还其他文字", "edited_human.png")
    if edited_result:
        print(f"成功编辑图片: {edited_result}")