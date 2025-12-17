from openai import OpenAI
import os
import base64


# 编码函数： 将本地文件转换为 Base64 编码的字符串
def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

# 将xxxx/test.mp4替换为你本地视频的绝对路径
base64_video = encode_video("/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/DataFlow-MM/api_test/test.mp4")
client = OpenAI(
    api_key="sk-ba117c8c8dbd465c8aac8e3bba7fdec2",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
    model="qwen3-vl-8b-instruct",  
    messages=[
        {
            "role": "user",
            "content": [
                {
                    # 直接传入视频文件时，请将type的值设置为video_url
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{base64_video}"},
                },
                {"type": "text", "text": "这段视频描绘的是什么景象?"},
            ],
        }
    ],
)
print(completion.choices[0].message.content)