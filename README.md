# audio测试

准备环境
```bash
cd ./DataFlow-MM
conda create -n df_audio_test python=3.12
pip install -e .
pip install -e ".[audio]"
pip install -e ".[vllm]"
```

测试命令
```bash
python /data0/gty/DataFlow-MM/test/test_whisper_promptedvqa.py
python /data0/gty/DataFlow-MM/test/test_audio_promptedvqa.py
```