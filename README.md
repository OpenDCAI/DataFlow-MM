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

python /mnt/public/data/guotianyu/dataflow_project/DataFlow-MM/test/test_merge.py
python /mnt/public/data/guotianyu/dataflow_project/DataFlow-MM/test/test_ctc_forced_aligner_filter.py
python /mnt/public/data/guotianyu/dataflow_project/DataFlow-MM/test/test_ctc_forced_aligner.py
python /mnt/public/data/guotianyu/dataflow_project/DataFlow-MM/test/test_silero_vad_generator.py
python /mnt/public/data/guotianyu/dataflow_project/DataFlow-MM/test/test_whisper_promptedaqa.py
python /mnt/public/data/guotianyu/dataflow_project/DataFlow-MM/test/test_promptedaqa.py
```


# nano-banana (gemini-v2.5-image)测试
测试命令
```bash
python test/test_image_editing.py --api_key < your api key >
```
we utilize the api from [yucha](http://123.129.219.111:3000/)
