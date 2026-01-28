# Dataflow-MM
<div align="center">
  <img src="https://github.com/user-attachments/assets/3fe636ad-3026-4faf-aa44-c84b8f97a05d">

[![Documents](https://img.shields.io/badge/Documents-Click_here-brightgreen?logo=read-the-docs)](https://OpenDCAI.github.io/Dataflow-MM-Doc/)
[![](https://img.shields.io/github/license/OpenDCAI/Dataflow-MM)](https://github.com/OpenDCAI/Dataflow-MM/blob/main/LICENSE)
[![](https://img.shields.io/github/stars/OpenDCAI/Dataflow-MM?style=social)](https://github.com/OpenDCAI/Dataflow-MM)
[![](https://img.shields.io/github/contributors/OpenDCAI/Dataflow-MM)](https://github.com/OpenDCAI/Dataflow-MM/graphs/contributors)
[![](https://img.shields.io/github/repo-size/OpenDCAI/Dataflow-MM?color=green)](https://github.com/OpenDCAI/Dataflow-MM)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/OpenDCAI/Dataflow-MM)

<!-- [![](https://img.shields.io/github/last-commit/OpenDCAI/Dataflow-MM)](https://github.com/OpenDCAI/Dataflow-MM/commits/main/) -->
<!--[![](https://img.shields.io/github/issues-raw/OpenDCAI/Dataflow-MM)](https://github.com/OpenDCAI/Dataflow-MM/issues) -->
🎉 If you like our project, please give us a star ⭐ on GitHub for the latest update.

[简体中文](./README-zh.md) | English
</div>

## Quick Start
Install with the following command:
```bash
cd ./Dataflow-MM
conda create -n Dataflow-MM python=3.12
pip install -e .
```

## Audio Test
Extra environments:
```bash
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
python /mnt/public/data/guotianyu/dataflow_project/DataFlow-MM/test/test_audio_asr_pipeline.py
```

# nano-banana (gemini-v2.5-image) Test
测试命令
```bash
python test/test_image_editing.py --api_key < your api key >
```
we utilize the api from [yucha](http://123.129.219.111:3000/)

## 多参考图生成测试
测试命令
```bash
python test/test_echo4o_w_nano.py --api_key < your api key >
```
we utilize the api from [yucha](http://123.129.219.111:3000/)

# data selection测试脚本
测试命令
```bash
python test/test_data_selection.py

# 目前没完成处理好显存占用问题。建议datatailor单独测试，剩下的可以一起测试
```
