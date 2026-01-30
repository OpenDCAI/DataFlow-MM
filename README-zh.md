# Dataflow-MM 多模态

<div align="center">
  <img src="./static/images/Face.jpg">

[![Documents](https://img.shields.io/badge/官方文档-单击此处-brightgreen?logo=read-the-docs)](https://OpenDCAI.github.io/Dataflow-MM-Doc/)
[![](https://img.shields.io/github/license/OpenDCAI/Dataflow-MM)](https://github.com/OpenDCAI/Dataflow-MM/blob/main/LICENSE)
[![](https://img.shields.io/github/stars/OpenDCAI/Dataflow-MM?style=social)](https://github.com/OpenDCAI/Dataflow-MM)
[![](https://img.shields.io/github/issues-raw/OpenDCAI/Dataflow-MM)](https://github.com/OpenDCAI/Dataflow-MM/issues)
[![](https://img.shields.io/github/contributors/OpenDCAI/Dataflow-MM)](https://github.com/OpenDCAI/Dataflow-MM/graphs/contributors)
[![](https://img.shields.io/github/repo-size/OpenDCAI/Dataflow-MM?color=green)](https://github.com/OpenDCAI/Dataflow-MM)

<!-- [![](https://img.shields.io/github/last-commit/OpenDCAI/Dataflow-MM)](https://github.com/OpenDCAI/Dataflow-MM/commits/main/) -->

🎉 如果你认可我们的项目，欢迎在 GitHub 上点个 ⭐ Star，关注项目最新进展。

简体中文 | [English](./README.md)
</div>

## 📰 1. 项目动态（News）

## 🔍 2. 项目概览（Overview）

<!--  ![dataflow_framework](https://github.com/user-attachments/assets/b44db630-754a-44a8-bec7-6d350bf5ed61) -->

![df\_overview\_final\_300](https://github.com/user-attachments/assets/57dd0838-6e24-4814-a89a-02ca0667bd5c)

**DataFlow 系列**是一个面向大模型的数据准备与训练系统，旨在从噪声数据源（如 PDF、纯文本、低质量问答数据等）中**解析、生成、处理并评估**高质量数据，从而通过有针对性的训练（预训练、监督微调、强化学习）或基于知识库清洗的 RAG 流程，显著提升大语言模型（LLMs）在特定领域中的性能。

具体而言，我们构建了大量多样化的 `operators`，涵盖基于规则的方法、深度学习模型、LLMs 以及 LLM API。这些 `operators` 被系统性地组织并集成到不同的 `pipelines` 中，整体构成完整的 **DataFlow 系统**。此外，我们还开发了智能化的 **DataFlow-agent**，能够根据需求动态重组已有的 `operators`，自动构建新的 `pipelines`，以适配不同的数据处理与建模任务。

**DataFlow-MM** 是优秀开源项目
[DataFlow](https://github.com/OpenDCAI/DataFlow)
的 **多模态扩展版本**，支持图像、视频、音频等多模态数据的统一处理与训练。

---

## 🚀 快速开始（Quick Start）

### 安装（Installation）

首先，克隆仓库并以可编辑模式安装 **DataFlow-MM**：

```bash
cd ./DataFlow-MM
conda create -n dataflow-mm python=3.12
conda activate dataflow-mm
pip install -e .
```

#### 可选依赖（Optional Dependencies）

根据使用场景安装对应的可选依赖：

**音频环境（Audio）**

```bash
pip install -e ".[audio]"
```

**图像环境（Image）**

```bash
pip install -e ".[image]"
```

---

### 初始化 DataFlow 工作空间

创建并初始化一个 DataFlow-MM 工作目录：

```bash
mkdir test_dataflow
cd test_dataflow
dataflowmm init
```

该命令会自动生成运行 DataFlow-MM pipelines 所需的基础目录结构和配置文件。

---

### 示例数据（Demo Data）

如果需要运行 **Image** 或 **Video** 相关的示例，请先从 Hugging Face 下载对应的演示数据集（由于文件体积较大，不适合直接托管在 GitHub 上）：

* **图像示例（Image Examples）**：
  [https://huggingface.co/datasets/OpenDCAI/dataflow-demo-image](https://huggingface.co/datasets/OpenDCAI/dataflow-demo-image)

* **视频示例（Video Examples）**：
  [https://huggingface.co/datasets/OpenDCAI/dataflow-demo-video](https://huggingface.co/datasets/OpenDCAI/dataflow-demo-video)

下载完成后，请按照各示例的说明，将数据放置在 `test_dataflow/example` 目录下。
