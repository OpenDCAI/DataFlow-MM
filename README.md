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

## 📰 1. News

## 🔍 2. Overview

<!--  ![dataflow_framework](https://github.com/user-attachments/assets/b44db630-754a-44a8-bec7-6d350bf5ed61) -->

![df_overview_final_300](https://github.com/user-attachments/assets/57dd0838-6e24-4814-a89a-02ca0667bd5c)

DataFlow series is a data preparation and training system designed to **parse, generate, process, and evaluate** high-quality data from noisy sources (PDF, plain-text, low-quality QA), thereby improving the performance of large language models (LLMs) in specific domains through targeted training (Pre-training, Supervised Fine-tuning, RL training) or RAG using knowledge base cleaning. 

Specifically, we are constructing diverse `operators` leveraging rule-based methods, deep learning models, LLMs, and LLM APIs. These operators are systematically integrated into distinct `pipelines`, collectively forming the comprehensive `DataFlow system`. Additionally, we develop an intelligent `DataFlow-agent` capable of dynamically assembling new `pipelines` by recombining existing `operators` on demand.

DataFlow-MM is the multimodal extension version of the awesome repo [DataFlow](https://github.com/OpenDCAI/DataFlow)


## Quick Start

### Installation

First, clone the repository and install **DataFlow-MM** in editable mode:

```bash
cd ./DataFlow-MM
conda create -n dataflow-mm python=3.12
conda activate dataflow-mm
pip install -e .
```

#### Optional Dependencies

Install additional dependencies based on your use case:

**Audio environment**

```bash
pip install -e ".[audio]"
```

**Image environment**

```bash
pip install -e ".[image]"
```

---

### Initialize a DataFlow Workspace

Create and initialize a DataFlow-MM workspace:

```bash
mkdir test_dataflow
cd test_dataflow
dataflowmm init
```

This command will generate the basic directory structure and configuration files required to run DataFlow-MM pipelines.

---

### Demo Data

To run the **Image** or **Video** examples, please download the corresponding demo datasets from Hugging Face (GitHub is not suitable for hosting large files):

* **Image Examples**:
  [https://huggingface.co/datasets/OpenDCAI/dataflow-demo-image](https://huggingface.co/datasets/OpenDCAI/dataflow-demo-image)

* **Video Examples**:
  [https://huggingface.co/datasets/OpenDCAI/dataflow-demo-video](https://huggingface.co/datasets/OpenDCAI/dataflow-demo-video)

After downloading, place the data in the "test_dataflow/example" directory as instructed in each example.

