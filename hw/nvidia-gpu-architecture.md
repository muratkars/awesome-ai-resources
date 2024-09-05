
# Awesome AI: GPU Architecture Training Resources



## Overview
This guide provides an in-depth look at NVIDIA's GPU architectures, focusing on the capabilities and performance improvements across various generations. From the powerful Grace Hopper Superchip to the Blackwell and Hopper architectures, these GPUs are designed to meet the growing demands of AI and large-scale model training.

### GH200: Grace Hopper Superchip
- Combines NVIDIA Grace CPU with Hopper GPU.
- Unified memory architecture via Coherent NVLink with up to **384GB HBM3e** memory and **16 TB/s bandwidth**.
- Designed for large-scale AI model training and inference, offering improved energy efficiency and speed.
- Includes a decompression engine, multimedia decoders, and ARM Neoverse V2 cores for optimal data handling.

### B200: Blackwell Architecture (2024)
- Delivers up to **9 PFLOPS** for dense FP4 tensor operations and **18 PFLOPS** for sparse FP4.
- Supports **192GB HBM3e memory** with **8 TB/s bandwidth**.
- Features NVLink 5 and PCIe Gen6 for enhanced data transfer.
- Power: **1000W TDP**, optimized for high-performance AI tasks.

### B100: Blackwell Architecture (2024)
- Offers up to **7 PFLOPS** for dense FP4 tensor operations and **14 PFLOPS** for sparse FP4.
- Supports **192GB HBM3e memory** with **8 TB/s bandwidth**.
- Enhanced NVLink with **1.8 TB/s bandwidth** for HPC workloads.
- Power: **700W TDP**, energy-efficient for varied setups.

### H200: Hopper Architecture (2024)
- Equipped with **141GB HBM3e memory** and **4.8 TB/s bandwidth**.
- Optimized for large language models (LLMs) and HPC workloads.
- Provides 1.6x faster inference performance for GPT-3 and 1.9x faster for LLaMA2 70B.
- Energy consumption reduced by 50% compared to H100.

### H100: Hopper Architecture (2022)
- Features NVLink-C2C for direct GPU-to-GPU communication.
- Designed for advanced AI tasks and large-scale models.
- Incorporates Tensor Cores and a Transformer Engine for enhanced performance.
- Power: **700W TDP**, with optimized energy efficiency.

---

## Additional Learning Resources

### Articles & Guides
- **[The Journey to High-Performance AI: A Deep Dive into NVIDIA’s GPU Architecture](https://developer.nvidia.com/blog)**
  - Overview of AI and ML workloads and how NVIDIA GPUs cater to them.
  
- **[VLLM Blog](https://blog.vllm.ai/2023/06/20/vllm.html)** 
  - Understanding distributed machine learning and scaling models across GPUs.
  
- **[Efficient Finetuning and Distributed Training](https://sumanthrh.com/post/distributed-and-efficient-finetuning/)**
  - Comprehensive guide to efficient finetuning techniques for AI models.

- **[AI Accelerators Comparison](https://towardsdatascience.com/a-complete-guide-to-ai-accelerators-for-deep-learning-inference)**
  - Detailed comparison of GPUs, AWS Inferentia, and Amazon Elastic Inference for AI workloads.

- **[Understanding GPU Memory: A PyTorch Series](https://pytorch.org/blog/understanding-gpu-memory-1/)**
  - Part 1: Visualizing memory allocation over time.
  - Part 2: Identifying and resolving memory reference cycles.

## Certifications
- [(NCA-AIIO) AI Infrastructure and Operations](https://www.nvidia.com/en-us/learn/certification/ai-infrastructure-operations-associate/)
- [(NCA-GENL) Generative AI LLMs](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/)
- [(NCP-IB) InfiniBand](https://www.nvidia.com/en-us/learn/certification/infiniband-professional/)
 
### Course
- [Deep Learning Fundamentals](https://lightning.ai/courses/deep-learning-fundamentals/)
- [Google - Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)

### Reference Architecture
- NVIDIA DGX SuperPOD (H100)
  - [Next Generation Scalable Infrastructure for AI Leadership (PDF)](https://docs.nvidia.com/https:/docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf)
  - [NVIDIA DGX SuperPOD (H100) Architecture (WEB)](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/dgx-superpod-architecture.html)

### Tools
- **[Nsight Compute](https://developer.nvidia.com/nsight-compute)**: A tool for analyzing GPU kernels and optimizing performance.
- **[Ipyexperiments](https://github.com/stas00/ipyexperiments)**: Lightweight tool for managing GPU memory in Jupyter notebooks.
- **[Pytorch_memlab](https://github.com/Stonesjtu/pytorch_memlab)**: A library for profiling PyTorch GPU memory usage.
- **[Nvtop](https://github.com/Syllo/nvtop)**: GPU process monitoring utility.
- **[Gpustat](https://github.com/wookayin/gpustat)**: GPU resource monitoring.
  
### Distributed Training
- **[DeepSpeed](https://github.com/microsoft/DeepSpeed)**: A framework for distributed training and large-scale model finetuning.

- **[Kyle’s Distributed Systems Testing Write-ups](https://jepsen.io/analyses)**: In-depth analysis of distributed system behavior in large-scale AI workloads.


