# Awesome AI: GPU Architecture Training Resources

## Overview
Explore the cutting-edge advancements in NVIDIA GPU architectures, tailored for the growing demands of AI and large-scale model training. This repository highlights key architectures like the Grace Hopper Superchip, Blackwell, and Hopper, providing insights into their capabilities, performance, and innovations.

### Key GPU Architectures

#### **GH200: Grace Hopper Superchip**
- **Highlights**:
  - Integrates NVIDIA Grace CPU with Hopper GPU.
  - Unified memory via Coherent NVLink, offering **384GB HBM3e memory** and **16 TB/s bandwidth**.
  - Optimized for large-scale AI model training and inference.
  - Features decompression engines, multimedia decoders, and ARM Neoverse V2 cores for efficient data handling.

#### **B200: Blackwell Architecture (2024)**
- **Performance**:
  - Up to **9 PFLOPS** dense FP4 tensor operations, **18 PFLOPS** sparse FP4.
  - Equipped with **192GB HBM3e memory** and **8 TB/s bandwidth**.
- **Connectivity**:
  - NVLink 5 and PCIe Gen6 for high-speed data transfer.
- **Power**:
  - **1000W TDP**, designed for high-performance AI tasks.

#### **B100: Blackwell Architecture (2024)**
- **Performance**:
  - Offers **7 PFLOPS** dense FP4, **14 PFLOPS** sparse FP4.
  - **192GB HBM3e memory** with **8 TB/s bandwidth**.
- **Connectivity**:
  - NVLink provides **1.8 TB/s bandwidth** for HPC workloads.
- **Power**:
  - **700W TDP**, energy-efficient for versatile applications.

#### **H200: Hopper Architecture (2024)**
- **Performance**:
  - Equipped with **141GB HBM3e memory** and **4.8 TB/s bandwidth**.
  - 1.6x faster inference for GPT-3 and 1.9x for LLaMA2 70B.
- **Efficiency**:
  - Reduces energy consumption by 50% compared to H100.

#### **H100: Hopper Architecture (2022)**
- **Key Features**:
  - NVLink-C2C for direct GPU-to-GPU communication.
  - Includes Tensor Cores and Transformer Engine for AI workloads.
- **Power**:
  - **700W TDP**, with enhanced energy efficiency.

---

## Additional Learning Resources

### **Articles & Guides**
- [The Journey to High-Performance AI: A Deep Dive into NVIDIA’s GPU Architecture](https://developer.nvidia.com/blog)  
  Insight into AI workloads and how NVIDIA GPUs are revolutionizing the field.

- [VLLM Blog](https://blog.vllm.ai/2023/06/20/vllm.html)  
  Explores distributed machine learning and scaling models across GPUs.

- [Efficient Finetuning and Distributed Training](https://sumanthrh.com/post/distributed-and-efficient-finetuning/)  
  Techniques for efficient AI model finetuning.

- [AI Accelerators Comparison](https://towardsdatascience.com/a-complete-guide-to-ai-accelerators-for-deep-learning-inference)  
  A comprehensive analysis of AI accelerators, including GPUs and AWS Inferentia.

- [Understanding GPU Memory: A PyTorch Series](https://pytorch.org/blog/understanding-gpu-memory-1/)  
  - Part 1: Memory allocation visualization.  
  - Part 2: Resolving memory reference cycles.

- [Scaling AI with Distributed Training](https://mlat.microsoft.com/distributed-training)  
  A guide to scaling AI models effectively across distributed systems.

- [NVIDIA’s Megatron-LM](https://developer.nvidia.com/megatron-lm)  
  Learn how NVIDIA’s Megatron-LM supports training large language models efficiently.

- [Inside Hopper: Architecture and Applications](https://developer.nvidia.com/blog/inside-hopper-architecture-and-applications)  
  Detailed breakdown of Hopper GPU architecture and its applications.

---

### **Certifications**
- [(NCA-AIIO) AI Infrastructure and Operations](https://www.nvidia.com/en-us/learn/certification/ai-infrastructure-operations-associate/)  
- [(NCA-GENL) Generative AI LLMs](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/)  
- [(NCP-IB) InfiniBand](https://www.nvidia.com/en-us/learn/certification/infiniband-professional/)  
- [(NCA-DLF) Deep Learning Fundamentals](https://www.nvidia.com/en-us/learn/certification/deep-learning-fundamentals/)  
- [Google Cloud Professional Machine Learning Engineer Certification](https://cloud.google.com/certification/machine-learning-engineer)  
- [AWS Certified Machine Learning - Specialty](https://aws.amazon.com/certification/certified-machine-learning-specialty/)  
- [Microsoft Certified: Azure AI Engineer Associate](https://learn.microsoft.com/en-us/certifications/azure-ai-engineer/)  

---

### **Courses**
- [Deep Learning Fundamentals](https://lightning.ai/courses/deep-learning-fundamentals/)  
- [Google - Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)  
- [Coursera - Machine Learning Specialization by Andrew Ng](https://www.coursera.org/specializations/machine-learning-introduction)  
- [Fast.ai - Practical Deep Learning for Coders](https://course.fast.ai/)  
- [NVIDIA DLI - Fundamentals of Deep Learning](https://www.nvidia.com/en-us/training/dli/)  
- [edX - AI for Everyone by Andrew Ng](https://www.edx.org/course/ai-for-everyone)  
- [DeepMind’s Introduction to Machine Learning](https://www.deepmind.com/learning-resources/machine-learning)  

---

### **Reference Architectures**
- [NVIDIA DGX SuperPOD (H100): Scalable Infrastructure for AI Leadership (PDF)](https://docs.nvidia.com/https:/docs.nvidia.com/dgx-superpod-reference-architecture-dgx-h100.pdf)  
- [NVIDIA DGX SuperPOD Architecture (Web)](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/dgx-superpod-architecture.html)  
- [AWS EC2 P4d Instances for AI/ML](https://aws.amazon.com/ec2/instance-types/p4/)  
- [Google Cloud TPU Reference Architecture](https://cloud.google.com/tpu/docs/reference)  
- [Azure AI Infrastructure Guide](https://learn.microsoft.com/en-us/azure/machine-learning/)  

---

### **Tools for GPU Optimization**
- [Nsight Compute](https://developer.nvidia.com/nsight-compute): GPU kernel analysis and optimization.
- [Ipyexperiments](https://github.com/stas00/ipyexperiments): GPU memory management for Jupyter notebooks.
- [Pytorch_memlab](https://github.com/Stonesjtu/pytorch_memlab): PyTorch GPU memory profiler.
- [Nvtop](https://github.com/Syllo/nvtop): GPU process monitoring utility.
- [Gpustat](https://github.com/wookayin/gpustat): Real-time GPU resource monitoring.
- [TensorFlow Profiler](https://www.tensorflow.org/tfx/guide/profiler): Profiling TensorFlow workloads for performance optimization.
- [Horovod](https://github.com/horovod/horovod): Distributed training framework for TensorFlow, PyTorch, and MXNet.

---

### **Distributed Training Frameworks**
- [DeepSpeed](https://github.com/microsoft/DeepSpeed): Large-scale model training framework.  
- [Kyle’s Distributed Systems Testing Write-ups](https://jepsen.io/analyses): Detailed analyses of distributed systems in AI.
- [Ray Train](https://docs.ray.io/en/latest/train/user-guide.html): Scalable distributed training for deep learning models.
- [TensorFlow Distributed Strategy](https://www.tensorflow.org/guide/distributed_training): Guide to distributed training with TensorFlow.
- [PyTorch DDP (Distributed Data Parallel)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html): Tutorial for distributed data parallelism in PyTorch.

---

## Contribution
Contributions are welcome! If you have additional resources, guides, or tools to recommend, feel free to create a pull request or open an issue. Let’s make this repository the go-to guide for AI enthusiasts and professionals!
