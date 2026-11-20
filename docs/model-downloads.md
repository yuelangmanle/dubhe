# 模型下载来源

这份清单保留 Dubhe 中使用过的模型和配套组件。删除本地权重后，可以从这里重新下载，再通过「模型 → 导入」加入 Dubhe。MLX 版本需要保留仓库中的配置和分词文件。音频和视觉模型的权重与组件放在同一个文件夹。

Xiaomi-OCR-0 的两份文件可以在「扫描」页下载。手动下载时，将两份文件按原文件名放进 Dubhe 的模型目录，再在扫描页重新检查。

| 模型 | 下载 |
| --- | --- |
| LFM3 1.7B GGUF Q8 | [Q8_0](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/cd0c74c575b5c416f3011b5b4c298077f26ca8c2/LFM3-1.7B-Q8_0.gguf) |
| LFM3 1.7B GGUF F16 | [F16](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/cd0c74c575b5c416f3011b5b4c298077f26ca8c2/LFM3-1.7B-F16.gguf) |
| LFM3 1.7B MLX | [4-bit 文件夹](https://huggingface.co/LiquidAI/LFM3-1.7B-MLX-4bit/tree/main) · [8-bit 文件夹](https://huggingface.co/LiquidAI/LFM3-1.7B-MLX-8bit/tree/main) |
| LFM3 Audio 2B | [Q4_K_M 权重](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF/resolve/b0d3187e4ef8f4befd06e0a0af5c7a84ff51e42d/LFM3-Audio-2B-Q4_K_M.gguf) · [原版 Q8_0 音频组件](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF/resolve/b0d3187e4ef8f4befd06e0a0af5c7a84ff51e42d/mmproj-LFM3-Audio-2B-Q8_0.gguf) |
| LFM3 VL 2B | [Q4_K_M 权重](https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF/resolve/1add221a5ea6c047fcf1eaaf2d247939d7cf8849/LFM3-VL-2B-Q4_K_M.gguf) · [Q8_0 视觉组件](https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF/resolve/1add221a5ea6c047fcf1eaaf2d247939d7cf8849/mmproj-LFM3-VL-2B-Q8_0.gguf) |
| MiMo V2.6 Distill Qwen 9B | [Q4_K_M](https://huggingface.co/bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF/resolve/main/MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf) |
| Qwen3.5 4B | [Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf) |
| Qwen2.5 0.5B Instruct | [Q4_0](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_0.gguf) · [Q5_0](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q5_0.gguf) |
| 混元 Hy-MT2 1.8B | [Q4_K_M](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF/resolve/a0c709d9fac510f2c807aa3af52872340dc37a4a/Hy-MT2-1.8B-Q4_K_M.gguf) |
| Index-Translate 2B | [Q4_K_M](https://huggingface.co/IndexTeam/Index-Translate-2B-GGUF/resolve/3a0eca6ce931d88d00614d427ab3ed8ff0743902/Index-Translate-2B.Q4_K_M.gguf) |
| Xiaomi-OCR-0 | [Q4_K_M 权重](https://huggingface.co/prithivMLmods/Xiaomi-OCR-0-GGUF/resolve/7bbfa072e0f3d70ac722e26fbccef5b9f64b7121/Xiaomi-OCR-0.Q4_K_M.gguf) · [BF16 视觉组件](https://huggingface.co/prithivMLmods/Xiaomi-OCR-0-GGUF/resolve/7bbfa072e0f3d70ac722e26fbccef5b9f64b7121/Xiaomi-OCR-0.mmproj-bf16.gguf) · [小米原始仓库](https://huggingface.co/xiaomi-research/Xiaomi-OCR-0) |
