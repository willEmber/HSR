```markdown
# HSR: Hierarchical Super-Resolution Multi-Image Hiding

**多图像隐藏 + 层级访问控制** 的图像隐写方案。当前分支 **`v3.0-1.5scale-3loop`** 默认采用 **1.5× 超分辨率** 与 **3 轮级联嵌入**（可隐藏 3 张秘密图像）。

- **基座网络**：参考并改造自 [AIDN](https://github.com/Doubiiu/AIDN) 的 **任意尺度可逆下采样/上采样** 能力（支持非整数放大，如 ×1.5），用于在超分过程中承载隐写信号。  
- **多图隐藏机制**：借鉴 [DeepMIH](https://github.com/TomTomTommi/DeepMIH) 的 **重要性图（Importance Map）** 思想，减少多轮嵌入带来的可见伪影与相互干扰。  
- **层级访问控制**：同一张 HR stego 图像支持 **跨层/单层** 解码；低权限用户只能从较低分辨率版本得到对应级别的秘密，高层秘密在低分辨率侧 **不可恢复**。


---

## ✨ 主要特性
- **一次分发，多级解码**：同一个高分辨率 stego，按权限以不同分辨率解密对应秘密；无需为不同人群生成多份载体，降低存储与带宽成本。  
- **任意尺度（含非整数）**：复用/改造 AIDN 的 Conditional Resampling/Scale-aware 设计，使 SR 和隐写耦合；当前默认 `×1.5`，可在配置中更换。  
- **多轮多图嵌入**：通过 **PIN（Permissions Integrated Network）+ IAM（Importance Attention Module）** 逐轮嵌入；**DEN（Distribution-Extraction Network）** 则负责单层/跨层提取。  
- **稳健性**：对“低分辨率上采样越权解码”“下采样-再上采样篡改”等攻击具备天然防护（详见文末“论文草稿（附）”）。

---

## 📦 仓库结构（节选）
Data/                     # 数据根目录（示例：DIV2K/Set5/Set14；也可切换到你的 COCO 配置）
 LOG/                      # 训练与评测日志/权重输出（示例：DIV2K/）
 assets/                   # 可选：示意图、样例图
 base/, utils/, metrics/   # 通用组件与评测指标
 config/                   # 训练与推理配置（yaml），含 scale、回合数、数据路径、损失权重等
 dataset/                  # 数据加载与切片脚本（如 DIV2K/Set5/Set14 或自定义数据集）
 models/                   # PIN/IAM/DEN 等网络结构与 AIDN 模块化改造
 scripts/                  # 常用启动脚本（训练/验证/批处理推理等）
 run_training.py           # 入口：读取 config/* 并启动训练/验证
 test_fix.py               # 推理脚本（单层/跨层示例参见下文）
 verify_fix.py             # 校验/演示越权失败与跨层提取
 apply_fix.py              # 批处理/实用工具
 requirements.txt          # 依赖列表
```



```
---

## 🔧 环境安装
```bash
# 1) 创建环境（建议 Python 3.8+；按你机器的 CUDA 选择合适的 PyTorch 版本）
python -m venv .venv && source .venv/bin/activate  # 或使用 conda

# 2) 安装依赖
pip install -r requirements.txt
```

> 说明：AIDN 原仓库示例使用了较老版本的 PyTorch；本项目以 `requirements.txt` 为准。

------

## 🗂️ 数据准备

- **默认示例**：`DIV2K` / `Set5` / `Set14`。将原始 HR 图像放到 `Data/`，并按 `config/` 中的路径填写；需要时可使用 `dataset/` 下的辅助脚本做裁剪/切片。
- **切换到 COCO 或自有数据**：在 `config/*.yaml` 中修改数据集名称与路径字段，使其指向你的数据根目录与列表文件。

------

## 🚀 训练（以 ×1.5 / 3 轮为例）

1. **编辑配置**：在 `config/` 中选择/创建一个配置（如 `train_sr15_3loop.yaml`），设置：
   - `scale: 1.5`
   - `rounds: 3`（可隐藏 3 张秘密图）
   - 数据集路径/列表、batch size、损失权重（`Lh, Lr, Ldist, Limp`）等
2. **启动训练**

```bash
python run_training.py  # 将读取上一步配置；如需多配置，请在脚本或环境变量中切换
```

1. **日志与权重**：默认保存在 `LOG/` 下的对应子目录（可在配置/脚本里改）。

------

## 🔍 推理 / 层级解码 Demo

- **单层解码**（从第 *n* 轮 stego 直接恢复第 *n* 张秘密）

```bash
python test_fix.py --weights path/to/ckpt.pth --input path/to/stegoN.png --out runs/...
```

- **跨层解码**（从高层 stego 下采样后恢复低层秘密）

```bash
# 将 Yn 以双三次下采样到目标层级后，再调用解码
python verify_fix.py --input path/to/stegoN.png --down 1-2 --out runs/...
```

- **越权失败示例**：将低层 stego 上采样到高分辨率再尝试恢复高层秘密，理论上会 **失败**（详见“论文草稿（附）”的权限校验实验结论）。

------

## 🧠 方法总览（PIN / IAM / DEN）

- **PIN（Permissions Integrated Network）**：在每一轮将一张秘密图嵌入上一轮输出（或原始封面），并通过 **尺度感知重采样模块** 与 **特征提取器** 在 `×1.5` 放大过程中“携带”隐写信息。
- **IAM（Importance Attention Module）**：生成重要性图，引导下一轮嵌入避开敏感区域，降低多轮级联导致的可见伪影。
- **DEN（Distribution-Extraction Network）**：负责分发/解码。支持 **单层**（直接从对应轮次 stego 恢复）与 **跨层**（先将高层 stego 下采样到目标层，再恢复）。
- **损失**：`Lh`（隐写/超分的容器保真）、`Lr`（秘密恢复）、`Ldist`（跨层分发/解码约束）、`Limp`（重要性图预训练/稳定）。

Warm-up IM（预热阶段）
- 仅前若干轮训练 IM，使 `x_imp ≈ x_stego_(t−1) − x_cover`，通过 `TRAIN.warmup_imp_epochs` 控制轮数、`LOSS.lambda_imp` 控制权重；联合阶段不再显式使用 `Limp`，避免 IM 退化为“简单残差”。

------

## 📊 评价指标与期望表现

- 本项目沿用 **PSNR / SSIM / APD(L1) / LPIPS** 评估 *cover ↔ stego* 与 *secret ↔ recovered* 的成对质量。
- 在两个秘密（2 轮）场景下，论文草稿报告了较强的恢复质量（如在 DIV2K 上 **Recovered1 ≈ 43.62 dB / 0.992，Recovered2 ≈ 41.85 dB / 0.987**），同时容器保真与跨层分发效果稳定（详见下文“论文草稿（附）”）。

------

## 🧩 引用与致谢

- **AIDN**（任意尺度可逆缩放）：Xing *et al.*, TIP 2023 — [GitHub](https://github.com/Doubiiu/AIDN)
- **DeepMIH**（多图隐藏与重要性图）：Guan *et al.*, TPAMI 2022 — [GitHub](https://github.com/TomTomTommi/DeepMIH)

如果本项目对你有帮助，请在引用中加入本仓库，并同时致谢上述开源工作。

------

## License

本项目仅供科研与教学使用；第三方依赖与上游代码请遵循其各自的开源协议。

------

# 论文草稿（附）

> 可将本节置于 README 末尾，或单独建立 `docs/paper_draft.md`。下述页码与图表编号以当前论文草稿为准。

**论文题目**：*Hierarchical Access Control for Image-in-Image Steganography via Super-Resolution*（基于超分辨率的图中图隐写层级访问控制）

**核心思想**

- 通过 **超分** 这一“必须放大才完整”的过程，把不同级别的秘密 **绑定到不同分辨率层级**；同一张 HR stego 可按权限分发，用户只能在授权分辨率上恢复对应秘密。
- 框架由 **PIN（多轮嵌入）+ IAM（重要性图）+ DEN（分发/提取）** 组成；支持 **单层** 与 **跨层** 提取（高层 stego 下采样后提取低层秘密）。
  - 参见 **Figure 1/2**（第 10–11 页）。

**关键实验结果（选摘）**

- **两轮隐藏（2 secrets）**：在 DIV2K/Set5/Set14 上，恢复质量与容器保真均取得高分；例如 **Recovered1（DIV2K）≈ 43.62 dB / 0.992，Recovered2（DIV2K）≈ 41.85 dB / 0.987**；详见 **表 1**（第 21 页）。
- **任意尺度 SR 能力**：在隐写与 ×1.5/×2/×2.25 放大同时进行时，相比传统 Bicubic 仍显著占优（见 **表 2**，第 22 页）。
- **三轮隐藏（3 secrets）**：平均 stego PSNR 从两轮的 35.79 dB 降至 33.56 dB，恢复平均 PSNR 从 46.23 dB 降至 40.20 dB，但整体可视质量仍保持一致（**表 3** 第 23 页；**图 4** 第 22 页）。
- **层级安全性验证**：
  1. **低分辨率上采样越权解码失败**：从 `↑Y1` 恢复高层秘密得到近似噪声（**PSNR ≈ 10.99 dB, SSIM ≈ 0.314**），而从正确的 `Yn` 可高保真恢复（**PSNR ≈ 41.50 dB, SSIM ≈ 0.997**）。见 **图 6**（第 25 页）。
  2. **下采样—上采样篡改** 破坏隐写信号：`Y′n = ↑↓(Yn)` 与 `Yn` 视觉相近，但解码失败（**PSNR ≈ 11.04 dB, SSIM ≈ 0.322**）；见 **图 7**（第 27 页）。

**实现提示**

- 任意尺度与可逆性来自 AIDN 的设计；在本项目中，SR 模块被复用/改造为 **尺度感知重采样**，与隐写联合优化（参见 AIDN 文档）。
- 多图隐藏/重要性图思想来源于 DeepMIH，并在本架构中以 **PIN + IAM** 的形式落地（参见 DeepMIH 仓库）。

