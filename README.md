# MDM-team: Human Motion Diffusion Model

本仓库基于 [Human Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model) 进行二次开发，主要面向 text-to-motion 任务中的人体动作生成、结构化动作表示和训练诊断分析。

当前主线在原始 MDM 的 `trans_enc` / `trans_dec` / `gru` 架构之外，新增了 `plan_one` 架构：将 HumanML3D / KIT 的动作向量恢复为结构化人体 token，分别经过局部分支、全局分支和 TCN refinement，再输出动作预测结果。

## 主要内容

- 基于 MDM 的文本到动作生成训练、采样和可视化流程
- 新增 `--arch plan_one`，支持结构化人体动作建模
- 按身体部位构建局部分支，显式建模 torso、arms、legs 等局部结构
- 使用 group summary 和 grouped-query attention 建模全局时序信息
- 使用 residual TCN refinement 对 base prediction 做残差修正
- 训练过程中输出 loss、gate、h_struct norm/variance 等诊断 CSV
- 整理了团队协作说明、方案文档和实验日志

## 当前进度

当前工作重点是优化 `plan_one` 架构的训练稳定性和收敛速度。针对实验中出现的 loss 收敛较慢问题，已经加入了 gate、h_struct norm/variance、不同结构阶段统计等诊断输出，并围绕结构注入方式开展消融实验。

现阶段正在对细节结构的选择进行对比试验，包括 root/global 信息注入范围、帧级位置编码注入范围、文本/时间条件注入范围，以及不同身体区域结构 token 对整体生成质量和 loss 曲线的影响。相关实验结果会持续整理到 `logging/` 和 `方案管理/` 文档中。

## 环境要求

推荐环境：

- Python 3.8
- PyTorch 2.4.1
- CUDA 12.1
- Conda / Miniconda
- ffmpeg

可以根据仓库中的环境文件配置 Conda 环境。Windows / 本地开发环境优先参考 `environment.yml`：

```powershell
conda env create -f environment.yml
conda activate mdm_clean
```

Linux / Unix 环境可以参考 `environment_unix.yml`：

```bash
conda env create -f environment_unix.yml
conda activate mdm_clean
```

如果需要使用 CLIP 文本编码器，需要从 OpenAI CLIP 的 GitHub 仓库安装：

```bash
pip install git+https://github.com/openai/CLIP.git
```

CLIP 仓库地址：

- https://github.com/openai/CLIP

## 数据准备

训练和采样需要 HumanML3D / KIT 相关数据。由于数据集、SMPL 资源和训练输出体积较大，以下内容不直接放入 GitHub：

```text
dataset/humanml/
dataset/humanml（官方）/
dataset/t2m_train.npy
dataset/t2m_test.npy
body_models/
checkpoints/
save/
output/
videos/
generate_npy/
```

常用数据目录：

```text
dataset/
├── humanml/
├── humanml_opt.txt
├── kit_opt.txt
├── t2m_mean.npy
├── t2m_std.npy
├── kit_mean.npy
└── kit_std.npy
```

HumanML3D 原始项目可参考：

- https://github.com/EricGuo5513/HumanML3D

## 快速训练

原始 MDM baseline 示例：

```powershell
python -m train.train_mdm ^
  --save_dir checkpoints\baseline_trans_enc ^
  --dataset humanml ^
  --batch_size 128 ^
  --arch trans_enc ^
  --text_encoder_type clip
```

Plan One 示例：

```powershell
python -m train.train_mdm ^
  --save_dir checkpoints\plan_one_v1 ^
  --dataset humanml ^
  --batch_size 128 ^
  --arch plan_one ^
  --text_encoder_type clip
```

Plan One 消融参数示例：

```powershell
python -m train.train_mdm ^
  --save_dir checkpoints\plan_one_ablate_root_torso ^
  --dataset humanml ^
  --batch_size 128 ^
  --arch plan_one ^
  --plan_one_root_mode torso_only ^
  --plan_one_struct_pos_mode all_joints ^
  --plan_one_struct_cond_mode all_joints
```

可选的 Plan One 结构注入模式：

```text
--plan_one_root_mode        all_joints | torso_only | none
--plan_one_struct_pos_mode  all_joints | torso_only | none
--plan_one_struct_cond_mode all_joints | torso_only | none
--plan_one_local_mode       full | zero
--plan_one_refine_mode      full | off
```

## 生成动作

使用训练好的 checkpoint 生成动作：

```powershell
python -m sample.generate ^
  --model_path checkpoints\plan_one_v1\model000000000.pt ^
  --output_dir generate_npy\plan_one_v1 ^
  --dataset humanml ^
  --arch plan_one ^
  --text_encoder_type clip ^
  --text_prompt "a person walks forward"
```

生成结果通常包含：

```text
generate_npy/plan_one_v1/results.npy
generate_npy/plan_one_v1/results.txt
generate_npy/plan_one_v1/results_len.txt
```

## 可视化

仓库保留了原始 MDM 的可视化流程，同时新增/整理了团队使用脚本：

```text
visualize/m_result_visualize.py
visualize/o_result_visualize.py
data_loaders/humanml/utils/plot_script.py
```

本地生成的视频建议输出到：

```text
videos/
```

该目录已按本地产物处理，不建议提交到 GitHub。

## Plan One 架构概览

`plan_one` 的主要流程：

```text
motion input
  -> shared embedding
  -> local branch
  -> global branch
  -> fusion
  -> base head
  -> residual TCN refinement
  -> motion output
```

核心目录：

```text
model/plan_one/
├── mdm_plan_one.py
├── shared_embedding/
├── local_branch/
├── global_branch/
└── tcn_refine/
```

关键模块：

- `shared_embedding`: 将 motion tensor 转换为结构化 token 和全局 token
- `local_branch`: 按身体部位处理局部结构信息
- `global_branch`: 从结构 token 构造 group summary 并建模全局时序关系
- `tcn_refine`: 输出 base prediction 并通过 residual TCN 做修正
- `training_loop`: 记录 loss、gate 和 h_struct 统计信息

## 训练诊断输出

训练时会在 `output/` 下记录若干 CSV，用于分析 loss、门控值和结构 token 的分布变化：

```text
output/loss_steps_plan_one.csv
output/plan_one_gate.csv
output/plan_one_hstruct_norm_stages.csv
```

其中 `plan_one_hstruct_norm_stages.csv` 会记录不同阶段的结构特征统计，例如：

```text
x_struct_raw
x_struct_mixed
x_struct_norm
h_struct_proj
h_struct_pre_norm
h_struct_pre_inject
h_struct_post_inject
```

## 项目结构

```text
.
├── data_loaders/        # 数据加载、HumanML/KIT 处理工具
├── diffusion/           # diffusion 训练和采样逻辑
├── model/               # 原始 MDM 和 Plan One 模型
├── train/               # 训练入口和训练循环
├── sample/              # 采样和生成入口
├── visualize/           # 动作可视化脚本
├── dataset/             # 数据集配置和本地数据目录
├── checkpoints/         # 本地 checkpoint，不提交
├── output/              # 本地训练诊断输出，不提交
├── videos/              # 本地可视化视频，不提交
├── logging/             # 团队实验日志
├── 方案管理/             # 方案设计、接口和实现文档
├── environment.yml
└── environment_unix.yml
```

## 团队协作

团队协作、分支命名、数据放置和常用 Git 命令见：

```text
TEAM_SETUP.md
```

常规开发建议：

```powershell
git status -sb
git pull
git add <files>
git commit -m "Update xxx"
git push
```

不要提交大体积数据集、checkpoint、训练输出、生成视频和临时缓存。

## 参考项目与引用

本项目基于 MDM。若使用原始 MDM 代码或结果，请引用：

```bibtex
@inproceedings{
tevet2023human,
title={Human Motion Diffusion Model},
author={Guy Tevet and Sigal Raab and Brian Gordon and Yoni Shafir and Daniel Cohen-or and Amit Haim Bermano},
booktitle={The Eleventh International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=SJ1kSyO2jwu}
}
```

相关链接：

- MDM paper: https://arxiv.org/abs/2209.14916
- Original MDM repository: https://github.com/GuyTevet/motion-diffusion-model
- HumanML3D: https://github.com/EricGuo5513/HumanML3D

## License

本仓库继承原始 MDM 项目的开源协议。具体请查看 `LICENSE` 文件。
