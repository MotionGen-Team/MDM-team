# 团队使用说明

本仓库用于团队协作。优先看这份文档，再看官方 `README.md`。

## 基本约定

- GitHub 仓库主要同步代码、团队文档、整理后的日志和必要的可视化脚本。
- 大体积原始数据仍然从团队网盘获取，不直接放到 GitHub。
- 训练输出、生成结果、可视化视频的目录命名统一使用“代码改动版本名”。
- 以后最新代码统一以 `main` 为准；我这边更新代码后会同步到 `main`，队友需要更新时直接从 `main` 拉。
- 个人开发如果需要单独提交，仍建议从 `main` 新建自己的功能分支，不要共用别人的分支。

## 分支协作规则

当前默认同步主线为 `main`。

每次开始工作前，先切到 `main` 并拉最新代码：

```powershell
git fetch origin
git switch main
git pull origin main
```

如果只是拉取我更新后的最新代码，停在 `main` 即可。

如果你需要自己开发并 push，建议再从最新 `main` 新建自己的功能分支：

例如：

- 负责 `conv`，就在 `feature/conv` 上开发和 push
- 负责环境，就在 `feature/A-env` 上开发和 push
- 负责数据，就在 `feature/B-data` 上开发和 push

典型流程：

```powershell
git fetch origin
git switch main
git pull origin main
git switch -c feature/B-data
```

如果你已经在自己的分支上，并且本地已经跟踪远端对应分支：

```powershell
git pull
git add 你改的文件
git commit -m "写你的修改说明"
git push
```

## 从进目录到最后 push 的常用命令

### 第一次在本地关联仓库

如果你本地这个目录还没有关联 GitHub 远端：

```powershell
cd 你的项目目录
git init
git remote add origin https://github.com/MotionGen-Team/MDM-team.git
git remote -v
git fetch origin
git branch -a
```

### 切到你自己的功能分支

先确保 `main` 是最新：

```powershell
git fetch origin
git switch main
git pull origin main
```

例如：

```powershell
git switch -c feature/B-data
```

或者：

```powershell
git switch -c feature/conv
```

新建分支已经基于最新 `main`，这一步之后直接改代码即可。

如果你切回的是已经存在的旧功能分支，先回到 `main` 拉最新，再自己决定是否把 `main` 合到功能分支里：

```powershell
git switch main
git pull origin main
git switch 你的功能分支名
git merge main
```

### 修改后提交全部改动

```powershell
git status
git add .
git commit -m "写你的提交说明"
git push
```

### 修改后只提交指定文件

```powershell
git status
git add 路径\文件名
git commit -m "写你的提交说明"
git push
```

例如：

```powershell
git add model\mdm.py
git commit -m "Update mdm logic"
git push
```

多个文件：

```powershell
git add 文件1 文件2 文件3
git commit -m "写你的提交说明"
git push
```

### 第一次 push 本地新分支

如果 `git push` 提示没有上游分支：

```powershell
git push -u origin 当前分支名
```

例如：

```powershell
git push -u origin feature/B-data
```

### 每次开始工作前建议先做

```powershell
git status -sb
git pull
```

## 不直接从 GitHub 获取的内容

以下内容仍然需要从团队网盘下载并放到本地：

- `dataset/humanml/`
- `dataset/humanml（官方）/`
- `dataset/t2m_train.npy`
- `dataset/t2m_test.npy`
- `body_models/`

## 本地目录结构

建议本地目录至少包含这些位置：

```text
motion-diffusion-model-main/
├── article/
├── checkpoints/
├── dataset/
├── generate_npy/
├── logging/
├── videos/
├── 方案管理/
└── ...
```

## 环境配置

如果团队已经有现成环境，直接激活：

```powershell
conda activate mdm_clean
```

如果没有环境，就根据仓库里的 `environment.yml` 创建（该文件中的环境名为 `mdm_clean`）：

```powershell
conda env create -f environment.yml
conda activate mdm_clean
```

当前仓库里的 `environment.yml` 已对齐本机 `C:\anoconda\envs\mdm_clean`，核心版本为 `Python 3.8.20`、`PyTorch 2.4.1`、`CUDA 12.1`。

## 训练、生成、可视化的统一约定

### 1. 训练输出

训练时，模型输出统一保存到：

```text
checkpoints/代码改动版本名
```

例如：

```powershell
python -m train.train_mdm --save_dir checkpoints\feature_conv_v1 --dataset humanml --batch_size 128
```

### 2. 生成结果

生成结果统一保存到：

```text
generate_npy/代码改动版本名
```

第一次需要手动创建目录，例如：

```powershell
mkdir generate_npy\feature_conv_v1
```

然后显式指定输出目录：

```powershell
python -m sample.generate --model_path checkpoints\feature_conv_v1\model000021448.pt --output_dir generate_npy\feature_conv_v1 --text_prompt "a person walks forward"
```

常见输出包括：

- `results.npy`
- `results.txt`
- `results_len.txt`
- `refined.npy`（如果做了后处理）

### 3. 可视化输出

视频统一放到：

```text
videos/代码改动版本名
```

第一次也需要手动建目录，例如：

```powershell
mkdir videos\feature_conv_v1
```

## 常用路径在哪些文件里改

### 训练保存路径

- [train_mdm.py](/D:/MDM/motion-diffusion-model-main/train/train_mdm.py)
  通过命令行参数 `--save_dir` 控制训练输出目录。

- [parser_util.py](/D:/MDM/motion-diffusion-model-main/utils/parser_util.py)
  定义了 `--save_dir`、`--model_path`、`--output_dir` 等常用参数。

### 生成读取和保存路径

- [generate.py](/D:/MDM/motion-diffusion-model-main/sample/generate.py)
  读取 `--model_path`，并把结果写到 `--output_dir`。

- [edit.py](/D:/MDM/motion-diffusion-model-main/sample/edit.py)
  编辑模式同样通过 `--model_path` 和 `--output_dir` 控制输入输出。

### 可视化路径

- [m_result_visualize.py](/D:/MDM/motion-diffusion-model-main/visualize/m_result_visualize.py)
- [o_result_visualize.py](/D:/MDM/motion-diffusion-model-main/visualize/o_result_visualize.py)

## 可视化脚本使用约定

当前团队优先使用：

- [m_result_visualize.py](/D:/MDM/motion-diffusion-model-main/visualize/m_result_visualize.py)
- [o_result_visualize.py](/D:/MDM/motion-diffusion-model-main/visualize/o_result_visualize.py)

这两个脚本现在都按原始关节坐标格式读取：

```text
(T, 22, 3)
```

官方可视化文件 [plot_script.py](/D:/MDM/motion-diffusion-model-main/data_loaders/humanml/utils/plot_script.py) 也已经修过，当前环境下可以正常出图，不再是之前的空白帧问题。

## 各角色需要 push 的内容

### 跑模型的人

- 训练结果保存到 `checkpoints/版本名`
- 生成结果保存到 `generate_npy/版本名`
- 如果需要给可视化同学用结果，至少保证对应目录里的 `results.npy` 或 `refined.npy` 已整理好

### 做可视化的人

- 视频统一放到 `videos/版本名`
- 优先使用 `visualize/m_result_visualize.py`

### 找文献和整理日志的人

- 文献整理统一放到 `article/`
- 日志整理统一放到 `logging/`

## 方案一接入说明（main）

这一节只针对方案一目前已经落地的改动。

### 1. 先拉取 main

以后方案一相关代码和文档也统一从 `main` 拉。

如果你本地已经有 `main`：

```powershell
git switch main
git pull origin main
```

如果你本地还没有 `main`：

```powershell
git fetch origin
git switch -c main origin/main
git pull origin main
```

### 2. 方案一相关文档现在放哪里

方案一相关文档现在统一放在：

```text
方案管理/方案一/
```

当前这几个文件都应该从仓库里直接拉下来：

```text
方案管理/方案一/三层结构图.png
方案管理/方案一/方案一_pipeline整理版.md
方案管理/方案一/方案一_接口文档.md
方案管理/方案一/方案一_实现方案.md
方案管理/方案一/代码书写规范（参照方案一当前实现风格）.md
```

也就是说，队友后面不要再去仓库根目录找这些方案一文档。

### 3. 方案一这次新增和改过的关键文件

#### 方案一模型目录

这些文件拉完 `main` 后会自动出现在本地，不需要手动新建：

```text
model/plan_one/
    __init__.py
    mdm_plan_one.py
    shared_embedding/
        __init__.py
        structure_adapter.py
        shared_embedding.py
    local_branch/
        __init__.py
        body_groups.py
        group_pooling.py
        multi_scale_temporal.py
        group_attention.py
        local_branch.py
    global_branch/
        __init__.py
        group_summary_builder.py
        summary_fusion.py
        temporal_gqa.py
        transformer_block.py
        global_branch.py
    tcn_refine/
        __init__.py
        fusion.py
        heads.py
        residual_tcn.py
```

#### 已改的入口文件

```text
utils/parser_util.py
utils/model_util.py
```

#### 已改的数据语义与可视化相关文件

```text
data_loaders/humanml_custom_utils.py
data_loaders/humanml/scripts/custom_motion_process.py
data_loaders/humanml/utils/plot_script.py
visualize/m_result_visualize.py
visualize/o_result_visualize.py
sample/generate.py
sample/edit.py
train/training_loop.py
diffusion/gaussian_diffusion.py
```

### 4. 这些文件放在哪

如果你是直接 `git pull`，这些文件会自动进入正确目录，不需要手动搬。

仍然不从 GitHub 拉、而是从网盘放到本地的，只有这些数据和模型资源：

```text
dataset/humanml/
dataset/humanml（官方）/
dataset/t2m_train.npy
dataset/t2m_test.npy
body_models/
```

### 5. 现在怎么训练方案一

方案一已经接入模型工厂，训练时直接指定：

```text
--arch plan_one
```

最基本的训练命令示例：

```powershell
python -m train.train_mdm ^
  --save_dir checkpoints\plan_one_smoke ^
  --dataset humanml ^
  --batch_size 128 ^
  --arch plan_one ^
  --text_encoder_type clip
```

如果是正式训练，建议显式命名版本目录：

```powershell
python -m train.train_mdm ^
  --save_dir checkpoints\plan_one_v1 ^
  --dataset humanml ^
  --batch_size 128 ^
  --arch plan_one ^
  --text_encoder_type clip
```

当前说明：

- 主支持 `humanml` / `kit`
- 当前正式支持的是 `hml_vec`
- 当前优先走 `text-to-motion`

### 6. 现在怎么生成

生成时同样直接指定：

```text
--arch plan_one
```

示例：

```powershell
python -m sample.generate ^
  --model_path checkpoints\plan_one_v1\model000000000.pt ^
  --output_dir generate_npy\plan_one_v1 ^
  --dataset humanml ^
  --arch plan_one ^
  --text_encoder_type clip ^
  --text_prompt "a person walks forward"
```

生成后主要看：

```text
generate_npy/plan_one_v1/results.npy
generate_npy/plan_one_v1/results.txt
generate_npy/plan_one_v1/results_len.txt
```

### 7. 现在怎么可视化

#### 官方可视化

官方可视化现在已经修过，输入按：

```text
(T, 22, 3)
```

即可正常出图。

#### 团队当前推荐可视化

我们当前仍优先使用：

- [m_result_visualize.py](/D:/MDM/motion-diffusion-model-main/visualize/m_result_visualize.py)
- [o_result_visualize.py](/D:/MDM/motion-diffusion-model-main/visualize/o_result_visualize.py)

这两个脚本现在也统一按：

```text
(T, 22, 3)
```

读取，不再需要旧版那种 `(22, 3, T)` 转置。

如果要可视化生成结果，先确认输入文件是：

```text
generate_npy/某个版本/results.npy
```

或后处理之后的：

```text
generate_npy/某个版本/refined.npy
```

视频统一输出到：

```text
videos/某个版本/
```

### 8. 方案一当前最重要的限制

当前方案一已经打通了最小训练链和最小采样链，但还没有完成正式长训练验证。

所以团队成员拉下去之后，建议优先做：

1. 确认 `--arch plan_one` 能正常建模
2. 跑一个小规模训练或最小采样
3. 再决定是否继续调结构

不要一上来就同时大改 `Local / Global / Refine`，否则排错成本太高。

## Git 使用提醒

- 最新代码以 `main` 为准，队友更新代码时先从 `main` 拉
- 如果只是使用最新代码，不需要切到旧的 `feature/conv`
- 自己要开发并 push 时，建议从最新 `main` 新建对应功能分支
- push 之前先确认已经同步最新 `main`
- 如果只改了单个文件，就只 `git add` 那个文件，不要默认 `git add .`
