# 团队使用说明

本仓库用于团队协作，优先看这份文档，再看官方 `README.md`。

## 基本约定

- GitHub 仓库主要同步代码、团队文档、整理后的日志和必要的可视化脚本。
- 大体积原始数据仍然从团队网盘获取，不直接放到 GitHub。
- 训练输出、生成结果、可视化视频的目录命名统一使用“代码改动版本名”。
- 每个人只在自己的功能分支上开发和 push，不要共用别人的分支，也不要直接往 `main` push。

## 分支协作规则

每个人按自己的身份或任务，在对应分支工作。

例如：

- 你负责 `conv`，就在 `feature/conv` 上开发和 push
- 你负责环境，就在 `feature/A-env` 上开发和 push
- 你负责数据，就在 `feature/B-data` 上开发和 push

也就是说，如果你的身份是 `B`，或者你负责的是 `B-data` 这部分工作，就只在：

```text
feature/B-data
```

这个分支提交代码，不要提交到别人的分支，也不要提交到 `main`。

典型流程：

```powershell
git fetch origin
git switch -c feature/B-data origin/feature/B-data
git pull
```

改完后：

```powershell
git add 你改的文件
git commit -m "写你的修改说明"
git push
```

如果你已经在自己的分支上，并且本地已经跟踪远端对应分支，那就直接：

```powershell
git pull
git add 你改的文件
git commit -m "写你的修改说明"
git push
```

## 从进目录到最后 push 的完整命令

### 第一次在本地关联仓库

如果你本地这个目录还没有关联 GitHub 远端，先执行：

```powershell
cd 你的项目目录
git init
git remote add origin https://github.com/MotionGen-Team/MDM-team.git
git remote -v
git fetch origin
git branch -a
```

### 切到你自己的功能分支

如果你负责 `B-data`，就切到：

```powershell
git switch -c feature/B-data origin/feature/B-data
```

如果你负责 `A-env`，就切到：

```powershell
git switch -c feature/A-env origin/feature/A-env
```

如果你负责 `conv`，就切到：

```powershell
git switch -c feature/conv origin/feature/conv
```

切过去后先同步：

```powershell
git pull
```

### 修改后提交整个改动

如果你确认这次要提交当前所有改动：

```powershell
git status
git add .
git commit -m "写你的提交说明"
git push
```

### 修改后只提交某个文件

如果你只想提交自己改过的某几个文件，不要用 `git add .`，而是只加指定文件：

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

如果是多个文件：

```powershell
git add 文件1 文件2 文件3
git commit -m "写你的提交说明"
git push
```

### 第一次 push 本地新分支

如果 `git push` 提示没有上游分支，用：

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
├─ article/
├─ checkpoints/
├─ dataset/
├─ generate_npy/
├─ logging/
├─ videos/
└─ ...
```

## 环境配置

如果团队已经有现成环境，直接激活：

```powershell
conda activate mdm_clean
```

如果没有环境，就根据仓库里的 `environment.yml` 创建：

```powershell
conda env create -f environment.yml
conda activate mdm_clean
```

## 训练、生成、可视化的统一约定

### 1. 训练输出

训练时，模型输出统一保存到：

```text
checkpoints/代码改动版本名/
```

里面保存的是训练得到的 `pt` 文件，例如：

```powershell
python -m train.train_mdm --save_dir checkpoints\feature_conv_v1 --dataset humanml
```

这样训练结果就会进入：

```text
checkpoints/feature_conv_v1/
```

### 2. 生成结果

跑 `generate` 时，生成结果统一保存到：

```text
generate_npy/代码改动版本名/
```

第一次需要手动创建目录，例如：

```powershell
mkdir generate_npy\feature_conv_v1
```

然后生成时显式指定输出目录：

```powershell
python -m sample.generate --model_path checkpoints\feature_conv_v1\model000021448.pt --output_dir generate_npy\feature_conv_v1 --text_prompt "a person walks forward"
```

常见输出文件包括：

- `results.npy`
- `results.txt`
- `results_len.txt`

如果还做了后处理，也会有例如：

- `refined.npy`

### 3. 可视化输出

做可视化的人统一在：

```text
videos/代码改动版本名/
```

里保存视频。第一次也需要手动建目录，例如：

```powershell
mkdir videos\feature_conv_v1
```

## 常用路径在哪些文件里改

### 训练保存路径

- [train_mdm.py](D:/MDM/motion-diffusion-model-main/train/train_mdm.py)
  这里通过命令行参数 `--save_dir` 决定训练输出目录。
- [parser_util.py](D:/MDM/motion-diffusion-model-main/utils/parser_util.py)
  这里定义了 `--save_dir`、`--model_path`、`--output_dir` 等常用参数。

### 生成读取和保存路径

- [generate.py](D:/MDM/motion-diffusion-model-main/sample/generate.py)
  这里读取 `--model_path`，并把生成结果写到 `--output_dir`。
  如果不传 `--output_dir`，默认会写到模型目录旁边，所以团队协作时建议总是显式传入 `generate_npy/版本名`。
- [edit.py](D:/MDM/motion-diffusion-model-main/sample/edit.py)
  编辑模式同样通过 `--model_path` 和 `--output_dir` 控制输入输出路径。
- [refine_motion.py](D:/MDM/motion-diffusion-model-main/utils/refine_motion.py)
  这里直接写了读取 `generate_npy/results.npy` 和保存 `refined.npy` 的路径，如果你的目录结构变了，需要在这里一起改。

### 评估日志路径

- [eval_humanml.py](D:/MDM/motion-diffusion-model-main/eval/eval_humanml.py)
  评估日志默认保存在 `args.model_path` 所在目录下，也就是对应 checkpoint 目录里。

### 可视化路径

- [m_result_visualize.py](D:/MDM/motion-diffusion-model-main/visualize/m_result_visualize.py)
  这是我们团队当前推荐使用的可视化脚本。
  它默认读取 `generate_npy/refined.npy`，并输出到 `videos/`。
- [o_result_visualize.py](D:/MDM/motion-diffusion-model-main/visualize/o_result_visualize.py)
  这是官方风格的可视化路径，不建议我们继续用。

## 可视化脚本使用约定

因为官方的关节标号和我们当前使用的不完全一致，所以：

- 不要优先用官方原始可视化流程
- 我们统一使用 [m_result_visualize.py](D:/MDM/motion-diffusion-model-main/visualize/m_result_visualize.py)

如果你改了生成结果目录，记得同时改这个脚本里的输入输出路径。

## 各角色需要 push 的内容

### 跑模型的人

- 训练结果保存在 `checkpoints/版本名/`
- 生成结果保存在 `generate_npy/版本名/`
- 如果需要给可视化同学用结果，至少要保证对应版本目录下的 `results.npy` 或 `refined.npy` 已经整理好

### 做可视化的人

- 视频统一放到 `videos/版本名/`
- 可视化优先使用 `visualize/m_result_visualize.py`

### 找文献和整理日志的人

- 文献整理统一放到 `article/`
- 日志整理统一放到 `logging/`
- 这两类内容每次整理完都直接 push

## Git 使用提醒

- 不要直接往 `main` 推
- 开发代码请在对应功能分支上完成
- push 之前先 `git pull`
- 如果只改了单个文件，就只 `git add` 那个文件，不要默认 `git add .`
