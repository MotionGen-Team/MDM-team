# Contact loss ablation work assignment

## 目标

本轮任务是并行验证 5 组 HumanML foot contact geometry loss 权重。

核心问题：

```text
在保持 foot_sliding 改善的同时，能不能把 no_contact_ratio / step_timing_score / normalized_jerk 拉回来。
```

这 5 组实验只改 loss 权重，不改模型结构，不改数据，不改评测口径。

## 执行要求

每个人只负责 1 个版本：

```text
1. 不修改代码。
2. 不修改训练脚本里的参数。
3. 不加 --overwrite。
4. 不同时跑多个版本。
5. 训练异常时不要自己改代码，直接反馈错误日志。
6. checkpoint 出来后按要求汇报路径和状态。
```

如果 `save_dir` 已经存在，训练会直接报错。这是正常保护机制，不要加 `--overwrite`，先确认是不是之前已经跑过。

## 直接分配命令

把下面对应版本的命令发给对应执行人即可。

注意：先把 `/path/to/motion-diffusion-model-main` 改成服务器上的真实项目路径。

V1：

```bash
cd /path/to/motion-diffusion-model-main
conda activate <your_env_name>
bash scripts/loss_ablation/run_v1_base_cont_smooth.sh
```

V2：

```bash
cd /path/to/motion-diffusion-model-main
conda activate <your_env_name>
bash scripts/loss_ablation/run_v2_strong_continuity.sh
```

V3：

```bash
cd /path/to/motion-diffusion-model-main
conda activate <your_env_name>
bash scripts/loss_ablation/run_v3_strong_smooth.sh
```

V4：

```bash
cd /path/to/motion-diffusion-model-main
conda activate <your_env_name>
bash scripts/loss_ablation/run_v4_no_smooth.sh
```

V5：

```bash
cd /path/to/motion-diffusion-model-main
conda activate <your_env_name>
bash scripts/loss_ablation/run_v5_weak_slide_strong_contact.sh
```

## 环境准备

登录服务器后，进入服务器上的项目根目录：

```bash
cd /path/to/motion-diffusion-model-main
```

激活项目使用的 Python/Conda 环境。

检查当前目录下能看到这些文件：

```text
train/train_mdm.py
scripts/loss_ablation/run_v1_base_cont_smooth.sh
scripts/loss_ablation/run_v2_strong_continuity.sh
scripts/loss_ablation/run_v3_strong_smooth.sh
scripts/loss_ablation/run_v4_no_smooth.sh
scripts/loss_ablation/run_v5_weak_slide_strong_contact.sh
```

## 版本分配

### V1 base_cont_smooth

目的：

```text
平衡滑脚、接触连续性和平滑性，是当前推荐主线。
```

运行命令：

```bash
bash scripts/loss_ablation/run_v1_base_cont_smooth.sh
```

输出目录：

```text
checkpoints/loss_v1_base_cont_smooth
```

权重：

```text
slide=0.01
height=0.01
vertical=0.01
continuity=0.005
smooth=0.001
```

### V2 strong_continuity

目的：

```text
加强接触连续性，重点观察 no_contact_ratio 是否下降。
```

运行命令：

```bash
bash scripts/loss_ablation/run_v2_strong_continuity.sh
```

输出目录：

```text
checkpoints/loss_v2_strong_continuity
```

权重：

```text
slide=0.01
height=0.01
vertical=0.01
continuity=0.01
smooth=0.001
```

### V3 strong_smooth

目的：

```text
加强接触帧脚部平滑，重点观察 normalized_jerk 是否下降。
```

运行命令：

```bash
bash scripts/loss_ablation/run_v3_strong_smooth.sh
```

输出目录：

```text
checkpoints/loss_v3_strong_smooth
```

权重：

```text
slide=0.01
height=0.01
vertical=0.01
continuity=0.005
smooth=0.002
```

### V4 no_smooth

目的：

```text
去掉 smooth，判断 smooth 是否有必要，或者是否带来副作用。
```

运行命令：

```bash
bash scripts/loss_ablation/run_v4_no_smooth.sh
```

输出目录：

```text
checkpoints/loss_v4_no_smooth
```

权重：

```text
slide=0.01
height=0.01
vertical=0.01
continuity=0.005
smooth=0
```

### V5 weak_slide_strong_contact

目的：

```text
降低 slide 权重，加强 contact，观察是否减少脚被钉住或动作变钝的风险。
```

运行命令：

```bash
bash scripts/loss_ablation/run_v5_weak_slide_strong_contact.sh
```

输出目录：

```text
checkpoints/loss_v5_weak_slide_strong_contact
```

权重：

```text
slide=0.005
height=0.01
vertical=0.01
continuity=0.01
smooth=0.001
```

## GPU 设置

默认使用 `--device 0`。

如果每个人是独立服务器，直接运行脚本即可。

如果多人共用同一台多卡服务器，需要先分配 GPU。被分到 GPU N 的人，在对应脚本的 python 命令后面追加：

```text
--device N
```

例如 GPU 1：

```bash
python -m train.train_mdm ... --device 1
```

不要多人同时占用同一张 GPU。

## 检查训练是否正常

启动后应该能看到类似输出：

```text
creating data loader...
creating model and diffusion...
Training...
```

输出目录下应该生成：

```text
args.json
model000000000.pt
opt000000000.pt
```

后续每隔 `50000` step 会保存：

```text
model000050000.pt
opt000050000.pt
model000100000.pt
opt000100000.pt
...
```

如果没有生成 `args.json`，说明训练没有正常启动。

## 汇报节点

每个人在以下节点汇报一次：

```text
启动后
50k checkpoint 出来后
100k checkpoint 出来后
150k checkpoint 出来后
200k checkpoint 出来后
250k checkpoint 出来后
```

如果 150k 之前指标明显坏掉，后续可能会停止该版本。

## 每次汇报格式

直接复制下面模板填写：

```text
版本：
执行人：
机器/GPU：
启动时间：
当前 step：
save_dir：
最新 model：
最新 opt：
是否报错：
错误日志摘要：
备注：
```

示例：

```text
版本：V2 strong_continuity
执行人：张三
机器/GPU：A100-01 / GPU 0
启动时间：2026-06-08 09:30
当前 step：100000
save_dir：checkpoints/loss_v2_strong_continuity
最新 model：checkpoints/loss_v2_strong_continuity/model000100000.pt
最新 opt：checkpoints/loss_v2_strong_continuity/opt000100000.pt
是否报错：否
错误日志摘要：无
备注：训练正常
```

## 异常处理

### save_dir already exists

含义：

```text
该版本目录已经存在，训练被保护机制拦截。
```

处理：

```text
不要加 --overwrite。
把报错和对应 save_dir 发回来确认。
```

### CUDA out of memory

处理：

```text
不要改 batch_size。
先确认是不是 GPU 被别人占用。
把 GPU 使用情况和错误日志发回来。
```

### 找不到数据或模型依赖

处理：

```text
不要改路径。
把完整报错发回来。
```

### 训练中断

处理：

```text
保留 checkpoint 目录。
不要删除文件。
把最后一个 model/opt checkpoint 和错误日志发回来。
```

## 评测交接

训练人员只负责产出 checkpoint。

每个版本至少需要交出：

```text
model000100000.pt
model000150000.pt
model000200000.pt
```

优先比较：

```text
100k 看趋势
150k 做第一轮淘汰
200k 做主比较
250k 只给最有希望的版本继续
```

评测指标由负责人统一跑，不要每个人自己改评测脚本。

重点指标：

```text
overall:
  foot_sliding
  no_contact_ratio
  step_timing_score
  contact_alternation
  normalized_jerk
  local_jerk_mean
  local_jerk_p95
  contact_height_var

locomotion:
  foot_sliding
  no_contact_ratio
  step_timing_score
  normalized_jerk
  local_jerk_p95
```

## 初步淘汰标准

以下情况优先淘汰：

```text
1. foot_sliding 没有明显优于 baseline。
2. no_contact_ratio 比 evalmask_0p01 还高。
3. normalized_jerk 继续明显高于 0.47。
4. locomotion no_contact_ratio 拉不回 0.08 以下。
5. 视频里出现明显脚被钉住、动作变钝或大幅抖动。
```

这些标准只用于快速筛选，不作为最终论文结论。
