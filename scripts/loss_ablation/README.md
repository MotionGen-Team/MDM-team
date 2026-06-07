# Loss 消融训练脚本说明

这个目录用于并行跑 5 组 contact geometry loss 权重实验。

使用前先登录服务器，进入服务器上的项目根目录，并激活项目 Python/Conda 环境。

项目路径以服务器实际路径为准，例如：

```bash
cd /path/to/motion-diffusion-model-main
conda activate <your_env_name>
```

下面命令都假设已经在项目根目录。

## 服务器运行命令

V1：

```bash
bash scripts/loss_ablation/run_v1_base_cont_smooth.sh
```

V2：

```bash
bash scripts/loss_ablation/run_v2_strong_continuity.sh
```

V3：

```bash
bash scripts/loss_ablation/run_v3_strong_smooth.sh
```

V4：

```bash
bash scripts/loss_ablation/run_v4_no_smooth.sh
```

V5：

```bash
bash scripts/loss_ablation/run_v5_weak_slide_strong_contact.sh
```

## 本地 Windows 备用命令

```powershell
cd E:\MDM\motion-diffusion-model-main
```

V1：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\loss_ablation\run_v1_base_cont_smooth.ps1
```

V2：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\loss_ablation\run_v2_strong_continuity.ps1
```

V3：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\loss_ablation\run_v3_strong_smooth.ps1
```

V4：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\loss_ablation\run_v4_no_smooth.ps1
```

V5：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\loss_ablation\run_v5_weak_slide_strong_contact.ps1
```

## 版本含义

```text
V1 base_cont_smooth:
  平衡版本
  slide=0.01 height=0.01 vertical=0.01 continuity=0.005 smooth=0.001

V2 strong_continuity:
  加强接触连续性，重点压 no_contact_ratio
  slide=0.01 height=0.01 vertical=0.01 continuity=0.01 smooth=0.001

V3 strong_smooth:
  加强接触帧脚部平滑，重点压 jerk
  slide=0.01 height=0.01 vertical=0.01 continuity=0.005 smooth=0.002

V4 no_smooth:
  去掉 smooth，判断 smooth 是否必要
  slide=0.01 height=0.01 vertical=0.01 continuity=0.005 smooth=0

V5 weak_slide_strong_contact:
  降低 slide、加强 contact，观察是否减少脚被钉住或动作变钝
  slide=0.005 height=0.01 vertical=0.01 continuity=0.01 smooth=0.001
```

## 输出目录

```text
checkpoints/loss_v1_base_cont_smooth
checkpoints/loss_v2_strong_continuity
checkpoints/loss_v3_strong_smooth
checkpoints/loss_v4_no_smooth
checkpoints/loss_v5_weak_slide_strong_contact
```

## 注意事项

```text
1. 不要修改脚本参数。
2. 不要加 --overwrite。
3. 如果 save_dir 已存在，先反馈，不要删除目录。
4. 如果多人共用多卡服务器，需要确认各自使用的 GPU，不要占同一张卡。
5. 出现报错时保留日志和 checkpoint 目录。
```
