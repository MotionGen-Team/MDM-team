@echo off
chcp 65001
cls

echo ========================================
echo 多尺度TCN实验 - 自动生成视频
echo ========================================
echo.

REM 设置变量
set VARIANTS=baseline 3-7-3-dil2 3-5-5-dil2
set NUM_STEPS=1000
set NUM_SAMPLES=3
set TEXT_PROMPT=the person walked forward and is picking up his toolbox.

REM 创建videos目录
if not exist videos mkdir videos

echo 将运行以下变体:
for %%v in (%VARIANTS%) do (
    echo   - %%v
)
echo.
echo 训练步数: %NUM_STEPS%
echo 生成样本数: %NUM_SAMPLES%
echo 文本提示: %TEXT_PROMPT%
echo.

REM 为每个变体运行
for %%v in (%VARIANTS%) do (
    echo.
    echo ========================================
    echo 处理变体: %%v
    echo ========================================
    
    set SAVE_DIR=save\ms_tcn_%%v
    set VIDEO_DIR=videos\%%v
    
    REM 检查是否已有模型
    if exist !SAVE_DIR! (
        for %%f in (!SAVE_DIR!\model*.pt) do (
            set MODEL_PATH=%%f
            goto :found_model_%%v
        )
    )
    
    :train_%%v
    echo [1/2] 训练 %%v 模型...
    python -m train.train_mdm --save_dir !SAVE_DIR! --dataset humanml --num_steps %NUM_STEPS% --use_temporal_tcn --ms_tcn_variant %%v --save_interval 500
    
    :found_model_%%v
    if "!MODEL_PATH!"=="" (
        echo 未找到模型，开始训练...
        goto :train_%%v
    )
    
    echo 使用模型: !MODEL_PATH!
    
    REM 生成视频
    echo [2/2] 生成 %%v 视频...
    python -m sample.generate --model_path !MODEL_PATH! --text_prompt "%TEXT_PROMPT%" --num_samples %NUM_SAMPLES% --num_repetitions 1 --output_dir !VIDEO_DIR!
    
    if exist !VIDEO_DIR! (
        echo [OK] %%v 视频生成成功
    ) else (
        echo [FAIL] %%v 视频生成失败
    )
)

echo.
echo ========================================
echo 实验完成!
echo ========================================
echo 视频保存在 videos\ 目录下
echo.

pause
