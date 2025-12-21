#!/bin/bash
# 訓練兩種配置的快速啟動腳本
# 使用方式：
#   ./train_both_configs.sh           # 訓練兩種配置
#   ./train_both_configs.sh pretrained # 僅訓練預訓練配置
#   ./train_both_configs.sh paper     # 僅訓練論文配置

set -e  # 遇到錯誤立即退出

# 激活虛擬環境
echo "激活虛擬環境..."
source run_env.sh

# 進入 src 目錄
cd src

# 檢查命令行參數
if [ "$1" == "pretrained" ]; then
    echo "="
    echo "訓練預訓練配置 (r=320)"
    echo "="
    python train_single_config.py --config pretrained_r320

elif [ "$1" == "paper" ]; then
    echo "="
    echo "訓練論文配置 (r=64)"
    echo "="
    python train_single_config.py --config paper_r64

else
    echo "="
    echo "訓練兩種配置"
    echo "="

    echo ""
    echo "步驟 1/2: 訓練論文配置 (r=64)"
    echo "輸出: output/paper_r64/"
    echo ""
    python train_single_config.py --config paper_r64

    echo ""
    echo "步驟 2/2: 訓練預訓練配置 (r=320)"
    echo "輸出: output/pretrained_r320/"
    echo ""
    python train_single_config.py --config pretrained_r320
fi

echo ""
echo "="
echo "訓練完成！"
echo "="
echo ""
echo "查看訓練日誌："
echo "  tensorboard --logdir ../output/"
echo ""
echo "模型位置："
if [ "$1" == "pretrained" ]; then
    echo "  ../output/pretrained_r320/final_model"
elif [ "$1" == "paper" ]; then
    echo "  ../output/paper_r64/final_model"
else
    echo "  ../output/pretrained_r320/final_model"
    echo "  ../output/paper_r64/final_model"
fi
echo ""
