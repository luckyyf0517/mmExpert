#!/bin/bash

# HuggingFace ViT模型下载脚本
# 用于CLIP实验所需的预训练模型批量下载
# 不实际运行，仅提供下载命令

set -e  # 遇到错误时退出

echo "🚀 HuggingFace ViT模型下载脚本"
echo "=================================="
echo "此脚本包含下载所有ViT-Small实验所需的预训练模型"
echo "请按需选择要下载的模型组"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 创建下载目录
DOWNLOAD_DIR="./downloaded_models"
mkdir -p "$DOWNLOAD_DIR"

echo "📁 模型将下载到: $DOWNLOAD_DIR"
echo ""

# 高优先级模型 - 核心实验
echo -e "${YELLOW}🔥 高优先级模型 (核心实验)${NC}"
echo "----------------------------------------"

# 1. 标准ViT-Small
echo -e "${GREEN}1. 标准ViT-Small模型:${NC}"
echo "huggingface-cli download timm/vit_small_patch16_224 --local-dir $DOWNLOAD_DIR/vit_small_patch16_224 --include=\"*.bin,*.json,*.txt,*.py\""

# 2. DINOv2 Patch14
echo -e "${GREEN}2. DINOv2 Patch14模型:${NC}"
echo "huggingface-cli download timm/vit_small_patch14_dinov2 --local-dir $DOWNLOAD_DIR/vit_small_patch14_dinov2 --include=\"*.bin,*.json,*.txt,*.py\""

# 3. SigLIP 224
echo -e "${GREEN}3. SigLIP 224分辨率模型:${NC}"
echo "huggingface-cli download timm/vit_small_patch16_siglip_224 --local-dir $DOWNLOAD_DIR/vit_small_patch16_siglip_224 --include=\"*.bin,*.json,*.txt,*.py\""

# 4. Patch14架构
echo -e "${GREEN}4. Patch14架构模型:${NC}"
echo "huggingface-cli download timm/vit_small_patch14_224 --local-dir $DOWNLOAD_DIR/vit_small_patch14_224 --include=\"*.bin,*.json,*.txt,*.py\""

echo ""

# 中优先级模型 - 扩展实验
echo -e "${YELLOW}⚡ 中优先级模型 (扩展实验)${NC}"
echo "----------------------------------------"

# 5. DINOv3 Patch16
echo -e "${BLUE}5. DINOv3 Patch16模型:${NC}"
echo "huggingface-cli download timm/vit_small_patch16_dinov3 --local-dir $DOWNLOAD_DIR/vit_small_patch16_dinov3 --include=\"*.bin,*.json,*.txt,*.py\""

# 6. ImageNet-21K预训练
echo -e "${BLUE}6. ImageNet-21K预训练模型:${NC}"
echo "huggingface-cli download timm/vit_small_patch16_224.augreg_in21k --local-dir $DOWNLOAD_DIR/vit_small_patch16_224.augreg_in21k --include=\"*.bin,*.json,*.txt,*.py\""

# 7. 相对位置编码
echo -e "${BLUE}7. 相对位置编码模型:${NC}"
echo "huggingface-cli download timm/vit_small_patch16_relpos_224 --local-dir $DOWNLOAD_DIR/vit_small_patch16_relpos_224 --include=\"*.bin,*.json,*.txt,*.py\""

echo ""

# 低优先级模型 - 探索性实验
echo -e "${YELLOW}🔍 低优先级模型 (探索性实验)${NC}"
echo "----------------------------------------"

# 8. DINOv3 QKVB
echo -e "${BLUE}8. DINOv3 QKVB变体:${NC}"
echo "huggingface-cli download timm/vit_small_patch16_dinov3_qkvb --local-dir $DOWNLOAD_DIR/vit_small_patch16_dinov3_qkvb --include=\"*.bin,*.json,*.txt,*.py\""

# 9. SigLIP 384分辨率
echo -e "${BLUE}9. SigLIP 384分辨率模型:${NC}"
echo "huggingface-cli download timm/vit_small_patch16_siglip_384 --local-dir $DOWNLOAD_DIR/vit_small_patch16_siglip_384 --include=\"*.bin,*.json,*.txt,*.py\""

# 10. ImageNet-21K + ImageNet-1K微调
echo -e "${BLUE}10. IN21K + IN1K微调模型:${NC}"
echo "huggingface-cli download timm/vit_small_patch16_224.augreg_in21k_ft_in1k --local-dir $DOWNLOAD_DIR/vit_small_patch16_224.augreg_in21k_ft_in1k --include=\"*.bin,*.json,*.txt,*.py\""

# 11. SAM分割任务预训练
echo -e "${BLUE}11. SAM分割任务预训练:${NC}"
echo "huggingface-cli download timm/vit_small_patch16_sam_224 --local-dir $DOWNLOAD_DIR/vit_small_patch16_sam_224 --include=\"*.bin,*.json,*.txt,*.py\""

# 12. ViT-Small+更大容量
echo -e "${BLUE}12. ViT-Small+更大容量:${NC}"
echo "huggingface-cli download timm/vit_small_plus_patch16_224 --local-dir $DOWNLOAD_DIR/vit_small_plus_patch16_224 --include=\"*.bin,*.json,*.txt,*.py\""

# 13. ResNet-26 stem ViT
echo -e "${BLUE}13. ResNet-26 stem ViT:${NC}"
echo "huggingface-cli download timm/vit_small_r26_s32_224 --local-dir $DOWNLOAD_DIR/vit_small_r26_s32_224 --include=\"*.bin,*.json,*.txt,*.py\""

# 14. 旋转位置编码 (RoPE)
echo -e "${BLUE}14. 旋转位置编码模型:${NC}"
echo "huggingface-cli download timm/vit_small_patch16_rope_224 --local-dir $DOWNLOAD_DIR/vit_small_patch16_rope_224 --include=\"*.bin,*.json,*.txt,*.py\""

# 15. 空间相对位置编码
echo -e "${BLUE}15. 空间相对位置编码:${NC}"
echo "huggingface-cli download timm/vit_small_srelpos_small_patch16_224 --local-dir $DOWNLOAD_DIR/vit_small_srelpos_small_patch16_224 --include=\"*.bin,*.json,*.txt,*.py\""

echo ""

# 使用说明
echo -e "${GREEN}📖 使用说明:${NC}"
echo "1. 选择性复制以下命令执行下载"
echo "2. 或者运行脚本中的特定函数下载指定组"
echo "3. 确保有足够磁盘空间 (预计3-6GB)"
echo "4. 建议在稳定网络环境下执行"
echo ""

# Python脚本下载函数
download_python() {
    echo "使用Python脚本下载模型..."

cat << 'EOF' > download_models.py
import os
from huggingface_hub import hf_hub_download
import torch

# 模型列表 (按优先级排序)
HIGH_PRIORITY = [
    "timm/vit_small_patch16_224",
    "timm/vit_small_patch14_dinov2",
    "timm/vit_small_patch16_siglip_224",
    "timm/vit_small_patch14_224"
]

MEDIUM_PRIORITY = [
    "timm/vit_small_patch16_dinov3",
    "timm/vit_small_patch16_224.augreg_in21k",
    "timm/vit_small_patch16_relpos_224"
]

LOW_PRIORITY = [
    "timm/vit_small_patch16_dinov3_qkvb",
    "timm/vit_small_patch16_siglip_384",
    "timm/vit_small_patch16_224.augreg_in21k_ft_in1k",
    "timm/vit_small_patch16_sam_224",
    "timm/vit_small_plus_patch16_224",
    "timm/vit_small_r26_s32_224",
    "timm/vit_small_patch16_rope_224",
    "timm/vit_small_srelpos_small_patch16_224"
]

def download_models(model_list, priority_name):
    print(f"\n🔥 下载 {priority_name} 优先级模型...")
    download_dir = f"downloaded_models/{priority_name.lower().replace(' ', '_')}"
    os.makedirs(download_dir, exist_ok=True)

    for model_id in model_list:
        try:
            print(f"下载 {model_id}...")
            # 下载模型文件
            hf_hub_download(
                repo_id=model_id,
                repo_type="model",
                local_dir=download_dir,
                local_dir_use_symlinks=False
            )
            print(f"✅ {model_id} 下载完成")
        except Exception as e:
            print(f"❌ {model_id} 下载失败: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="下载ViT模型用于CLIP实验")
    parser.add_argument("--priority", choices=["high", "medium", "low", "all"],
                       default="high", help="下载优先级")
    args = parser.parse_args()

    if args.priority == "all":
        download_models(HIGH_PRIORITY, "高优先级")
        download_models(MEDIUM_PRIORITY, "中优先级")
        download_models(LOW_PRIORITY, "低优先级")
    elif args.priority == "high":
        download_models(HIGH_PRIORITY, "高优先级")
    elif args.priority == "medium":
        download_models(MEDIUM_PRIORITY, "中优先级")
    elif args.priority == "low":
        download_models(LOW_PRIORITY, "低优先级")

    print("\n🎉 模型下载完成!")
    print("💡 模型保存在 downloaded_models/ 目录中")
EOF

    echo "Python脚本已创建: download_models.py"
    echo ""
    echo "使用方法:"
    echo "python download_models.py --priority high    # 下载高优先级模型"
    echo "python download_models.py --priority medium  # 下载中优先级模型"
    echo "python download_models.py --priority low     # 下载低优先级模型"
    echo "python download_models.py --priority all     # 下载所有模型"
}

# 便捷函数
echo -e "${GREEN}🔧 便捷函数:${NC}"
echo "source /etc/network_turbo  # 启用网络加速"
echo ""

# 预计下载大小
echo -e "${YELLOW}📊 预计下载大小:${NC}"
echo "• 高优先级模型: ~1.2GB"
echo "• 中优先级模型: ~1.8GB"
echo "• 低优先级模型: ~2.5GB"
echo "• 全部模型: ~5.5GB"
echo ""

echo -e "${GREEN}💡 提示:${NC}"
echo "• 可以按Ctrl+C停止下载"
echo "• 已下载的模型会跳过"
echo "• 网络不稳定时建议使用VPN或镜像"
echo "• 下载完成后可以删除此脚本"

echo ""
echo -e "${YELLOW}🚀 脚本准备完成，请按需执行下载命令${NC}"