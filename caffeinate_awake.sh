#!/bin/bash

# caffeinate命令可以防止系统休眠
# -d 防止显示器休眠
# -i 防止系统空闲休眠
# -s 在计算机休眠时保持系统活动

echo "屏幕保持唤醒程序已启动"
echo "使用系统内置caffeinate命令"
echo "按Ctrl+C终止程序"

# 运行caffeinate命令防止系统和显示器休眠
caffeinate -di 