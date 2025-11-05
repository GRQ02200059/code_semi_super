#!/bin/bash

# 定义时间间隔（以秒为单位）
INTERVAL=60

echo "屏幕保持唤醒脚本已启动"
echo "每${INTERVAL}秒会模拟按一次Shift键"
echo "按Ctrl+C终止脚本"

while true; do
    # 使用AppleScript模拟按下Shift键
    osascript -e 'tell application "System Events" to key code 56' # 56是Shift键的键码
    
    # 显示时间戳
    echo "在 $(date +"%H:%M:%S") 按下了Shift键"
    
    # 等待指定的时间间隔
    sleep $INTERVAL
done 