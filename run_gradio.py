#!/usr/bin/env python3
"""
Chain-of-Zoom Gradio界面启动脚本

使用方法:
python run_gradio.py

或者直接运行:
python gradio_interface.py
"""

import sys
import os

# 确保当前目录在Python路径中
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# 导入并运行Gradio界面
from gradio_interface import main

if __name__ == "__main__":
    print("🔎 启动 Chain-of-Zoom Gradio 界面...")
    print("📝 界面将在 http://localhost:7860 启动")
    print("🛑 按 Ctrl+C 停止服务器")
    print("-" * 50)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 界面已关闭")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("请检查是否已安装所有依赖: pip install -r requirements.txt") 