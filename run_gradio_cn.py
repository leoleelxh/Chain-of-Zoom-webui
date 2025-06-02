#!/usr/bin/env python3
"""
Chain-of-Zoom 中文版Gradio界面启动脚本
"""

import sys
import os

def main():
    """启动中文版Gradio界面"""
    print("🚀 启动Chain-of-Zoom中文版界面...")
    print("=" * 50)
    
    try:
        # 导入并运行中文界面
        from gradio_interface_cn import main as run_cn_interface
        run_cn_interface()
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保已安装所有依赖: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 