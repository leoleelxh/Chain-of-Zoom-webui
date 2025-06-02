#!/usr/bin/env python3
"""
GPU内存优化工具
用于Chain-of-Zoom项目的内存管理和优化
"""

import torch
import gc
import os
import psutil
import subprocess
import time

class MemoryOptimizer:
    def __init__(self):
        """初始化内存优化器"""
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
    def get_gpu_memory_info(self):
        """获取GPU内存信息"""
        if not torch.cuda.is_available():
            return "CUDA不可用"
        
        info = []
        for i in range(self.device_count):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = props.total_memory / 1024**3
            
            info.append({
                'device': i,
                'name': props.name,
                'total': total,
                'allocated': allocated,
                'reserved': reserved,
                'free': total - reserved
            })
        return info
    
    def print_memory_status(self):
        """打印内存状态"""
        print("=" * 60)
        print("🔍 GPU内存状态")
        print("=" * 60)
        
        gpu_info = self.get_gpu_memory_info()
        if isinstance(gpu_info, str):
            print(gpu_info)
            return
            
        for info in gpu_info:
            print(f"GPU {info['device']}: {info['name']}")
            print(f"  总内存: {info['total']:.1f}GB")
            print(f"  已分配: {info['allocated']:.1f}GB ({info['allocated']/info['total']*100:.1f}%)")
            print(f"  已保留: {info['reserved']:.1f}GB ({info['reserved']/info['total']*100:.1f}%)")
            print(f"  可用: {info['free']:.1f}GB ({info['free']/info['total']*100:.1f}%)")
            print()
        
        # 系统内存
        ram = psutil.virtual_memory()
        print(f"系统内存: {ram.used/1024**3:.1f}GB / {ram.total/1024**3:.1f}GB ({ram.percent:.1f}%)")
        print("=" * 60)
    
    def clear_gpu_memory(self):
        """清理GPU内存"""
        print("🧹 清理GPU内存...")
        
        if torch.cuda.is_available():
            # 清理PyTorch缓存
            torch.cuda.empty_cache()
            
            # 强制垃圾回收
            gc.collect()
            
            # 同步所有GPU
            for i in range(self.device_count):
                torch.cuda.synchronize(i)
            
            print("✅ GPU内存清理完成")
        else:
            print("❌ CUDA不可用，无法清理GPU内存")
    
    def optimize_for_coz(self):
        """为Chain-of-Zoom优化系统"""
        print("🚀 为Chain-of-Zoom优化系统...")
        
        # 清理内存
        self.clear_gpu_memory()
        
        # 设置环境变量以优化内存使用
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步执行，便于调试
        
        # 设置PyTorch内存分配策略
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False  # 禁用cudnn benchmark以节省内存
            torch.backends.cudnn.deterministic = True
        
        print("✅ 系统优化完成")
    
    def monitor_memory_usage(self, duration=60, interval=5):
        """监控内存使用情况"""
        print(f"📊 开始监控内存使用情况 ({duration}秒，每{interval}秒更新)")
        print("按Ctrl+C停止监控")
        
        try:
            start_time = time.time()
            while time.time() - start_time < duration:
                os.system('clear' if os.name == 'posix' else 'cls')
                self.print_memory_status()
                print(f"监控时间: {time.time() - start_time:.1f}s / {duration}s")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n监控已停止")
    
    def get_recommended_settings(self):
        """根据GPU内存推荐设置"""
        if not torch.cuda.is_available():
            return "CUDA不可用，无法提供推荐设置"
        
        gpu_info = self.get_gpu_memory_info()
        main_gpu = gpu_info[0]  # 使用第一个GPU的信息
        total_memory = main_gpu['total']
        
        print("🎯 推荐设置（基于你的GPU内存）:")
        print("=" * 50)
        
        if total_memory >= 20:  # 20GB+
            print("你的GPU内存充足 (≥20GB)，推荐设置:")
            print("- Process Size: 512-768")
            print("- VAE Tiled Size: 256-384")
            print("- Latent Tiled Size: 96-128")
            print("- Recursion Number: 4-6")
            print("- Efficient Memory: 可选")
            print("- Mixed Precision: fp16")
        elif total_memory >= 12:  # 12-20GB
            print("你的GPU内存中等 (12-20GB)，推荐设置:")
            print("- Process Size: 384-512")
            print("- VAE Tiled Size: 192-256")
            print("- Latent Tiled Size: 80-96")
            print("- Recursion Number: 3-4")
            print("- Efficient Memory: 建议开启")
            print("- Mixed Precision: fp16")
        elif total_memory >= 8:   # 8-12GB
            print("你的GPU内存较少 (8-12GB)，推荐设置:")
            print("- Process Size: 256-384")
            print("- VAE Tiled Size: 128-192")
            print("- Latent Tiled Size: 64-80")
            print("- Recursion Number: 2-3")
            print("- Efficient Memory: 必须开启")
            print("- Mixed Precision: fp16")
        else:  # <8GB
            print("你的GPU内存不足 (<8GB)，推荐设置:")
            print("- Process Size: 256")
            print("- VAE Tiled Size: 128")
            print("- Latent Tiled Size: 64")
            print("- Recursion Number: 2")
            print("- Efficient Memory: 必须开启")
            print("- Mixed Precision: fp16")
            print("⚠️  警告: 内存可能仍然不足，建议升级GPU")
        
        print("=" * 50)
    
    def kill_gpu_processes(self):
        """终止占用GPU的进程（谨慎使用）"""
        print("⚠️  查找占用GPU的进程...")
        
        try:
            # 使用nvidia-smi查找GPU进程
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                print("发现以下GPU进程:")
                for line in lines:
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        pid, name, memory = parts[0], parts[1], parts[2]
                        print(f"  PID: {pid}, 进程: {name}, 内存: {memory}MB")
                
                response = input("\n是否要终止这些进程? (y/N): ")
                if response.lower() == 'y':
                    for line in lines:
                        pid = line.split(', ')[0]
                        try:
                            os.kill(int(pid), 9)
                            print(f"✅ 已终止进程 {pid}")
                        except:
                            print(f"❌ 无法终止进程 {pid}")
                else:
                    print("取消操作")
            else:
                print("✅ 没有发现占用GPU的进程")
                
        except FileNotFoundError:
            print("❌ nvidia-smi未找到，无法查询GPU进程")
        except Exception as e:
            print(f"❌ 查询失败: {e}")

def main():
    """主函数"""
    optimizer = MemoryOptimizer()
    
    print("🔧 Chain-of-Zoom GPU内存优化工具")
    print("=" * 50)
    
    while True:
        print("\n选择操作:")
        print("1. 查看内存状态")
        print("2. 清理GPU内存")
        print("3. 优化系统设置")
        print("4. 获取推荐设置")
        print("5. 监控内存使用")
        print("6. 终止GPU进程 (谨慎)")
        print("0. 退出")
        
        choice = input("\n请输入选择 (0-6): ").strip()
        
        if choice == '1':
            optimizer.print_memory_status()
        elif choice == '2':
            optimizer.clear_gpu_memory()
            optimizer.print_memory_status()
        elif choice == '3':
            optimizer.optimize_for_coz()
        elif choice == '4':
            optimizer.get_recommended_settings()
        elif choice == '5':
            duration = input("监控时长(秒，默认60): ").strip()
            duration = int(duration) if duration.isdigit() else 60
            optimizer.monitor_memory_usage(duration)
        elif choice == '6':
            optimizer.kill_gpu_processes()
        elif choice == '0':
            print("👋 再见!")
            break
        else:
            print("❌ 无效选择，请重试")

if __name__ == "__main__":
    main() 