import os
import sys
import tempfile
import shutil
import gradio as gr
import torch
from PIL import Image
import subprocess
import glob
from pathlib import Path

class ChainOfZoomInterface:
    def __init__(self):
        """初始化Chain-of-Zoom中文界面"""
        self.temp_dir = None
        self.output_dir = None
        
    def setup_directories(self):
        """设置临时目录"""
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir = tempfile.mkdtemp(prefix="coz_")
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        return self.temp_dir, self.output_dir
    
    def get_gpu_info(self):
        """获取GPU信息"""
        if not torch.cuda.is_available():
            return "CUDA不可用"
        
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1024**3
            gpu_info.append(f"GPU {i}: {props.name} ({total_memory:.1f}GB)")
        
        return "\n".join(gpu_info)
    
    def process_image(self, 
                     input_image,
                     rec_type="recursive_multiscale",
                     prompt_type="vlm", 
                     custom_prompt="",
                     rec_num=3,
                     upscale=4,
                     align_method="nofix",
                     process_size=384,
                     efficient_memory=True,
                     mixed_precision="fp16",
                     vae_tiled_size=192,
                     latent_tiled_size=80,
                     latent_tiled_overlap=24):
        """
        处理图像的主要函数
        """
        
        if input_image is None:
            return None, "❌ 请先上传图像！", None
            
        try:
            # 设置目录
            temp_dir, output_dir = self.setup_directories()
            
            # 保存输入图像
            input_path = os.path.join(temp_dir, "input.png")
            if isinstance(input_image, str):
                shutil.copy(input_image, input_path)
            else:
                input_image.save(input_path)
            
            # 构建命令行参数
            cmd = [
                "python", "inference_coz.py",
                "-i", input_path,
                "-o", output_dir,
                "--rec_type", rec_type,
                "--prompt_type", prompt_type,
                "--rec_num", str(rec_num),
                "--upscale", str(upscale),
                "--align_method", align_method,
                "--process_size", str(process_size),
                "--mixed_precision", mixed_precision,
                "--vae_decoder_tiled_size", str(vae_tiled_size),
                "--latent_tiled_size", str(latent_tiled_size),
                "--latent_tiled_overlap", str(latent_tiled_overlap),
                "--pretrained_model_name_or_path", "stabilityai/stable-diffusion-3-medium-diffusers"
            ]
            
            # 添加可选参数
            if custom_prompt.strip():
                cmd.extend(["--prompt", custom_prompt.strip()])
                
            if efficient_memory:
                cmd.append("--efficient_memory")
            
            # 添加模型路径（如果存在）
            model_paths = {
                "--lora_path": "ckpt/SR_LoRA/model_20001.pkl",
                "--vae_path": "ckpt/SR_VAE/vae_encoder_20001.pt", 
                "--ram_ft_path": "ckpt/DAPE/DAPE.pth",
                "--ram_path": "ckpt/RAM/ram_swin_large_14m.pth"
            }
            
            for arg, path in model_paths.items():
                if os.path.exists(path):
                    cmd.extend([arg, path])
            
            # 执行推理
            print(f"🚀 执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                error_msg = f"❌ 处理失败!\n\n📋 标准输出:\n{result.stdout}\n\n🚨 错误输出:\n{result.stderr}"
                return None, error_msg, None
            
            # 查找输出图像
            recursive_dir = os.path.join(output_dir, "recursive")
            if os.path.exists(recursive_dir):
                output_files = glob.glob(os.path.join(recursive_dir, "*.png"))
                if output_files:
                    final_output = output_files[0]
                    
                    # 查找各个尺度的图像
                    per_scale_dir = os.path.join(output_dir, "per-scale")
                    scale_images = []
                    if os.path.exists(per_scale_dir):
                        for i in range(rec_num + 1):
                            scale_dir = os.path.join(per_scale_dir, f"scale{i}")
                            if os.path.exists(scale_dir):
                                scale_files = glob.glob(os.path.join(scale_dir, "*.png"))
                                if scale_files:
                                    scale_images.append(scale_files[0])
                    
                    success_msg = f"✅ 处理成功！生成了 {rec_num + 1} 个尺度的图像。\n\n🎯 最终放大倍数: {upscale ** rec_num}x"
                    return final_output, success_msg, scale_images
            
            return None, "❌ 未找到输出图像，请检查处理过程", None
            
        except Exception as e:
            error_msg = f"❌ 处理过程中发生错误: {str(e)}"
            return None, error_msg, None
    
    def create_interface(self):
        """创建中文Gradio界面"""
        
        # 获取GPU信息
        gpu_info = self.get_gpu_info()
        
        with gr.Blocks(
            title="Chain-of-Zoom: 极端超分辨率处理工具", 
            theme=gr.themes.Soft(),
            css="""
            .gpu-info { background: linear-gradient(45deg, #1e3c72, #2a5298); color: white; padding: 15px; border-radius: 10px; margin: 10px 0; }
            .memory-warning { background: linear-gradient(45deg, #ff6b6b, #ee5a24); color: white; padding: 10px; border-radius: 8px; margin: 5px 0; }
            .memory-tip { background: linear-gradient(45deg, #00b894, #00a085); color: white; padding: 10px; border-radius: 8px; margin: 5px 0; }
            .preset-btn { margin: 5px; }
            """
        ) as interface:
            
            gr.Markdown(f"""
            # 🔎 Chain-of-Zoom: 极端超分辨率处理工具
            
            ## 📊 当前GPU状态
            <div class="gpu-info">
            {gpu_info}
            </div>
            
            ### 🎯 使用说明
            1. **上传图像** - 选择要处理的图片
            2. **配置参数** - 根据你的GPU显存调整设置
            3. **开始处理** - 点击处理按钮生成高分辨率图像
            4. **下载结果** - 保存处理后的图像
            
            <div class="memory-warning">
            ⚠️ <strong>24GB显存优化提示</strong><br>
            • 建议使用"保守模式"预设配置<br>
            • 如果出现显存不足，请使用"极度保守"模式<br>
            • 处理大图时务必降低"处理尺寸"参数
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 输入配置")
                    
                    # 输入图像
                    input_image = gr.Image(
                        label="📷 输入图像",
                        type="pil",
                        height=300
                    )
                    
                    # 基本参数
                    with gr.Group():
                        gr.Markdown("#### 🔧 基本参数")
                        
                        rec_type = gr.Dropdown(
                            choices=[
                                ("最近邻插值 (最快)", "nearest"),
                                ("双三次插值 (快速)", "bicubic"), 
                                ("单步超分辨率", "onestep"),
                                ("递归处理", "recursive"),
                                ("多尺度递归 (推荐)", "recursive_multiscale")
                            ],
                            value="recursive_multiscale",
                            label="🔄 递归类型",
                            info="选择图像处理方法，多尺度递归效果最好但耗时最长"
                        )
                        
                        prompt_type = gr.Dropdown(
                            choices=[
                                ("无提示 (最快)", "null"),
                                ("DAPE模型提示", "dape"),
                                ("视觉语言模型 (推荐)", "vlm")
                            ],
                            value="vlm", 
                            label="💬 提示类型",
                            info="VLM提示效果最好但速度较慢"
                        )
                        
                        custom_prompt = gr.Textbox(
                            label="✏️ 自定义提示词 (可选)",
                            placeholder="输入额外的描述文本...",
                            lines=2,
                            info="添加描述可以改善生成质量"
                        )
                    
                    # 显存优化预设
                    with gr.Group():
                        gr.Markdown("""
                        #### 🎯 显存优化预设 (24GB GPU专用)
                        <div class="memory-tip">
                        💡 <strong>快速配置建议</strong><br>
                        • <strong>极度保守</strong>: 最安全，适合大图或显存紧张<br>
                        • <strong>保守模式</strong>: 平衡性能和显存使用<br>
                        • <strong>平衡模式</strong>: 较好效果，需要充足显存
                        </div>
                        """)
                        
                        def apply_ultra_conservative():
                            return (256, 2, 2, True, "fp16", 128, 64, 16)
                        
                        def apply_conservative():
                            return (384, 3, 3, True, "fp16", 160, 80, 24)
                        
                        def apply_balanced():
                            return (512, 4, 4, True, "fp16", 192, 96, 32)
                        
                        with gr.Row():
                            ultra_conservative_btn = gr.Button(
                                "🛡️ 极度保守", 
                                variant="secondary", 
                                size="sm",
                                elem_classes="preset-btn"
                            )
                            conservative_btn = gr.Button(
                                "⚖️ 保守模式", 
                                variant="primary", 
                                size="sm",
                                elem_classes="preset-btn"
                            )
                            balanced_btn = gr.Button(
                                "🚀 平衡模式", 
                                variant="secondary", 
                                size="sm",
                                elem_classes="preset-btn"
                            )
                    
                    # 高级参数
                    with gr.Accordion("🔬 高级参数设置", open=False):
                        gr.Markdown("""
                        <div class="memory-warning">
                        ⚠️ <strong>显存使用警告</strong><br>
                        调整这些参数会直接影响显存使用量，请谨慎修改！
                        </div>
                        """)
                        
                        with gr.Row():
                            rec_num = gr.Slider(
                                minimum=1,
                                maximum=5,
                                value=3,
                                step=1,
                                label="🔢 递归次数",
                                info="⚡ 影响最终放大倍数和显存使用"
                            )
                            
                            upscale = gr.Slider(
                                minimum=2,
                                maximum=6,
                                value=4,
                                step=1,
                                label="📈 单步放大倍数",
                                info="🎯 每次递归的放大倍数"
                            )
                        
                        process_size = gr.Slider(
                            minimum=256,
                            maximum=768,
                            value=384,
                            step=64,
                            label="🖼️ 处理尺寸",
                            info="🔥 显存使用最关键参数！越大越耗显存"
                        )
                        
                        align_method = gr.Dropdown(
                            choices=[
                                ("无校正", "nofix"),
                                ("小波变换校正", "wavelet"),
                                ("AdaIN校正", "adain")
                            ],
                            value="nofix",
                            label="🎨 颜色对齐方法",
                            info="颜色校正方法，影响最终效果"
                        )
                        
                        with gr.Row():
                            efficient_memory = gr.Checkbox(
                                value=True,
                                label="💾 内存优化模式",
                                info="🔒 24GB显存强烈建议开启！"
                            )
                            
                            mixed_precision = gr.Dropdown(
                                choices=[("半精度 (推荐)", "fp16"), ("全精度", "fp32")],
                                value="fp16",
                                label="🎛️ 数值精度",
                                info="💡 fp16可节省约50%显存"
                            )
                    
                    # 内存微调参数
                    with gr.Accordion("⚙️ 显存微调参数 (专家级)", open=False):
                        gr.Markdown("""
                        <div class="memory-warning">
                        🚨 <strong>专家级设置</strong><br>
                        这些参数直接控制显存分配，不当设置可能导致显存溢出！
                        </div>
                        """)
                        
                        vae_tiled_size = gr.Slider(
                            minimum=128,
                            maximum=384,
                            value=192,
                            step=32,
                            label="🧩 VAE分块大小",
                            info="🔥 越小越省显存，但处理越慢"
                        )
                        
                        latent_tiled_size = gr.Slider(
                            minimum=64,
                            maximum=128,
                            value=80,
                            step=16,
                            label="🎭 潜在空间分块大小", 
                            info="🔥 控制Transformer显存使用"
                        )
                        
                        latent_tiled_overlap = gr.Slider(
                            minimum=16,
                            maximum=48,
                            value=24,
                            step=8,
                            label="🔗 分块重叠大小",
                            info="⚖️ 影响分块边界质量"
                        )
                        
                        gr.Markdown("""
                        <div class="memory-tip">
                        💡 <strong>参数调优建议</strong><br>
                        • 显存不足时：降低所有"分块大小"参数<br>
                        • 处理太慢时：适当增加分块大小<br>
                        • 质量不佳时：增加重叠大小
                        </div>
                        """)
                    
                    # 处理按钮
                    process_btn = gr.Button(
                        "🚀 开始处理图像",
                        variant="primary",
                        size="lg"
                    )
                    
                    # 绑定预设按钮
                    ultra_conservative_btn.click(
                        fn=apply_ultra_conservative,
                        outputs=[process_size, rec_num, upscale, efficient_memory, mixed_precision, vae_tiled_size, latent_tiled_size, latent_tiled_overlap]
                    )
                    
                    conservative_btn.click(
                        fn=apply_conservative,
                        outputs=[process_size, rec_num, upscale, efficient_memory, mixed_precision, vae_tiled_size, latent_tiled_size, latent_tiled_overlap]
                    )
                    
                    balanced_btn.click(
                        fn=apply_balanced,
                        outputs=[process_size, rec_num, upscale, efficient_memory, mixed_precision, vae_tiled_size, latent_tiled_size, latent_tiled_overlap]
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📥 处理结果")
                    
                    # 状态信息
                    status_text = gr.Textbox(
                        label="📊 处理状态",
                        interactive=False,
                        lines=6,
                        info="显示处理进度和结果信息"
                    )
                    
                    # 最终结果
                    final_output = gr.Image(
                        label="🎨 最终结果 (所有尺度合并)",
                        type="filepath",
                        height=300
                    )
                    
                    # 各个尺度的图像
                    scale_gallery = gr.Gallery(
                        label="🔍 各尺度单独结果",
                        show_label=True,
                        elem_id="gallery",
                        columns=2,
                        rows=2,
                        height="auto"
                    )
            
            # 示例图像
            with gr.Row():
                gr.Markdown("### 🖼️ 示例图像")
                example_images = []
                samples_dir = "samples"
                if os.path.exists(samples_dir):
                    sample_files = glob.glob(os.path.join(samples_dir, "*.png"))[:3]
                    for sample_file in sample_files:
                        example_images.append([sample_file])
                
                if example_images:
                    gr.Examples(
                        examples=example_images,
                        inputs=[input_image],
                        label="点击加载示例图像进行测试"
                    )
            
            # 使用说明和故障排除
            with gr.Accordion("📖 使用说明与故障排除", open=False):
                gr.Markdown("""
                ### 🎯 参数详解
                
                **递归类型说明:**
                - **最近邻插值**: 最快速度，质量最低，适合快速预览
                - **双三次插值**: 快速处理，中等质量
                - **单步超分辨率**: 一次性处理，速度较快
                - **递归处理**: 逐步放大，质量较好
                - **多尺度递归**: 最佳质量，但耗时最长 (推荐)
                
                **提示类型说明:**
                - **无提示**: 最快速度，不使用文本引导
                - **DAPE模型**: 使用预训练模型生成提示
                - **视觉语言模型**: 最佳效果，但速度较慢 (推荐)
                
                ### 🔥 24GB显存优化策略
                
                **如果遇到显存不足 (CUDA OOM):**
                1. 🛡️ **立即使用"极度保守"预设**
                2. 📉 **降低处理尺寸** 到 256 或更低
                3. 🧩 **减小VAE分块大小** 到 128
                4. 🔢 **减少递归次数** 到 2
                5. 💾 **确保开启内存优化模式**
                6. 🎛️ **使用fp16精度**
                
                **性能与显存权衡:**
                - **处理尺寸**: 最关键参数，直接影响显存使用
                - **VAE分块大小**: 显存使用第二重要参数
                - **递归次数**: 影响总处理时间和最终放大倍数
                - **内存优化**: 显著节省显存，但会降低速度
                
                **不同场景推荐设置:**
                - **大图 (>1024px)**: 极度保守模式 + 处理尺寸256
                - **中图 (512-1024px)**: 保守模式
                - **小图 (<512px)**: 平衡模式
                
                ### 🚨 常见问题解决
                
                **问题: 处理卡住不动**
                - ✅ 使用极度保守预设
                - ✅ 检查GPU是否被其他程序占用
                - ✅ 重启Python进程清理显存
                
                **问题: CUDA内存溢出**
                - ✅ 降低所有尺寸参数至最小值
                - ✅ 确保开启内存优化模式
                - ✅ 使用fp16精度
                - ✅ 减少递归次数到1-2
                
                **问题: 处理速度极慢**
                - ✅ 关闭内存优化模式 (如果显存充足)
                - ✅ 适当增加分块大小
                - ✅ 使用更简单的递归类型
                
                **问题: 生成质量不佳**
                - ✅ 增加处理尺寸
                - ✅ 使用VLM提示类型
                - ✅ 增加分块重叠大小
                - ✅ 添加自定义提示词
                
                ### 📊 显存使用估算
                
                **极度保守模式**: ~8-12GB
                **保守模式**: ~12-18GB  
                **平衡模式**: ~18-24GB
                
                <div class="memory-tip">
                💡 <strong>专业提示</strong><br>
                建议在处理前运行 <code>python memory_optimizer.py</code> 清理显存并获取个性化推荐设置！
                </div>
                """)
            
            # 绑定处理函数
            process_btn.click(
                fn=self.process_image,
                inputs=[
                    input_image, rec_type, prompt_type, custom_prompt,
                    rec_num, upscale, align_method, process_size,
                    efficient_memory, mixed_precision, vae_tiled_size,
                    latent_tiled_size, latent_tiled_overlap
                ],
                outputs=[final_output, status_text, scale_gallery]
            )
        
        return interface

def main():
    """主函数"""
    # 检查系统环境
    try:
        import torch
        print("🔧 Chain-of-Zoom 中文界面启动中...")
        print(f"📦 PyTorch版本: {torch.__version__}")
        print(f"🔥 CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"🎮 CUDA设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
        else:
            print("⚠️ 警告: CUDA不可用，某些功能可能无法正常工作")
            
    except ImportError:
        print("❌ 警告: PyTorch未找到，请先安装依赖: pip install -r requirements.txt")
    
    # 创建并启动界面
    print("🚀 正在启动Gradio界面...")
    print("🌐 网络配置:")
    print("   - 本地访问: http://localhost:7861")
    print("   - WSL访问: http://127.0.0.1:7861") 
    print("   - 如果无法访问，请尝试以下方法:")
    print("     1. 在Windows PowerShell中运行: netsh interface portproxy add v4tov4 listenport=7861 listenaddress=0.0.0.0 connectport=7861 connectaddress=127.0.0.1")
    print("     2. 或者在WSL中运行: export DISPLAY=:0")
    print("     3. 检查Windows防火墙设置")
    
    coz_interface = ChainOfZoomInterface()
    interface = coz_interface.create_interface()
    
    # 启动界面 - 使用更兼容的配置
    try:
        interface.launch(
            server_name="127.0.0.1",    # 使用localhost而不是0.0.0.0
            server_port=7861,           # 使用不同端口避免冲突
            share=False,                # 不创建公共链接
            debug=True,                 # 调试模式
            inbrowser=False,            # 不自动打开浏览器
            prevent_thread_lock=False   # 允许线程锁定
        )
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("🔄 尝试备用配置...")
        # 备用配置
        interface.launch(
            server_name="0.0.0.0",      # 允许所有IP访问
            server_port=7862,           # 使用备用端口
            share=False,
            debug=False
        )

if __name__ == "__main__":
    main() 