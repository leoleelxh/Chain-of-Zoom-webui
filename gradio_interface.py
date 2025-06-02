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
        """初始化Chain-of-Zoom界面"""
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
    
    def process_image(self, 
                     input_image,
                     rec_type="recursive_multiscale",
                     prompt_type="vlm", 
                     custom_prompt="",
                     rec_num=4,
                     upscale=4,
                     align_method="nofix",
                     process_size=512,
                     efficient_memory=True,
                     mixed_precision="fp16",
                     vae_tiled_size=224,
                     latent_tiled_size=96,
                     latent_tiled_overlap=32):
        """
        处理图像的主要函数
        
        参数说明：
        - input_image: 输入图像
        - rec_type: 递归类型 (nearest/bicubic/onestep/recursive/recursive_multiscale)
        - prompt_type: 提示类型 (null/dape/vlm)
        - custom_prompt: 自定义提示文本
        - rec_num: 递归次数
        - upscale: 放大倍数
        - align_method: 颜色对齐方法 (wavelet/adain/nofix)
        - process_size: 处理尺寸
        - efficient_memory: 是否使用内存优化
        - mixed_precision: 混合精度 (fp16/fp32)
        - vae_tiled_size: VAE分块大小
        - latent_tiled_size: 潜在空间分块大小
        - latent_tiled_overlap: 潜在空间分块重叠
        """
        
        if input_image is None:
            return None, "Please upload an image first!", None
            
        try:
            # 设置目录
            temp_dir, output_dir = self.setup_directories()
            
            # 保存输入图像
            input_path = os.path.join(temp_dir, "input.png")
            if isinstance(input_image, str):
                # 如果是文件路径
                shutil.copy(input_image, input_path)
            else:
                # 如果是PIL图像
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
            print(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                error_msg = f"处理失败!\n标准输出: {result.stdout}\n错误输出: {result.stderr}"
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
                    
                    success_msg = f"处理成功! 生成了 {rec_num + 1} 个尺度的图像。"
                    return final_output, success_msg, scale_images
            
            return None, "未找到输出图像", None
            
        except Exception as e:
            error_msg = f"处理过程中发生错误: {str(e)}"
            return None, error_msg, None
    
    def create_interface(self):
        """创建Gradio界面"""
        
        with gr.Blocks(title="Chain-of-Zoom: Extreme Super-Resolution", theme=gr.themes.Soft()) as interface:
            
            gr.Markdown("""
            # 🔎 Chain-of-Zoom: Extreme Super-Resolution
            
            This tool performs extreme super-resolution using scale autoregression and preference alignment.
            Upload an image and configure the parameters to enhance its resolution through recursive zooming.
            
            ## How it works:
            1. **Upload** your input image
            2. **Configure** the processing parameters  
            3. **Process** to generate multiple resolution scales
            4. **Download** the enhanced results
            
            ### 💡 Memory Optimization Tips for 24GB GPU:
            - Use **Efficient Memory Mode** for maximum memory savings
            - Reduce **Process Size** to 256-384 for large images
            - Lower **VAE Tiled Size** to 128-192 for extreme memory saving
            - Use **fp16** precision to halve memory usage
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input Configuration")
                    
                    # 输入图像
                    input_image = gr.Image(
                        label="Input Image",
                        type="pil",
                        height=300
                    )
                    
                    # 基本参数
                    with gr.Group():
                        gr.Markdown("#### Basic Parameters")
                        
                        rec_type = gr.Dropdown(
                            choices=["nearest", "bicubic", "onestep", "recursive", "recursive_multiscale"],
                            value="recursive_multiscale",
                            label="Recursion Type",
                            info="Type of inference method to use"
                        )
                        
                        prompt_type = gr.Dropdown(
                            choices=["null", "dape", "vlm"],
                            value="vlm", 
                            label="Prompt Type",
                            info="Type of prompt generation method"
                        )
                        
                        custom_prompt = gr.Textbox(
                            label="Custom Prompt (Optional)",
                            placeholder="Enter additional prompt text...",
                            lines=2
                        )
                    
                    # 高级参数
                    with gr.Accordion("Advanced Parameters", open=False):
                        rec_num = gr.Slider(
                            minimum=1,
                            maximum=6,  # 降低最大值以节省内存
                            value=3,    # 降低默认值
                            step=1,
                            label="Recursion Number",
                            info="Number of recursive zoom steps (lower = less memory)"
                        )
                        
                        upscale = gr.Slider(
                            minimum=2,
                            maximum=6,  # 降低最大值
                            value=4,
                            step=1,
                            label="Upscale Factor",
                            info="Magnification factor for each step"
                        )
                        
                        align_method = gr.Dropdown(
                            choices=["nofix", "wavelet", "adain"],
                            value="nofix",
                            label="Color Alignment Method",
                            info="Method for color correction"
                        )
                        
                        process_size = gr.Slider(
                            minimum=256,
                            maximum=768,  # 降低最大值
                            value=384,    # 降低默认值以节省内存
                            step=64,
                            label="Process Size",
                            info="Processing resolution (lower = less memory, faster)"
                        )
                        
                        efficient_memory = gr.Checkbox(
                            value=True,
                            label="Efficient Memory Mode",
                            info="Use memory optimization (STRONGLY RECOMMENDED for 24GB GPU)"
                        )
                        
                        mixed_precision = gr.Dropdown(
                            choices=["fp16", "fp32"],
                            value="fp16",
                            label="Mixed Precision",
                            info="fp16 saves ~50% memory"
                        )
                    
                    # 内存优化参数
                    with gr.Accordion("Memory Optimization (Advanced)", open=False):
                        gr.Markdown("#### 🔧 Fine-tune memory usage for your 24GB GPU")
                        
                        vae_tiled_size = gr.Slider(
                            minimum=128,
                            maximum=512,
                            value=192,  # 更保守的默认值
                            step=32,
                            label="VAE Decoder Tiled Size",
                            info="Smaller = less memory, slower processing"
                        )
                        
                        latent_tiled_size = gr.Slider(
                            minimum=64,
                            maximum=128,
                            value=80,   # 更保守的默认值
                            step=16,
                            label="Latent Tiled Size", 
                            info="Smaller = less memory for transformer"
                        )
                        
                        latent_tiled_overlap = gr.Slider(
                            minimum=16,
                            maximum=64,
                            value=24,   # 更保守的默认值
                            step=8,
                            label="Latent Tiled Overlap",
                            info="Overlap between tiles (affects quality vs memory)"
                        )
                        
                        # 预设配置
                        gr.Markdown("#### 🎯 Quick Presets for 24GB GPU:")
                        
                        def apply_ultra_conservative():
                            return (256, 192, 2, 128, 64, 16, True, "fp16")
                        
                        def apply_conservative():
                            return (384, 224, 3, 160, 80, 24, True, "fp16")
                        
                        def apply_balanced():
                            return (512, 256, 4, 192, 96, 32, True, "fp16")
                        
                        with gr.Row():
                            ultra_conservative_btn = gr.Button("Ultra Conservative", variant="secondary", size="sm")
                            conservative_btn = gr.Button("Conservative", variant="secondary", size="sm")
                            balanced_btn = gr.Button("Balanced", variant="primary", size="sm")
                        
                        ultra_conservative_btn.click(
                            fn=apply_ultra_conservative,
                            outputs=[process_size, vae_tiled_size, rec_num, latent_tiled_size, latent_tiled_overlap, latent_tiled_overlap, efficient_memory, mixed_precision]
                        )
                        
                        conservative_btn.click(
                            fn=apply_conservative,
                            outputs=[process_size, vae_tiled_size, rec_num, latent_tiled_size, latent_tiled_overlap, latent_tiled_overlap, efficient_memory, mixed_precision]
                        )
                        
                        balanced_btn.click(
                            fn=apply_balanced,
                            outputs=[process_size, vae_tiled_size, rec_num, latent_tiled_size, latent_tiled_overlap, latent_tiled_overlap, efficient_memory, mixed_precision]
                        )
                    
                    # 处理按钮
                    process_btn = gr.Button(
                        "🚀 Process Image",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### Results")
                    
                    # 状态信息
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=4
                    )
                    
                    # 最终结果
                    final_output = gr.Image(
                        label="Final Result (All Scales Combined)",
                        type="filepath",
                        height=300
                    )
                    
                    # 各个尺度的图像
                    scale_gallery = gr.Gallery(
                        label="Individual Scale Results",
                        show_label=True,
                        elem_id="gallery",
                        columns=2,
                        rows=2,
                        height="auto"
                    )
            
            # 示例图像
            with gr.Row():
                gr.Markdown("### Example Images")
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
                        label="Click to load example images"
                    )
            
            # 使用说明
            with gr.Accordion("Usage Instructions & Troubleshooting", open=False):
                gr.Markdown("""
                ### Parameter Explanations:
                
                **Recursion Type:**
                - `nearest/bicubic`: Simple interpolation methods (fastest, lowest quality)
                - `onestep`: Single-step super-resolution  
                - `recursive`: Recursive processing with single-scale prompts
                - `recursive_multiscale`: Recursive processing with multi-scale aware prompts (recommended)
                
                **Prompt Type:**
                - `null`: No text prompts (fastest)
                - `dape`: Use DAPE model for prompt generation
                - `vlm`: Use Vision Language Model for prompt generation (recommended but slower)
                
                ### Memory Optimization for 24GB GPU:
                
                **If you get CUDA out of memory errors:**
                1. **Use Ultra Conservative preset** - safest option
                2. **Reduce Process Size** to 256 or lower
                3. **Lower VAE Tiled Size** to 128
                4. **Reduce Recursion Number** to 2-3
                5. **Use fp16 precision** (default)
                6. **Enable Efficient Memory Mode** (default)
                
                **Performance vs Memory Trade-offs:**
                - **Smaller Process Size**: Faster, less memory, lower quality
                - **Smaller VAE Tiled Size**: Much less memory, slower processing
                - **Fewer Recursion Steps**: Less total magnification, faster
                - **Efficient Memory**: Significant memory savings, slower processing
                
                **Recommended Settings for Different Scenarios:**
                - **Large images (>1024px)**: Ultra Conservative preset
                - **Medium images (512-1024px)**: Conservative preset  
                - **Small images (<512px)**: Balanced preset
                
                ### Troubleshooting:
                - **Process hangs**: Try Ultra Conservative preset
                - **CUDA OOM**: Reduce all size parameters by 50%
                - **Very slow**: Disable Efficient Memory if you have enough VRAM
                - **Poor quality**: Increase Process Size and VAE Tiled Size
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
    # 检查必要的依赖
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
    except ImportError:
        print("Warning: PyTorch not found. Please install requirements first.")
    
    # 创建界面
    coz_interface = ChainOfZoomInterface()
    interface = coz_interface.create_interface()
    
    # 启动界面
    interface.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,       # 端口
        share=False,            # 不创建公共链接
        debug=True              # 调试模式
    )

if __name__ == "__main__":
    main() 