from diffusers import DiffusionPipeline
import torch

# 指定您的模型路径
model_path = r"C:\models\Z-Image-Turbo"

print(f"正在尝试加载模型: {model_path}")
try:
    # 尝试使用通用的 DiffusionPipeline 加载
    pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.to("cuda")
    print("✅ 模型加载成功！")

    # 尝试生成一张小图
    print("正在生成测试图片...")
    image = pipe(prompt="a cat", num_inference_steps=10, height=512, width=512).images[0]
    image.save("test_output.jpg")
    print("✅ 图片生成并保存为 'test_output.jpg'。模型文件完全正常。")
    print("结论：问题在于 ComfyUI 节点不兼容。")

except Exception as e:
    print(f"❌ 加载或生成失败，模型文件或环境可能有问题。")
    print(f"错误详情: {e}")