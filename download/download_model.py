import os
from huggingface_hub import snapshot_download

# 1. 强制使用国内镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 定义待下载模型列表 (Repo ID, 本地存储路径)
models = [
    ("llava-hf/llava-1.5-7b-hf", "./models/llava-1.5-7b-hf"),
    ("Qwen/Qwen2-VL-7B-Instruct", "./models/Qwen2-VL-7B-Instruct"),
    ("Qwen/Qwen2.5-VL-7B-Instruct", "./models/Qwen2.5-VL-7B-Instruct"),
]

def download_all():
    for repo_id, local_dir in models:
        print(f"\n🚀 正在开始下载: {repo_id} 到 {local_dir}...")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                resume_download=True,
                # 过滤掉不需要的非主流格式，节省空间和带宽
                ignore_patterns=["*.msgpack", "*.h5", "*.ot", "onnx/*"],
                # 针对 InternVL2，它文件较多，建议开启 max_workers
                max_workers=8 
            )
            print(f"✅ {repo_id} 下载完成！")
        except Exception as e:
            print(f"❌ {repo_id} 下载出错: {str(e)}")

if __name__ == "__main__":
    download_all()