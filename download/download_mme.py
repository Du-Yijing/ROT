# download_mme.py
from datasets import load_dataset

print("开始连接 HuggingFace 下载 MME 数据集 (lmms-lab/MME)...")
try:
    dataset = load_dataset("lmms-lab/MME", split="test")
    print(f"下载成功！共包含 {len(dataset)} 条测试数据。")
    categories = set(dataset['category'])
    print(f"包含的子集类别有: {categories}")
    print("下载完成，可以开始运行基线与 FSHC 测试！")
except Exception as e:
    print(f"下载失败，请检查网络或是否开启了学术加速。错误信息: {e}")