import os
import urllib.request
import zipfile

def download_and_extract_coco(data_dir="./coco_data"):
    os.makedirs(data_dir, exist_ok=True)
    url = "http://images.cocodataset.org/zips/val2014.zip"
    zip_path = os.path.join(data_dir, "val2014.zip")
    
    if not os.path.exists(zip_path):
        print(f"正在下载 COCO val2014 数据集到 {zip_path} ... (约 6GB，请耐心等待)")
        urllib.request.urlretrieve(url, zip_path)
        print("下载完成！")
    else:
        print("压缩包已存在，跳过下载。")
        
    extract_path = os.path.join(data_dir, "val2014")
    if not os.path.exists(extract_path):
        print("正在解压...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("解压完成！图像保存在:", extract_path)
    else:
        print("图像文件夹已存在，跳过解压。")

if __name__ == "__main__":
    download_and_extract_coco()