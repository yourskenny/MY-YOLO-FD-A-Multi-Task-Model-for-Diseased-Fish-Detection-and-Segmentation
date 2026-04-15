import urllib.request
import zipfile
import os
import shutil

mirrors = [
    "https://ghproxy.net/https://github.com/feifei-Lee/YOLO-FD/archive/refs/heads/master.zip",
    "https://ghproxy.net/https://github.com/feifei-Lee/YOLO-FD/archive/refs/heads/main.zip",
    "https://github.moeyy.xyz/https://github.com/feifei-Lee/YOLO-FD/archive/refs/heads/master.zip",
    "https://github.moeyy.xyz/https://github.com/feifei-Lee/YOLO-FD/archive/refs/heads/main.zip",
    "https://kgithub.com/feifei-Lee/YOLO-FD/archive/refs/heads/master.zip",
    "https://kgithub.com/feifei-Lee/YOLO-FD/archive/refs/heads/main.zip",
]

zip_path = "repo.zip"

def download_and_extract(url):
    print(f"Trying to download from {url}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response, open(zip_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print("Download successful. Extracting...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # The root folder in the zip file
            namelist = zip_ref.namelist()
            if not namelist:
                return False
            root_dir = namelist[0].split('/')[0]
            zip_ref.extractall(".")
            
        print(f"Extracted to {root_dir}. Moving files...")
        
        # Move files from root_dir to current directory
        for item in os.listdir(root_dir):
            src = os.path.join(root_dir, item)
            dst = os.path.join(".", item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        
        # Cleanup
        os.rmdir(root_dir)
        os.remove(zip_path)
        print("Extraction and cleanup complete.")
        return True
    except Exception as e:
        print(f"Failed: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False

for url in mirrors:
    if download_and_extract(url):
        print("Successfully downloaded and extracted.")
        break
else:
    print("All mirrors failed.")
