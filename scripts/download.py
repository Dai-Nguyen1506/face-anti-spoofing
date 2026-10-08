import os
import shutil
import kagglehub

def download_data():
    # 1. Trỏ đến thư mục datasets/LCC_FASD ở ngoài thư mục gốc
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_dir = os.path.join(project_root, "datasets")
    dataset_dir = os.path.join(base_dir, "LCC_FASD")
    
    # 2. Kiểm tra xem thư mục LCC_FASD đã tồn tại chưa
    if os.path.exists(dataset_dir):
        print(f"Dữ liệu đã sẵn sàng tại: {dataset_dir}")
        return

    print("Đang tiến hành tải dataset LCC-FASD từ Kaggle...")
    download_path = kagglehub.dataset_download("faber24/lcc-fasd")
    print(f"Đã tải vào cache tạm: {download_path}")

    # 3. Quét tìm cái vỏ chứa dữ liệu thực tế trong cache tải về
    target_parent_dir = None
    for root, dirs, files in os.walk(download_path):
        if "LCC_FASD_training" in dirs:
            target_parent_dir = root
            break
            
    if not target_parent_dir:
        print("LỖI: Không tìm thấy thư mục 'LCC_FASD_training' bên trong bộ dữ liệu tải về!")
        return

    # 4. Copy nguyên cái vỏ đó sang thư mục datasets của dự án
    print(f"Đang chép toàn bộ dữ liệu vào {dataset_dir} ...")
    shutil.copytree(target_parent_dir, dataset_dir, dirs_exist_ok=True)

    # 5. Vào trong thư mục LCC_FASD vừa copy để đổi tên các thư mục con
    folder_mapping = {
        "LCC_FASD_training": "train",
        "LCC_FASD_development": "val",
        "LCC_FASD_evaluation": "test"
    }

    print("Đang đổi tên các thư mục con bên trong LCC_FASD...")
    for original_name, new_name in folder_mapping.items():
        src_path = os.path.join(dataset_dir, original_name)
        dst_path = os.path.join(dataset_dir, new_name)
        
        if os.path.exists(src_path):
            os.rename(src_path, dst_path) # Đổi tên trực tiếp
            print(f"  [OK] Đã đổi tên: {original_name} -> {new_name}/")
        else:
            # Phòng trường hợp code bị ngắt giữa chừng và chạy lại
            if os.path.exists(dst_path):
                print(f"  [BỎ QUA] Thư mục {new_name} đã tồn tại từ trước.")
            else:
                print(f"  [LỖI] Bị thiếu thư mục: {original_name}")

    print("\nHoàn tất! Cấu trúc dữ liệu hiện tại là: datasets/LCC_FASD/ [train, val, test]")

if __name__ == "__main__":
    download_data()