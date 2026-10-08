import os

def get_data_stats(base_dir):
    """
    Hàm đọc dữ liệu từ thư mục gốc, trả về đường dẫn, danh sách nhãn và phân bổ số lượng.
    """
    splits = ['train', 'val', 'test']
    classes = ['real', 'spoof']
    stats = {}
    paths = {}

    for split in splits:
        split_dir = os.path.join(base_dir, split)
        paths[split] = split_dir
        stats[split] = {}
        
        if os.path.exists(split_dir):
            for cls in classes:
                cls_dir = os.path.join(split_dir, cls)
                if os.path.exists(cls_dir):
                    stats[split][cls] = len(os.listdir(cls_dir))
                else:
                    stats[split][cls] = 0
                    
    return paths, classes, stats
