import cv2
import matplotlib.pyplot as plt

def visualize_images(image_paths, titles, rows=1, cols=None, figsize=(15, 5)):
    """
    Hiển thị ảnh trên các trục (axes) với số hàng và cột linh hoạt.

    Args:
        image_paths (list): Danh sách đường dẫn đến các ảnh.
        titles (list): Danh sách tiêu đề tương ứng với các ảnh.
        rows (int): Số hàng trong lưới subplots.
        cols (int): Số cột trong lưới subplots. Nếu không được cung cấp, sẽ tự động tính toán.
        figsize (tuple): Kích thước của toàn bộ hình (width, height).
    """
    if cols is None:
        cols = len(image_paths) // rows + (len(image_paths) % rows > 0)
    
    # Tạo lưới subplots
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    ax = ax.flatten() if rows * cols > 1 else [ax]
    
    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển đổi sang RGB
        ax[i].imshow(image)
        ax[i].set_title(titles[i])
        ax[i].axis('off')
    
    # Ẩn các trục không sử dụng
    for j in range(i + 1, len(ax)):
        ax[j].axis('off')
    
    plt.tight_layout()
    plt.show()