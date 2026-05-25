import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'transunet'])
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--pos_weight', type=float, default=10.0,
                        help='Trọng số cho pixel khối u trong BCE Loss (xử lý mất cân bằng dữ liệu)')
    parser.add_argument('--dataset_path_local', type=str, default=r"D:\\u0110eTaiTotNghiep\\archive\\kaggle_3m")
    parser.add_argument('--dataset_path_kaggle', type=str, default="/kaggle/input/datasets/anhpnht/lgg-mri-segmentation/kaggle_3m")
    parser.add_argument('--resume', type=str, default='', help='Đường dẫn tới file .pth để tiếp tục huấn luyện')
    parser.add_argument('--test_mode', action='store_true')
    return parser.parse_args()
