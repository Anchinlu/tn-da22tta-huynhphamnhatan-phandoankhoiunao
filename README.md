# 🧠 TỔNG QUAN DỰ ÁN: PHÂN ĐOẠN KHỐI U NÃO VỚI MÔ HÌNH TRANSUNET TÙY BIẾN

Dự án này là kết quả nghiên cứu và thực nghiệm nhằm giải quyết bài toán phân đoạn khối u não (Brain Tumor Segmentation) trên ảnh cộng hưởng từ (MRI). Dự án tập trung khắc phục những nhược điểm cố hữu của mạng U-Net truyền thống bằng cách kết hợp sức mạnh của Vision Transformer và cơ chế Attention.

---

## 1. THÔNG TIN CHUNG
- **Bài toán:** Phân đoạn ảnh y tế (Medical Image Segmentation). Mục tiêu là tự động khoanh vùng chính xác vị trí và ranh giới của khối u não trên các lát cắt MRI.
- **Mô hình nền tảng (Baseline):** U-Net (2015).
- **Mô hình đề xuất:** TransUNet V2 (Kết hợp CNN, Vision Transformer và Attention Gate).

## 2. BỘ DỮ LIỆU SỬ DỤNG
- **Nguồn gốc:** Kaggle 3M Lower Grade Glioma (LGG) Segmentation Dataset.
- **Quy mô:** 110 bệnh nhân với tổng cộng **3.929 cặp ảnh** (Ảnh MRI đầu vào và ảnh Nhãn/Mask đi kèm).
- **Đặc trưng:** Đầu vào là ảnh đa kênh (FLAIR, T1w, T1gd, T2w) được tổng hợp thành RGB kích thước 256x256. Đầu ra là ảnh nhị phân đen trắng (Trắng = Khối u, Đen = Nền).
- **Thách thức cốt lõi:** Hiện tượng **mất cân bằng lớp (Class Imbalance)** cực kỳ nghiêm trọng, khi diện tích khối u đôi khi chỉ chiếm chưa tới 1% tổng diện tích bức ảnh, khiến các mạng AI thông thường dễ bị rơi vào trạng thái "lười biếng" dự đoán toàn bộ là nền đen.

## 3. CÁC ĐÓNG GÓP CÁ NHÂN VÀ ĐIỂM MỚI CỦA DỰ ÁN
Dự án không sử dụng các mô hình gốc có sẵn mà tiến hành thiết kế, tinh chỉnh để giải quyết 3 bài toán lớn:

1. **Khắc phục điểm mù cục bộ (Tunnel Vision):** 
   - Thay thế toàn bộ khối CNN ở đáy mạng (Bottleneck) của U-Net bằng kiến trúc **Vision Transformer (ViT)**. 
   - *Kết quả:* Giúp mô hình có tầm nhìn toàn cục (Global Context), kết nối thông tin từ mọi ngóc ngách của não bộ, qua đó **triệt tiêu hiện tượng phân đoạn thừa** (Over-segmentation) - lỗi thường gặp khi U-Net nhìn nhầm vân não khỏe mạnh thành khối u.
2. **Khắc phục nhiễu truyền dẫn (Noise Propagation):** 
   - Tích hợp **Cơ chế Cổng chú ý (Attention Gate)** vào các luồng kết nối tắt (Skip Connections). 
   - *Kết quả:* Đóng vai trò như một màng lọc thông minh, chặn đứng các đặc trưng rác (hộp sọ, màng não) từ luồng mã hóa (Encoder) trước khi đưa sang luồng giải mã (Decoder).
3. **Khắc phục mất cân bằng lớp (Class Imbalance):** 
   - Thiết kế hàm mất mát đa mục tiêu **DiceBCELoss** kết hợp trọng số phạt khắt khe `pos_weight = 15.0`. 
   - *Kết quả:* Ép mô hình phải chú ý đến các khối u có kích thước siêu nhỏ, không được phép đoán bừa thành nền đen.

## 4. KẾT QUẢ THỰC NGHIỆM
Mô hình được huấn luyện trên 150 Epochs và so sánh trực tiếp với U-Net trên cùng một tập dữ liệu chuẩn hóa.

| Chỉ số đánh giá | U-Net (Baseline) | TransUNet V2 (Đề xuất) | Đánh giá |
| :--- | :---: | :---: | :--- |
| **Dice Score (Val)** | 0.8352 | **0.8588** | 🟢 **Tăng ~2.36%** (Mức tăng cực kỳ ấn tượng trong y tế) |
| **IoU Score (Val)** | 0.7645 | **0.7891** | 🟢 **Tăng ~2.46%** |
| Số lượng tham số | ~17.2 triệu | ~40.5 triệu | 🟡 Đánh đổi tài nguyên lấy độ chính xác |
| Thời gian phân tích | ~616ms / ảnh | ~2347ms / ảnh | 🟡 Đánh đổi tốc độ lấy độ chuẩn xác |

**Nhận xét biểu đồ (Boxplot & Histogram):**
- TransUNet V2 đạt độ ổn định rất cao: đường Validation Loss bám rất sát Train Loss, chứng tỏ mô hình không bị học vẹt (Overfitting).
- Số lượng ảnh phân đoạn thất bại (Dice = 0.0) của TransUNet giảm đáng kể so với U-Net, trong khi số lượng ảnh đạt điểm tuyệt đối (Dice ~ 1.0) tăng vọt vượt mốc 500 ảnh.

## 5. GIÁ TRỊ THỰC TIỄN
Mặc dù thời gian xử lý chậm hơn U-Net (~2.3s/ảnh), nhưng trong bối cảnh Y tế lâm sàng, sự an toàn và độ chính xác là ưu tiên tối thượng. Mô hình TransUNet V2 hoàn toàn đáp ứng được vai trò như một **Hệ thống chẩn đoán thứ hai (Second Opinion)**. 
Nó hỗ trợ các bác sĩ chẩn đoán hình ảnh khoanh vùng nhanh chóng các khu vực nghi ngờ, tiết kiệm thời gian phân tích thủ công và giảm thiểu rủi ro bỏ sót các khối u ở giai đoạn mầm mống.

---
*Dự án hoàn thành tháng 06/2026.*
