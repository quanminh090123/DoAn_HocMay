# Dự án phân loại bệnh thận mạn tính

Hiện thư mục hoàn thành phần dữ liệu và EDA của người 1 (bước 1, 3, 4, 5) và phần mô hình, đánh giá của người 2 (bước 6–9). Chưa có ứng dụng Streamlit, kiểm thử cuối và tài liệu demo (bước 10–11).

Bài toán: nhận 24 thuộc tính để phân loại CKD=1 / Not CKD=0. Ưu tiên Recall CKD, theo dõi thêm Precision, F1, ROC-AUC và baseline. Dự án phục vụ học tập và demo.

## Bắt đầu đọc

1. [Báo cáo dữ liệu và EDA](reports/bao_cao_nguoi_1.md).
2. [Bàn giao, kết quả kiểm tra và mức hoàn thành](reports/ban_giao_nguoi_2.md).
3. Mục **Nhiệm vụ 2** dưới đây: danh sách file mô hình, cách chạy toàn bộ và kết quả đánh giá.

Nguồn CSV và dấu kiểm nằm trong [data/README.md](data/README.md); ý nghĩa cột trong [từ điển dữ liệu](reports/data_dictionary.md). Nguồn mô tả gốc UCI đã đối chiếu; URL/ngày tải thực tế của CSV cục bộ cần người tải bổ sung.

## Chạy toàn bộ trên Windows PowerShell

Chạy tại thư mục gốc dự án. Dùng Python 3.13 như môi trường đã kiểm tra; phần nhiệm vụ 2 cũng đã kiểm chứng lại trên Python 3.14 (scikit-learn 1.9.0, pandas 3.0.5).

```powershell
# 1) Chỉ tạo khi chưa có môi trường .venv
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2) Kiểm thử toàn bộ (hợp đồng dữ liệu người 1 + pipeline người 2)
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v

# 3) Bước 6–8: chia dữ liệu, huấn luyện, chọn mô hình (khoảng 50 giây)
.\.venv\Scripts\python.exe -X utf8 -m src.train

# 4) Bước 9: chấm điểm tập test (một lần duy nhất)
.\.venv\Scripts\python.exe -X utf8 -m src.evaluate

# 5) Jupyter
.\.venv\Scripts\python.exe -m jupyter notebook
```

Thứ tự bắt buộc: `src.train` phải chạy trước `src.evaluate` vì evaluate cần `models/pipeline.joblib`. Không dựng `.venv` thì dùng `python` trên PATH thay thế.

Trong Jupyter chọn kernel Python của `.venv`, mở lần lượt `01_eda.ipynb` → `02_preprocessing.ipynb` → `03_model_experiments.ipynb`, mỗi file chọn Restart Kernel and Run All (02 trước, 03 sau; 03 tự lưu lại `models/` và bảng kết quả). Notebook tạo lại hình trong `reports/figures/` và bảng trong `reports/tables/`. Chạy từ thư mục gốc hoặc thư mục notebooks; không chạy từ một thư mục tùy ý khác.

Notebook đã lưu kết quả lần chạy kiểm tra. Hình và bảng sẽ ghi đè khi chạy lại; CSV gốc trong `data/raw/` không bị ghi đè. Báo cáo Markdown chứa nhận xét theo dataset hiện tại; nếu thay dữ liệu, phải cập nhật số liệu và lời nhận xét trong notebook/báo cáo, không chỉ chạy lại biểu đồ.

## Nhiệm vụ 2 — Mô hình & đánh giá (bước 6–9)

Đã hoàn thành: chia train/test, Pipeline tiền xử lý, Cross-Validation, tinh chỉnh và chọn mô hình, đánh giá trên tập test. Chưa có: ứng dụng Streamlit, kiểm thử cuối và tài liệu demo (bước 10–11, thuộc người 3).

### Các file của nhiệm vụ 2

| File | Công dụng |
|---|---|
| `src/train.py` | Bước 6–8: chia 80/20 phân tầng, xây Pipeline (`clean` → `prepare` → `clf`), baseline 5 mô hình, GridSearchCV 4 mô hình × 2 cách mã hóa `sg/al/su`, chọn theo Recall (ưu tiên mô hình đơn giản khi chênh < 1 độ lệch chuẩn), lưu model |
| `src/evaluate.py` | Bước 9: nạp model, dự đoán tập test đúng một lần, xuất bảng chỉ số, hình ma trận nhầm lẫn/ROC, phân tích lỗi FN, tầm quan trọng thuộc tính |
| `tests/test_model.py` | 4 kiểm thử: chia dữ liệu tái lập và phân tầng, Pipeline nhận dữ liệu thô + dòng toàn ô thiếu, hoạt động khi thiếu cột id, lưu–nạp `joblib` cho kết quả giống hệt |
| `notebooks/02_preprocessing.ipynb` | Bước 6–7: hiển thị chia dữ liệu và cấu trúc Pipeline (chỉ gọi hàm từ `src/`, có chú thích giải thích) |
| `notebooks/03_model_experiments.ipynb` | Bước 8–9: hiển thị bảng CV, quy tắc chọn mô hình, đánh giá test kèm hình |
| `models/pipeline.joblib` | Pipeline hoàn chỉnh đã huấn luyện — đầu vào cho ứng dụng của người 3 |
| `models/metadata.json` | Phiếu thông tin: mô hình, siêu tham số, điểm CV, kích thước train/test, phiên bản thư viện |
| `reports/tables/cv_baseline.csv` | So sánh 5 mô hình baseline bằng CV (15 fold) trên train |
| `reports/tables/cv_results.csv` | Kết quả GridSearch: 8 tổ hợp mô hình × cách mã hóa |
| `reports/tables/test_metrics.csv` | Accuracy, Precision, Recall, F1, ROC-AUC trên tập test |
| `reports/tables/test_predictions.csv` | Dự đoán từng mẫu + cột loại lỗi (`FN — bỏ sót CKD` / `FP — báo nhầm` / `đúng`) |
| `reports/tables/feature_importance.csv` | Độ quan trọng thuộc tính (top: `sg`, `hemo`, `pcv`, `sc`, `al`) |
| `reports/figures/confusion_matrix.png`, `reports/figures/roc_curve.png` | Hình đánh giá test |

### Kết quả đã kiểm chứng (13/11/2026)

- Mô hình chọn: **Logistic Regression** (`C=10`, coding `ordinal`) — Recall CV 0,992 ± 0,020, F1 0,994, ROC-AUC 1,000.
- Tập test 80 mẫu, đánh giá một lần: **Recall 0,960** (48/50 ca CKD), Precision 1,000, F1 0,980, ROC-AUC 0,999, Accuracy 0,975; **FN = 2** (id 224, 102), FP = 0.
- Baseline `dummy`: Recall 1,000 nhưng F1 0,769 và AUC 0,500 → Recall luôn phải đọc kèm F1/ROC-AUC.

### Giới hạn bắt buộc khi viết báo cáo

- Test chỉ 80 mẫu, mỗi mẫu ≈ 1,25% Recall; đọc điểm test cùng kết quả CV ± độ lệch chuẩn.
- EDA (notebook 01) đã xem toàn bộ dataset — không dùng các quan sát đó để chọn mô hình dựa trên test.
- Điểm gần tuyệt đối phải cảnh báo rò rỉ và ghi rõ đây là demo học thuật, không phải chẩn đoán lâm sàng.
- Không chọn lại mô hình sau khi đã nhìn điểm test; nếu thay đổi thì cần tập test mới.

### Bàn giao cho người 3 (bước 10–11)

- Nạp model: `joblib.load("models/pipeline.joblib")`, đọc kèm `models/metadata.json` (phiên bản scikit-learn khi nạp phải khớp).
- Ứng dụng Streamlit: nhận input → `pipeline.predict_proba` → hiển thị nhãn và xác suất; ô trống được coi là giá trị thiếu; phải gắn cảnh báo "chỉ phục vụ học thuật".
- Kiểm thử cuối: file rỗng, thiếu cột, sai kiểu dữ liệu → báo lỗi rõ; dự đoán phải tái lập được với cùng model và seed.

## Phạm vi các phần

- Người 1: mục tiêu, nguồn, kiểm tra dữ liệu, làm sạch định dạng, EDA và bàn giao.
- Người 2: tách X/y, chia train/test phân tầng, Pipeline/CV, chọn và đánh giá mô hình.
- Người 3: ứng dụng, kiểm thử cuối, tài liệu chung và demo.

Không fit imputer/scaler trên toàn dataset. `clean_features` chỉ nhận X; `encode_target` xử lý y riêng. Xem tài liệu bàn giao trước khi ghép pipeline.

Các phiên bản trong requirements là môi trường đã kiểm tra phần dữ liệu, chưa bao gồm Streamlit hay thư viện phát sinh ở các phần sau. Chưa kiểm chứng cài lại từ đầu trên một máy khác.
