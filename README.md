# Dự án phân loại bệnh thận mạn tính

Dự án đã hoàn thành phần dữ liệu và EDA của người 1 (bước 1, 3, 4, 5), phần mô hình và đánh giá của người 2 (bước 6–9), cùng ứng dụng Streamlit và kiểm thử của người 3 (bước 10–11). Ứng dụng phục vụ học tập và trình diễn, không dùng để chẩn đoán y khoa.

Bài toán: nhận 24 thuộc tính để phân loại CKD=1 / Not CKD=0. Ưu tiên Recall CKD, theo dõi thêm Precision, F1, ROC-AUC và baseline. Dự án phục vụ học tập và demo.

## Bắt đầu đọc

1. [Báo cáo dữ liệu và EDA](reports/bao_cao_nguoi_1.md).
2. [Bàn giao, kết quả kiểm tra và mức hoàn thành](reports/ban_giao_nguoi_2.md).
3. Mục **Nhiệm vụ 2** dưới đây: danh sách file mô hình, cách chạy toàn bộ và kết quả đánh giá.
4. Mục **Nhiệm vụ 3**: hướng dẫn sử dụng Streamlit, dự đoán từ CSV và kiểm thử ứng dụng.

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

# 6) Khởi động ứng dụng Streamlit
.\.venv\Scripts\python.exe -m streamlit run app.py

# 7) Kiểm thử phần ứng dụng của người 3
.\.venv\Scripts\python.exe -m pytest tests/test_app.py -v
```

Thứ tự bắt buộc: `src.train` phải chạy trước `src.evaluate` vì evaluate cần `models/pipeline.joblib`. Không dựng `.venv` thì dùng `python` trên PATH thay thế.

Trong Jupyter chọn kernel Python của `.venv`, mở lần lượt `01_eda.ipynb` → `02_preprocessing.ipynb` → `03_model_experiments.ipynb`, mỗi file chọn Restart Kernel and Run All (02 trước, 03 sau; 03 tự lưu lại `models/` và bảng kết quả). Notebook tạo lại hình trong `reports/figures/` và bảng trong `reports/tables/`. Chạy từ thư mục gốc hoặc thư mục notebooks; không chạy từ một thư mục tùy ý khác.

Notebook đã lưu kết quả lần chạy kiểm tra. Hình và bảng sẽ ghi đè khi chạy lại; CSV gốc trong `data/raw/` không bị ghi đè. Báo cáo Markdown chứa nhận xét theo dataset hiện tại; nếu thay dữ liệu, phải cập nhật số liệu và lời nhận xét trong notebook/báo cáo, không chỉ chạy lại biểu đồ.

## Nhiệm vụ 2 — Mô hình & đánh giá (bước 6–9)

Đã hoàn thành: chia train/test, Pipeline tiền xử lý, Cross-Validation, tinh chỉnh và chọn mô hình, đánh giá trên tập test. Phần ứng dụng Streamlit và kiểm thử của người 3 được mô tả tại **Nhiệm vụ 3** bên dưới.

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

## Nhiệm vụ 3 — Ứng dụng Streamlit và kiểm thử (bước 10–11)

### Chức năng đã thực hiện

Ứng dụng tại `app.py` sử dụng Pipeline đã huấn luyện trong `models/pipeline.joblib`. Người dùng có thể nhập thông tin một bệnh nhân hoặc tải một file CSV để dự đoán nhiều bệnh nhân. Kết quả gồm nhãn CKD / Not CKD và xác suất mô hình cho từng lớp. Ứng dụng không tự huấn luyện lại mô hình khi người dùng dự đoán.

Giao diện có hai tab:

- **Nhập từng bệnh nhân:** Điền 24 chỉ số, kiểm tra giá trị đầu vào, nhấn **Kiểm tra và dự đoán**. Các chỉ số chưa biết được phép bỏ trống, nhưng ứng dụng sẽ hiển thị cảnh báo vì kết quả có thể kém tin cậy.
- **Dự đoán từ CSV:** Chọn file, xem thông tin kiểm tra, nhấn **Dự đoán toàn bộ**, sau đó tải kết quả bằng nút **Tải kết quả CSV**. Bản demo nhận tối đa **1.000 dòng** mỗi lần.

> **Lưu ý:** CKD / Not CKD là nhãn do mô hình học máy dự đoán. Xác suất hiển thị là ước lượng của mô hình, không được hiểu là chẩn đoán đã xác nhận. Không dùng ứng dụng để quyết định điều trị.

### Cài đặt và chạy

Mở **Windows PowerShell tại thư mục gốc dự án**. Cần Python 3.13 và một bản `models/pipeline.joblib` đáng tin cậy từ chính nhóm. Nếu `.venv` đã tạo và đã cài đủ thư viện thì không cần thực hiện lại hai lệnh đầu.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit thường mở trình duyệt tự động. Nếu không, truy cập `http://localhost:8501`. Khi muốn dừng ứng dụng, quay lại cửa sổ PowerShell đang chạy Streamlit và nhấn `Ctrl + C`.

Để tránh nhầm Python khi PowerShell không cho kích hoạt `.venv`, các lệnh trên đều gọi trực tiếp `python.exe` trong môi trường ảo.

### Cấu trúc CSV đầu vào và đầu ra

CSV đầu vào phải dùng **dấu phẩy** để phân tách cột và có đủ 24 tên cột thuộc tính định nghĩa trong `src/config.py`: 

```text
age,bp,sg,al,su,rbc,pc,pcc,ba,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc,htn,dm,cad,appet,pe,ane
```

File có thể có thêm `id` để đối chiếu từng bệnh nhân và `classification` nếu sử dụng dữ liệu gốc; ứng dụng tự bỏ `classification` trước khi dự đoán, không dùng nhãn thật làm đầu vào. Các cột ngoài danh sách được phép này hoặc thiếu cột bắt buộc sẽ bị báo lỗi. Dữ liệu sai định dạng hoặc giá trị phân loại không hợp lệ cũng được kiểm tra trước khi dự đoán.

CSV kết quả do ứng dụng xuất ra gồm bốn cột:

| Cột | Ý nghĩa |
|---|---|
| `id` | Mã bệnh nhân trong file, hoặc số thứ tự nếu không có `id` |
| `Du_doan` | Nhãn dự đoán `CKD` hoặc `Not CKD` |
| `Xac_suat_CKD` | Xác suất mô hình dành cho lớp CKD (số từ 0 đến 1) |
| `Xac_suat_Not_CKD` | Xác suất mô hình dành cho lớp Not CKD (số từ 0 đến 1) |

Khi mở CSV bằng Excel, nếu dữ liệu dồn vào một cột, hãy nhập dữ liệu với dấu phân cách **Comma (,)** thay vì đổi thủ công các giá trị trong file gốc.

### Luồng xử lý dự đoán

1. Ứng dụng nhận 24 thuộc tính từ form hoặc từ các dòng CSV.
2. `validate_input()` gọi `clean_features()` để kiểm tra cấu trúc, kiểu dữ liệu và các giá trị không hợp lệ; bản dữ liệu đã làm sạch được dùng cho kiểm tra và xem trước.
3. `load_model()` nạp Pipeline đã huấn luyện bằng `joblib` (được Streamlit cache, tránh nạp lại mỗi lần tương tác).
4. `predict_frame()` gọi `predict()` và `predict_proba()` trên dữ liệu đầu vào; Pipeline tự áp dụng các phép biến đổi đã học trong giai đoạn huấn luyện.
5. Ứng dụng hiển thị nhãn, xác suất và xuất kết quả CSV nếu là dự đoán hàng loạt.

Giá trị thiếu được Pipeline xử lý theo cấu hình đã huấn luyện. Ứng dụng từ chối mẫu thiếu toàn bộ 24 chỉ số và cảnh báo khi chỉ thiếu một phần chỉ số. Giới hạn tuổi 120 và huyết áp 300 mmHg là quy tắc nhập liệu của bản demo, **không phải ngưỡng chẩn đoán**.

### Kiểm thử ứng dụng

Kiểm thử tự động riêng phần người 3:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_app.py -v
```

Kiểm thử các phần cũ của nhóm, nếu cần:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

Kết quả kiểm thử được ghi nhận trên máy phát triển sau khi cập nhật ứng dụng:

| Loại kiểm thử | Kết quả | Phạm vi |
|---|---:|---|
| Pytest `tests/test_app.py` | 8/8 đạt | Dữ liệu và Pipeline: mẫu hợp lệ, thiếu cột, sai kiểu, giá trị âm, thiếu giá trị, dự đoán lặp lại và xác suất |
| Kiểm thử thủ công CSV | 11/11 đạt | File hợp lệ, thiếu cột, sai kiểu, huyết áp âm, sai phân loại, cột dư, không có dòng, thiếu toàn bộ chỉ số, thiếu một giá trị, quá 1.000 dòng và tải kết quả |

Các số trên là **kết quả kiểm thử chức năng**, không phải độ chính xác chẩn đoán của mô hình. Bộ Pytest mới chủ yếu kiểm thử chức năng dữ liệu và Pipeline; thao tác trên giao diện Streamlit được thử thủ công.

### Các file liên quan

| File | Vai trò |
|---|---|
| `app.py` | Giao diện, kiểm tra dữ liệu và dự đoán đơn/hàng loạt |
| `tests/test_app.py` | 8 kiểm thử tự động của phần người 3 |
| `models/pipeline.joblib` | Pipeline đã huấn luyện, do người 2 bàn giao |
| `models/metadata.json` | Thông tin mô hình và phiên bản thư viện |
| `src/config.py` | Định nghĩa tên và kiểu thuộc tính đầu vào |
| `src/preprocess.py` | Hàm `clean_features()` để làm sạch và kiểm tra dữ liệu |
| `requirements.txt` | Danh sách thư viện, bao gồm Streamlit và Pytest |

### Giới hạn của ứng dụng

Ứng dụng chỉ phục vụ demo học thuật với bộ dữ liệu nhỏ. Chức năng dự đoán có thể hoạt động khi thiếu một số thuộc tính nhưng không bảo đảm kết quả đáng tin cậy nếu người dùng nhập quá ít dữ liệu. Kết quả đánh giá của nhóm dựa trên tập test 80 mẫu, chưa chứng minh được hiệu quả sử dụng trong môi trường lâm sàng.

## Phạm vi các phần

- Người 1: mục tiêu, nguồn, kiểm tra dữ liệu, làm sạch định dạng, EDA và bàn giao.
- Người 2: tách X/y, chia train/test phân tầng, Pipeline/CV, chọn và đánh giá mô hình.
- Người 3: ứng dụng, kiểm thử cuối, tài liệu chung và demo.

Không fit imputer/scaler trên toàn dataset. `clean_features` chỉ nhận X; `encode_target` xử lý y riêng. Xem tài liệu bàn giao trước khi ghép pipeline.

File `requirements.txt` đã được người 3 bổ sung `streamlit==1.67.0` và `pytest==9.2.0` vào danh sách thư viện của nhóm. Chưa kiểm chứng cài lại từ đầu trên một máy khác.
