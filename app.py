from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import FEATURE_COLUMNS, NUMERIC_COLUMNS, CODED_COLUMNS, CATEGORY_VALUES
from src.preprocess import clean_features

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "pipeline.joblib"
MAX_ROWS = 1000

LABELS = {
    "age": "Tuổi", "bp": "Huyết áp (mmHg)", "sg": "Tỷ trọng nước tiểu",
    "al": "Albumin", "su": "Đường trong nước tiểu", "rbc": "Tình trạng hồng cầu",
    "pc": "Tế bào mủ", "pcc": "Cụm tế bào mủ", "ba": "Vi khuẩn",
    "bgr": "Đường huyết ngẫu nhiên", "bu": "Urê máu",
    "sc": "Creatinine huyết thanh", "sod": "Natri", "pot": "Kali",
    "hemo": "Hemoglobin", "pcv": "Thể tích hồng cầu", "wc": "Số lượng bạch cầu",
    "rc": "Số lượng hồng cầu", "htn": "Tăng huyết áp", "dm": "Tiểu đường",
    "cad": "Bệnh động mạch vành", "appet": "Khẩu vị",
    "pe": "Phù chân", "ane": "Thiếu máu",
}

st.set_page_config(page_title="Dự đoán bệnh thận", page_icon="🩺", layout="wide")


@st.cache_resource
def load_model():
    """Chỉ nạp file Joblib thuộc nguồn đã kiểm tra và tin cậy."""
    return joblib.load(MODEL_PATH)


def validate_input(df: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str]]:
    """Kiểm tra đầu vào, trả dữ liệu đã làm sạch hoặc danh sách lỗi.

    Dữ liệu đã làm sạch dùng để kiểm tra; pipeline vẫn nhận bản gốc
    và tự thực hiện cùng bước làm sạch khi dự đoán.
    """
    errors = []
    try:
        cleaned = clean_features(df)
    except (TypeError, ValueError, KeyError) as exc:
        return None, [str(exc)]

    for col in NUMERIC_COLUMNS + CODED_COLUMNS:
        values = cleaned[col].dropna()
        if (values < 0).any():
            errors.append(f"{LABELS[col]} ({col}) không được là số âm.")
    # Kiểm tra các giá trị hợp lệ của sg, al, su
    allowed_sg = [1.005, 1.010, 1.015, 1.020, 1.025]

    if not cleaned["sg"].dropna().isin(allowed_sg).all():
        errors.append("Tỷ trọng nước tiểu (sg) không hợp lệ.")

    for col in ["al", "su"]:
        if not cleaned[col].dropna().isin(range(6)).all():
            errors.append(f"{LABELS[col]} ({col}) phải có giá trị từ 0 đến 5.")

    if (cleaned["age"].dropna() > 120).any():
        errors.append("Tuổi không được lớn hơn 120 (giới hạn giao diện demo).")
    if (cleaned["bp"].dropna() > 300).any():
        errors.append("Huyết áp lớn hơn 300 mmHg (ngoài giới hạn giao diện demo).")
    if cleaned.isna().all(axis=1).any():
        errors.append("Có bệnh nhân thiếu toàn bộ 24 chỉ số; không thể dự đoán.")
    return cleaned, errors


def predict_frame(model, frame: pd.DataFrame) -> pd.DataFrame:
    """Trả nhãn và xác suất theo nhãn thực của model.classes_."""
    predictions = model.predict(frame)
    probs = model.predict_proba(frame)
    classes = list(model.classes_)
    if 0 not in classes or 1 not in classes:
        raise ValueError(f"Nhãn mô hình không như dự kiến: {classes}")
    return pd.DataFrame({
        "Du_doan": ["CKD" if int(x) == 1 else "Not CKD" for x in predictions],
        "Xac_suat_CKD": probs[:, classes.index(1)],
        "Xac_suat_Not_CKD": probs[:, classes.index(0)],
    })


def patient_widget(feature: str):
    label = f"{LABELS[feature]} ({feature})"
    if feature in NUMERIC_COLUMNS:
        value = st.text_input(label, placeholder="Nhập số hoặc để trống", key=f"patient_{feature}")
        return value.strip() or None
    if feature in CODED_COLUMNS:
        choices = (["", "1.005", "1.010", "1.015", "1.020", "1.025"]
                   if feature == "sg" else ["", "0", "1", "2", "3", "4", "5"])
    else:
        choices = [""] + list(CATEGORY_VALUES[feature])
    selected = st.selectbox(
        label, choices, key=f"patient_{feature}",
        format_func=lambda value: "Chưa biết" if value == "" else str(value),
    )
    return selected or None


st.title("HỆ THỐNG DỰ ĐOÁN BỆNH THẬN")
st.write("Đồ án kết thúc học phần Học máy – Phân lớp CKD / Not CKD")
st.warning("Ứng dụng minh họa học thuật, không thay thế chẩn đoán hoặc tư vấn y khoa.")

tab_single, tab_csv = st.tabs(["Nhập từng bệnh nhân", "Dự đoán từ CSV"])

with tab_single:
    st.subheader("Thông tin bệnh nhân (24 chỉ số)")
    with st.form("patient_form"):
        left, right = st.columns(2)
        form_data = {}
        for i, feature in enumerate(FEATURE_COLUMNS):
            with (left if i % 2 == 0 else right):
                form_data[feature] = patient_widget(feature)
        submit_patient = st.form_submit_button("Kiểm tra và dự đoán", type="primary")

    if submit_patient:
        patient_df = pd.DataFrame([form_data], columns=list(FEATURE_COLUMNS))
        cleaned, errors = validate_input(patient_df)
        if errors:
            for error in errors:
                st.error(error)
        else:
            missing = int(cleaned.isna().sum().sum())
            st.success("Dữ liệu hợp lệ theo quy tắc kiểm tra của ứng dụng.")
            st.caption(f"Số chỉ số chưa nhập: {missing}/24")
            if missing:
                st.warning("Một số chỉ số còn thiếu; mô hình sẽ điền giá trị theo pipeline đã huấn luyện.")
            with st.expander("Xem dữ liệu đầu vào đã làm sạch"):
                st.dataframe(cleaned, hide_index=True)
            try:
                model = load_model()
                result = predict_frame(model, patient_df).iloc[0]
                st.subheader("KẾT QUẢ DỰ ĐOÁN")
                if result["Du_doan"] == "CKD":
                    st.error("Mô hình dự đoán: CKD")
                else:
                    st.success("Mô hình dự đoán: Not CKD")
                a, b = st.columns(2)
                a.metric("Xác suất mô hình – CKD", f"{result['Xac_suat_CKD']:.2%}")
                b.metric("Xác suất mô hình – Not CKD", f"{result['Xac_suat_Not_CKD']:.2%}")
                st.caption("Xác suất do mô hình ước lượng, chưa được hiệu chuẩn hoặc xác nhận cho chẩn đoán thực tế.")
            except Exception as exc:
                st.error("Không thể dự đoán. Kiểm tra mô hình và môi trường Python.")
                st.code(str(exc))

with tab_csv:
    st.subheader("Dự đoán nhiều bệnh nhân từ file CSV")
    st.write("CSV cần đủ 24 cột đầu vào; có thể có thêm `id` và `classification`.")
    uploaded = st.file_uploader("Chọn CSV (dấu phẩy)", type=["csv"], key="batch_upload")
    if uploaded is not None:
        try:
            source = pd.read_csv(uploaded, encoding="utf-8-sig")
            if source.empty:
                st.error("File CSV không có dữ liệu.")
            elif len(source) > MAX_ROWS:
                st.error(f"Bản demo chỉ nhận tối đa {MAX_ROWS} dòng mỗi lần.")
            elif source.columns.duplicated().any():
                st.error("File CSV có tên cột bị trùng.")
            else:
                # Nhãn thực tế không được đưa vào mô hình.
                if "classification" in source.columns:
                    st.info("Đã bỏ cột classification (nhãn thật) khỏi đầu vào dự đoán.")
                    source = source.drop(columns=["classification"])
                ids = source["id"].reset_index(drop=True) if "id" in source.columns else pd.Series(range(1, len(source) + 1))
                cleaned, errors = validate_input(source)
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    st.success(f"Đã đọc {len(source)} bệnh nhân.")
                    missing_cells = int(cleaned.isna().sum().sum())
                    if missing_cells:
                        st.warning(f"Có {missing_cells} ô chưa có dữ liệu; pipeline sẽ xử lý giá trị thiếu.")
                    with st.expander("Xem dữ liệu đã tải"):
                        st.dataframe(cleaned, hide_index=True)
                    if st.button("Dự đoán toàn bộ", type="primary", key="run_batch"):
                        try:
                            model = load_model()
                            results = predict_frame(model, source)
                            results.insert(0, "id", ids.to_numpy())
                            st.subheader("Kết quả dự đoán")
                            st.dataframe(results, hide_index=True, column_config={
                                "Xac_suat_CKD": st.column_config.NumberColumn("Xác suất CKD", format="percent"),
                                "Xac_suat_Not_CKD": st.column_config.NumberColumn("Xác suất Not CKD", format="percent"),
                            })
                            left_metric, right_metric = st.columns(2)
                            left_metric.metric("Dự đoán CKD", int((results["Du_doan"] == "CKD").sum()))
                            right_metric.metric("Dự đoán Not CKD", int((results["Du_doan"] == "Not CKD").sum()))
                            st.download_button(
                                "Tải kết quả CSV",
                                data=results.to_csv(index=False).encode("utf-8-sig"),
                                file_name="ket_qua_du_doan_benh_than.csv",
                                mime="text/csv",
                            )
                        except Exception as exc:
                            st.error("Có lỗi khi dự đoán hàng loạt.")
                            st.code(str(exc))
        except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
            st.error("Không thể đọc file CSV. Kiểm tra mã hóa và dấu phân cách (phải là dấu phẩy).")
            st.code(str(exc))

st.caption("Kết quả chỉ phục vụ minh họa học thuật, không dùng để tự chẩn đoán hoặc điều trị.")
