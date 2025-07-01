# This script requires Streamlit to run. Make sure you have Streamlit installed in your local Python environment.
# Install via: pip install streamlit

try:
    import streamlit as st
except ModuleNotFoundError:
    raise ModuleNotFoundError("This script requires the 'streamlit' package. Please install it using 'pip install streamlit'")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Page style
st.set_page_config(page_title="Batch Defect Predictor", layout="centered")
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "Arial"  # Use a universal font to avoid emoji/Unicode errors

# Dummy model training (to be replaced with actual model persistence in real use)
def train_model():
    np.random.seed(42)
    n_samples = 5000  # Increased sample size for better accuracy
    data = {
        "Temperature_C": np.random.normal(25, 1.5, n_samples),
        "Pressure_bar": np.random.normal(1.0, 0.05, n_samples),
        "MixingSpeed_rpm": np.random.normal(120, 10, n_samples),
        "pH": np.random.normal(7.0, 0.3, n_samples),
        "Yield_percent": np.random.normal(95, 2, n_samples),
        "Contaminant_ppm": np.random.exponential(1.0, n_samples),
    }
    df = pd.DataFrame(data)
    df['Defective'] = ((df['Yield_percent'] < 92) | (df['Contaminant_ppm'] > 5)).astype(int)

    X = df.drop(columns='Defective')
    y = df['Defective']
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist(), model.feature_importances_

# Train model and retrieve feature importances
model, feature_names, feature_importances = train_model()

# Streamlit App UI
st.title("의약품 생산 배치 불량 예측 툴")
st.markdown("""
불량률을 사전에 예측하여 생산 효율성과 품질을 높일 수 있습니다. 

업로드한 CSV 데이터를 기반으로 불량 가능성과 주요 영향을 미치는 변수들을 시각화하여 보여줍니다.
""")

uploaded_file = st.file_uploader("📁 CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    # ⛏️ 컬럼 이름에서 단위 제거 (예: 'Temperature_C (℃)' → 'Temperature_C')
    input_df.columns = input_df.columns.str.replace(r"\s*\(.*?\)", "", regex=True)

    required_columns = ["Temperature_C", "Pressure_bar", "MixingSpeed_rpm", "pH", "Yield_percent", "Contaminant_ppm"]

    if all(col in input_df.columns for col in required_columns):
        proba = model.predict_proba(input_df[required_columns])[:, 1]
        predictions = model.predict(input_df[required_columns])
        input_df['Predicted_Defective'] = predictions
        input_df['Defect_Probability_%'] = (proba * 100).round(2)

        st.success("예측 완료! 아래에서 결과를 확인하세요.")
        st.dataframe(input_df.style.background_gradient(cmap='Reds', subset=['Defect_Probability_%']).format("{:.2f}"))

        # ✨ CSV 가독성 개선: 컬럼 순서 조정 및 소수점 제한
        ordered_columns = ["Predicted_Defective", "Defect_Probability_%"] + required_columns
        export_df = input_df[ordered_columns].copy()
        export_df["Defect_Probability_%"] = export_df["Defect_Probability_%"].round(2)

        csv = export_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 결과 CSV 다운로드",
            data=csv,
            file_name='prediction_results.csv',
            mime='text/csv',
        )

        # Feature importance plot (Vertical bar chart for clarity)
        st.subheader("주요 변수 중요도")
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": feature_importances
        }).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = sns.barplot(x="Feature", y="Importance", data=importance_df, ax=ax, palette="Blues_d")
        ax.set_title("변수 중요도 순위", fontsize=14)
        ax.set_xlabel("변수명")
        ax.set_ylabel("중요도")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", label_type="edge")
        fig.tight_layout()
        st.pyplot(fig)

    else:
        st.error(f"❗ 필수 컬럼이 누락되었습니다: {required_columns}")
else:
    st.info("👆 위에서 CSV 파일을 업로드하시면 예측 결과가 표시됩니다.")

# Custom footer
st.markdown("""
---
[🔗 GitHub 저장소 바로가기](https://github.com/kod89/hypermax/tree/main)

테스트 결과를 확인하려면 샘플 데이터를 다운로드하여 업로드하세요.
""")
