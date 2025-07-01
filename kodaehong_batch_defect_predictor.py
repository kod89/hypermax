try:
    import streamlit as st
except ModuleNotFoundError:
    raise ModuleNotFoundError("This script requires the 'streamlit' package. Please install it using 'pip install streamlit'")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Dummy model training (to be replaced with actual model persistence in real use)
def train_model():
    np.random.seed(42)
    n_samples = 1000
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
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist(), model.feature_importances_

# Train model and retrieve feature importances
model, feature_names, feature_importances = train_model()

# Streamlit App
st.title("의약품 생산 배치 불량 예측 툴")
st.write("CSV 파일을 업로드하면 각 배치의 불량 여부와 예측 확률을 보여줍니다.")

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    required_columns = ["Temperature_C", "Pressure_bar", "MixingSpeed_rpm", "pH", "Yield_percent", "Contaminant_ppm"]

    if all(col in input_df.columns for col in required_columns):
        proba = model.predict_proba(input_df[required_columns])[:, 1]
        predictions = model.predict(input_df[required_columns])
        input_df['Predicted_Defective'] = predictions
        input_df['Defect_Probability_%'] = (proba * 100).round(2)

        st.success("예측 완료! 결과를 아래에서 확인하세요.")
        st.dataframe(input_df)

        csv = input_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="결과 CSV 다운로드",
            data=csv,
            file_name='prediction_results.csv',
            mime='text/csv',
        )

        # Feature importance plot
        st.subheader("📊 주요 변수 중요도")
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": feature_importances
        }).sort_values(by="Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
        ax.set_title("Feature Importances")
        st.pyplot(fig)

    else:
        st.error(f"필수 컬럼이 누락되었습니다: {required_columns}")
else:
    st.info("예측을 시작하려면 먼저 CSV 파일을 업로드하세요.")

# Deployment Info
st.markdown("""
---
### ☁️ 배포 안내
- **Streamlit Cloud**:
  - [https://streamlit.io/cloud](https://streamlit.io/cloud) 에서 GitHub와 연동해 바로 배포 가능
  - 이 코드를 `kodaehong_batch_defect_predictor.py`로 저장 후 GitHub 저장소에 올리고 Streamlit Cloud에서 연결하면 끝!

- **AWS EC2 / Lightsail**:
  - 인스턴스 생성 후 Python, pip 설치
  - 본 파일 업로드 후 `streamlit run kodaehong_batch_defect_predictor.py`로 실행
  - `nohup`, `tmux`, `screen` 등으로 서버 유지 가능

배포에 도움이 필요하시면 언제든지 질문 주세요!
""")

