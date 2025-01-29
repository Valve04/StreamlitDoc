import streamlit as st


st.title('Factor Analysis')

st.markdown("""   
<ul>
    <p><strong>Factor Analysis (FA)</strong> คือวิธีการทางสถิติที่ใช้ในการลดมิติ (Dimensionality Reduction) และทำความเข้าใจโครงสร้างภายในของข้อมูล โดยการค้นหาตัวแปรแฝง (Latent Variables) ที่ไม่สามารถสังเกตได้โดยตรง แต่เป็นตัวกำหนดความแปรปรวนร่วมของตัวแปรที่สังเกตได้ (Observed Variables)</p>
    <h5>วัตถุประสงค์ของ Factor Analysis
        <p>1.ลดจำนวนตัวแปรลง (Data Reduction) โดยคงข้อมูลสำคัญไว้</p>
        <p>2.ระบุโครงสร้างหรือรูปแบบความสัมพันธ์ระหว่างตัวแปร</p>
        <p>3.ค้นหาตัวแปรแฝงที่อยู่เบื้องหลังตัวแปรที่สังเกตได้</p>
        <p>4.ใช้สำหรับสร้างแบบวัด (Scale Development) ในการวิจัยทางสังคมศาสตร์หรือจิตวิทยา เช่น แบบวัดความพึงพอใจ</p>
    </h5>
    <h5>แนวคิดหลักของ Factor Analysis</h5>
    <p>Factor Analysis ใช้หลักการในการอธิบายตัวแปรที่สังเกตได้ (X1,X2,...,Xp) ว่าเกิดจากการรวมเชิงเส้นของ:</p>
    <ul>
        <li>Factor Loadings: ค่าสัมประสิทธิ์ที่แสดงความสัมพันธ์ระหว่างตัวแปรที่สังเกตได้และตัวแปรแฝง</li>
        <li>Latent Factors (ตัวแปรแฝง): ตัวแปรที่ไม่สามารถสังเกตได้โดยตรง</li>
        <li>Unique Variance (เอกลักษณ์): ความแปรปรวนเฉพาะตัวที่ไม่ได้อธิบายโดยตัวแปรแฝง</li>
    </ul>    
    <p>สมการ</p>
</ul>
""", unsafe_allow_html=True)
st.latex(r"X = LF + E")
st.markdown("""   
<ul>
    <p>X : ตัวแปรที่สังเกตได้</p>
    <p>L : Factor Loadings (เมทริกซ์ที่แสดงความสัมพันธ์)</p>
    <p>F : ตัวแปรแฝง</p>
    <p>E : Error หรือ Unique Variance</p>
    <h5>ประเภทของ Factor Analysis</h5>
        <h6>1.Exploratory Factor Analysis (EFA)</h6>
            <li>ใช้เมื่อต้องการสำรวจโครงสร้างของข้อมูลโดยไม่มีสมมติฐานล่วงหน้าเกี่ยวกับจำนวนตัวแปรแฝง</li>
            <li>เหมาะสำหรับการค้นพบรูปแบบใหม่ในข้อมูล</li>
        <h6>2.Confirmatory Factor Analysis (CFA)</h6>
            <li>ใช้เมื่อต้องการตรวจสอบว่าโครงสร้างของตัวแปรในข้อมูลตรงกับสมมติฐานหรือแบบจำลองที่ตั้งไว้ล่วงหน้า</li>
            <li>ใช้ในบริบทที่มีสมมติฐานชัดเจน เช่น การทดสอบทฤษฎี</li>
    <h5>ขั้นตอนในการทำ Factor Analysis</h5>
        <h6>1.เตรียมข้อมูล</h6>
            <li>ตรวจสอบความเหมาะสมของข้อมูล เช่น การมีความสัมพันธ์กันของตัวแปร (Correlation Matrix)</li>
            <li>ตรวจสอบ Kaiser-Meyer-Olkin (KMO) และ Bartlett’s Test of Sphericity</li>
        <h6>2.เลือกจำนวน Factors</h6>
            <li>ใช้เกณฑ์ต่าง ๆ เช่น Eigenvalue > 1 หรือ Scree Plot</li>
        <h6>หมุนแกน (Factor Rotation)</h6>
            <li>เพื่อปรับปรุงความสามารถในการตีความ (Interpretability) ของ Factor Loadings</li>
            <li>มีทั้งแบบ Orthogonal (เช่น Varimax) และ Oblique (เช่น Oblimin)</li>
        <h6>แปลผลลัพธ์</h6>
            <li>วิเคราะห์ Factor Loadings เพื่อระบุความสัมพันธ์ระหว่างตัวแปรที่สังเกตได้กับตัวแปรแฝง</li>
</ul>
""", unsafe_allow_html=True)
code = ("""
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.datasets import load_iris

# โหลดข้อมูลตัวอย่าง
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# สร้าง Factor Analysis Model
fa = FactorAnalysis(n_components=2, random_state=42)
factors = fa.fit_transform(df)

# แสดงผลลัพธ์
print("Factor Loadings:")
print(pd.DataFrame(fa.components_, columns=df.columns))
print("\nFactor Scores:")
print(factors[:5])  # แสดงผลเฉพาะ 5 แถวแรก

""")
st.code(code,language='python')
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.datasets import load_iris

# โหลดข้อมูลตัวอย่าง
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# สร้าง Factor Analysis Model
fa = FactorAnalysis(n_components=2, random_state=42)
factors = fa.fit_transform(df)

# แสดงผลลัพธ์
st.write("Factor Loadings:")
st.write(pd.DataFrame(fa.components_, columns=df.columns))
st.write("\nFactor Scores:")
st.write(factors[:5])  # แสดงผลเฉพาะ 5 แถวแรก
st.markdown("""   
<ul>
    <h5>การวิเคราะห์ผลลัพธ์</h5>
        <h6>1.Factor Loadings</h6>
            <li>แสดงความสัมพันธ์ระหว่างตัวแปรที่สังเกตได้กับตัวแปรแฝง</li>
            <li>ค่า Loadings ที่สูงในตัวแปรใดบ่งชี้ว่าตัวแปรนั้นเกี่ยวข้องกับ Factor ใด</li>
        <h6>2.Factor Scores</h6>
            <li>แสดงคะแนนของแต่ละข้อมูลในตัวแปรแฝง</li>
        <h6>3.Explained Variance</h6>
            <li>แสดงว่าตัวแปรแฝงอธิบายความแปรปรวนของข้อมูลได้มากน้อยเพียงใด</li>
    <h5>ข้อดีของ Factor Analysis</h5>
        <p>1.ลดจำนวนตัวแปรโดยไม่สูญเสียข้อมูลสำคัญ</p>
        <p>2.ช่วยค้นพบโครงสร้างที่ซ่อนอยู่ในข้อมูล</p>
        <p>3.ใช้สร้างแบบจำลองที่เข้าใจง่ายขึ้น</p>
    <h5>ข้อเสียของ Factor Analysis</h5>
        <p>1.ความยากในการตีความตัวแปรแฝง</p>
        <p>2.สมมติฐานเรื่อง Linear Relationships อาจไม่เหมาะกับข้อมูลที่ซับซ้อน</p>
        <p>3.อ่อนไหวต่อการสุ่มตัวอย่างและข้อมูลที่มีน้อยเกินไป</p>
    <h5>การประยุกต์ใช้งาน Factor Analysis</h5>
        <p>1.จิตวิทยาและสังคมศาสตร์</p>
            <li>การวิเคราะห์แบบวัด (เช่น แบบสอบถามบุคลิกภาพ)</li>
        <p>2.การวิจัยตลาด</p>
            <li>การแบ่งกลุ่มผู้บริโภคตามปัจจัยที่ซ่อนอยู่</li>
        <p>3.ชีววิทยาและวิทยาศาสตร์การแพทย์</p>
            <li>การลดมิติของข้อมูลยีนหรือข้อมูลทางการแพทย์</li>
        <p>4.การเงินและเศรษฐศาสตร์</p>
            <li>วิเคราะห์ความสัมพันธ์ในชุดข้อมูลทางการเงิน</li>
    <p>Factor Analysis เป็นเครื่องมือสำคัญที่ช่วยให้สามารถวิเคราะห์ข้อมูลที่ซับซ้อนและค้นพบโครงสร้างภายในของข้อมูลได้อย่างมีประสิทธิภาพ!</p>
</ul>
""", unsafe_allow_html=True)


import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# โหลดข้อมูลตัวอย่าง (Iris Dataset)
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# แสดงข้อมูลตัวอย่าง
st.write("ข้อมูลตัวอย่าง:")
st.write(df.head())

# สร้าง Factor Analysis Model (เลือก 2 Factors)
fa = FactorAnalysis(n_components=2, random_state=42)
factors = fa.fit_transform(df)

# แสดง Factor Loadings
factor_loadings = pd.DataFrame(fa.components_, columns=df.columns)
st.write("\nFactor Loadings:")
st.write(factor_loadings)

# แสดง Factor Scores
factor_scores = pd.DataFrame(factors, columns=['Factor 1', 'Factor 2'])
st.write("\nFactor Scores (5 แถวแรก):")
st.write(factor_scores.head())

# Visualization: Factor Loadings
plt.figure(figsize=(10, 6))
sns.heatmap(factor_loadings, annot=True, cmap="coolwarm", cbar=True)
plt.title('Heatmap of Factor Loadings')
plt.xlabel('Features')
plt.ylabel('Factors')
plt.show()
st.pyplot(plt)

st.markdown("""   
<ul>
    <h5>การตรวจสอบความเหมาะสมของข้อมูล</h5>
    <p>ก่อนทำ Factor Analysis ควรตรวจสอบความเหมาะสมของข้อมูลด้วย:</p>
        <ul>
            <li>Kaiser-Meyer-Olkin (KMO): ตรวจสอบความสามารถในการลดมิติของข้อมูล</li>
            <li>Bartlett’s Test of Sphericity: ตรวจสอบว่ามีความสัมพันธ์ระหว่างตัวแปรหรือไม่</li>
        </ul>

</ul>
""", unsafe_allow_html=True)

code = ("""
from factor_analyzer import calculate_kmo, Bartlett_sphericity

# คำนวณค่า KMO และ Bartlett's Test
kmo_all, kmo_model = calculate_kmo(df)
bartlett_chi2, bartlett_p = Bartlett_sphericity(df)

print(f"KMO: {kmo_model:.2f}")
print(f"Bartlett's Test: Chi2 = {bartlett_chi2:.2f}, p-value = {bartlett_p:.2f}")

""")
st.code(code,language='python')

from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
# Bartlett's Test
chi_square_value, p_value = calculate_bartlett_sphericity(df)
st.write(f"Bartlett's Test: Chi-Square = {chi_square_value:.2f}, p-value = {p_value:.2f}")

kmo_all, kmo_model = calculate_kmo(df)
st.write(f"KMO for all variables_kmo_all:\n{kmo_all}")
st.write("ค่า KMO ของแต่ละตัวแปร (kmo_all) ที่ต่ำกว่า 0.5 อาจต้องพิจารณาลบตัวแปรออกจากชุดข้อมูล")
st.write(f"KMO Model (Overall)_kmo_model: {kmo_model:.2f}")
st.write("ค่า KMO รวม (kmo_model) ควรมีค่าอย่างน้อย 0.6 เพื่อแสดงถึงความเหมาะสมสำหรับการทำ Factor Analysis")

st.markdown("""   
<ul>
    <li>ค่า KMO ควรอยู่ระหว่าง 0.5–1.0 เพื่อเหมาะสมสำหรับ Factor Analysis</li>
    <li>Bartlett’s Test ควรมี p-value < 0.05 เพื่อแสดงว่ามีความสัมพันธ์ที่มีนัยสำคัญ</li>
</ul>
""", unsafe_allow_html=True)

code = ("""
from sklearn.decomposition import PCA

# คำนวณ Eigenvalues
pca = PCA()
pca.fit(df)
eigenvalues = pca.explained_variance_

# Scree Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(eigenvalues)+1), eigenvalues, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Number of Factors')
plt.ylabel('Eigenvalue')
plt.axhline(y=1, color='red', linestyle='--')
plt.show()

""")
st.code(code,language='python')

from sklearn.decomposition import PCA

# คำนวณ Eigenvalues
pca = PCA()
pca.fit(df)
eigenvalues = pca.explained_variance_

# Scree Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(eigenvalues)+1), eigenvalues, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Number of Factors')
plt.ylabel('Eigenvalue')
plt.axhline(y=1, color='red', linestyle='--')
st.pyplot(plt)

st.markdown("""   
<ul>
    <h5>สรุป</h5>
    <p>Factor Analysis เป็นเทคนิคที่ช่วยลดมิติของข้อมูลและค้นหาตัวแปรแฝงได้อย่างมีประสิทธิภาพ โดย Python มีเครื่องมือที่ช่วยให้การวิเคราะห์นี้ทำได้ง่าย เช่น sklearn และ factor_analyzer พร้อมทั้งสามารถวิเคราะห์ผลลัพธ์ได้ทั้งในรูปแบบตารางและภาพกราฟิก</p>
</ul>
""", unsafe_allow_html=True)