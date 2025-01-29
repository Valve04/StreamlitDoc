


import streamlit as st


st.title('Generalized Linear Models')

st.markdown("""   
<ul>
    <p><strong>Generalized Linear Models (GLMs)</strong> เป็นการขยายความของ Linear Regression เพื่อให้สามารถจัดการกับข้อมูลและการกระจายตัวของตัวแปรเป้าหมาย (Dependent Variable) ได้หลากหลายรูปแบบ GLMs ถูกใช้เมื่อความสัมพันธ์ระหว่างตัวแปรต้น (Independent Variables) และตัวแปรเป้าหมายไม่สามารถแสดงออกในรูปของเส้นตรงได้โดยตรง เช่น กรณีที่ตัวแปรเป้าหมายไม่ได้มีการกระจายแบบปกติ (Normal Distribution)</p>
    <h5>องค์ประกอบสำคัญของ GLMs</h5>
    <p>GLMs มี 3 องค์ประกอบหลักที่ทำงานร่วมกัน:</p>
    <h5>1. Random Component (ส่วนของการกระจายตัวของข้อมูล)</h5>
    <p>กำหนดประเภทของการกระจายตัวของตัวแปรเป้าหมาย เช่น:</p>
        <ul>
            <li>การกระจายแบบปกติ (Normal Distribution) สำหรับตัวแปรต่อเนื่อง</li>
            <li>การกระจายแบบเบอร์นูลี (Bernoulli Distribution) สำหรับตัวแปรเป้าหมายแบบสองค่า (Binary)</li>
            <li>การกระจายแบบปัวซอง (Poisson Distribution) สำหรับการนับ (Count Data)</li>  
        <ul>
    <h5>2.Systematic Component (ส่วนของตัวแปรต้น)</h5>
    <p>เป็นการกำหนดตัวแปรต้น (X) ที่ใช้ในการทำนายค่าของตัวแปรเป้าหมาย  สูตรทั่วไป:</p>
</ul>
""", unsafe_allow_html=True)
# Display the equation using LaTeX
st.latex(r"\eta = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p")
st.markdown("""   
<ul>
    <p>η: Linear Predictor (ผลลัพธ์จากการรวมเชิงเส้น)</p>
    <p>β: ค่าสัมประสิทธิ์ของตัวแปร</p>
    <h5>3. Link Function (ฟังก์ชันเชื่อมโยง)</h5>
    <p>ใช้สำหรับแปลงค่าของ Linear Predictor (η) ไปยังค่าที่เหมาะสมสำหรับการกระจายตัวของตัวแปรเป้าหมาย</p>
    <p>ตัวอย่างของ Link Function</p>

</ul>
""", unsafe_allow_html=True)
st.latex(r"สำหรับ Linear Regression: g(\mu) = \mu \quad \text{(Identity Link)}")
st.latex(r"สำหรับ Logistic Regression: g(\mu) = \text{logit}(\mu) = \ln\left(\frac{\mu}{1 - \mu}\right)")
st.latex(r"สำหรับ Poisson Regression: g(\mu) = \ln(\mu)")


st.markdown("""   
<ul>
    <h5>สูตรทั่วไปของ GLMs</h5>
    <p>สมการพื้นฐานของ GLMs คือ:</p>
</ul>
""", unsafe_allow_html=True)
st.latex(r"g(\mu) = \eta = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p")
st.markdown("""   
<ul>
    <p>g(μ) : Link Function</p>
    <p>μ : ค่าเฉลี่ยของตัวแปรเป้าหมาย </p>
    <p>η : Linear Predictor</p>
</ul>
""", unsafe_allow_html=True)

st.markdown("""   
<ul>
    <h5>ประเภทของ GLMs</h5>
    <p>GLMs มีหลายประเภทตามประเภทของตัวแปรเป้าหมายและการกระจายตัวของข้อมูล:</p>
    <h6>1. Linear Regression</h6>
    <li>ใช้กับตัวแปรเป้าหมายแบบต่อเนื่องที่มีการกระจายตัวแบบปกติ</li>
    <li>Link Function: Identity (𝑔(μ)=μ)</li>
    <h6>2. Logistic Regression</h6>
    <li>ใช้สำหรับตัวแปรเป้าหมายแบบ Binary (0 หรือ 1)</li>
    <li>การกระจายตัว: Bernoulli Distribution</li>
</ul>
""", unsafe_allow_html=True)
st.latex(r"\text{Link Function: Logit } \left(g(\mu)\right) = \ln\left(\frac{\mu}{1 - \mu}\right)")
st.markdown("""   
<ul>
    <h6>3. Poisson Regression</h6>
    <li>ใช้สำหรับตัวแปรเป้าหมายที่เป็นข้อมูลการนับ (Count Data)</li>
    <li>การกระจายตัว: Poisson Distribution</li>
</ul>
""", unsafe_allow_html=True)
st.latex(r"\text{Link Function: Logarithm } \left(g(\mu)\right) = \ln(\mu)")
st.markdown("""   
<ul>
    <h6>4. Gamma Regression</h6>
    <li>ใช้สำหรับตัวแปรเป้าหมายที่เป็นค่าบวก เช่น ค่าเวลา หรืออัตรา (Rate)</li>
    <li>การกระจายตัว: Gamma Distribution</li>
</ul>
""", unsafe_allow_html=True)
st.latex(r"\text{Link Function: Inverse } \left(g(\mu)\right) = \frac{1}{\mu}")
st.markdown("""   
<ul>
    <h6>5.Binomial Regression</h6>
    <li>ใช้สำหรับตัวแปรเป้าหมายที่เป็นจำนวนความสำเร็จจากจำนวนทดลองที่กำหนด</li>
    <li>การกระจายตัว: Binomial Distribution</li>
</ul>
""", unsafe_allow_html=True)

st.markdown("""   
<ul>
    <h6>การใช้งาน GLMs ด้วย Python</h6>
    <p>ตัวอย่าง: Logistic Regression</p>
</ul>
""", unsafe_allow_html=True)
code = ("""
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# โหลดข้อมูล
iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)  # ทำให้เป็น Binary Classification

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้าง Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# ทำนายผล
y_pred = model.predict(X_test)

# รายงานผล
print(classification_report(y_test, y_pred))


""")
st.code(code,language='python')

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# โหลดข้อมูล
iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)  # ทำให้เป็น Binary Classification

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้าง Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# ทำนายผล
y_pred = model.predict(X_test)

# รายงานผล
st.write(classification_report(y_test, y_pred))
st.markdown("""   
<ul>
    <h5>ข้อดีของ GLMs</h5>
    <p>1.สามารถจัดการกับข้อมูลที่มีลักษณะต่าง ๆ ได้อย่างยืดหยุ่น</p>
    <p>2.ใช้งานได้ทั้งสำหรับการวิเคราะห์เชิงพรรณนา (Descriptive Analysis) และการทำนาย (Predictive Modeling)</p>
    <p>3.ใช้ Link Function เพื่อปรับเปลี่ยนรูปแบบความสัมพันธ์ระหว่างตัวแปรต้นและตัวแปรเป้าหมาย</p>
</ul>
""", unsafe_allow_html=True)

st.markdown("""   
<ul>
    <h5>ข้อจำกัดของ GLMs</h5>
    <p>1.ความยากในการเลือก Link Function และประเภทของการกระจายตัวที่เหมาะสม</p>
    <p>2.อาจไม่สามารถจัดการกับข้อมูลที่มีโครงสร้างซับซ้อนได้ดีเท่ากับโมเดลที่ไม่เชิงเส้น เช่น Neural Networks</p>
    <p>3.การประมาณค่าสัมประสิทธิ์อาจไม่เสถียรในกรณีที่มี Multicollinearity</p>
    <h5>สรุป</h5>
    <p>Generalized Linear Models (GLMs) เป็นเครื่องมือที่ทรงพลังสำหรับการวิเคราะห์ข้อมูล โดยสามารถรองรับตัวแปรเป้าหมายที่มีการกระจายตัวที่หลากหลาย GLMs ถูกออกแบบให้ปรับเปลี่ยนได้ง่ายด้วยการใช้ Link Function เพื่อเชื่อมโยงระหว่าง Linear Predictor กับตัวแปรเป้าหมาย</p>

</ul>
""", unsafe_allow_html=True)