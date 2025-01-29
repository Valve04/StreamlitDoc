


import streamlit as st


st.title('Regularization')
st.markdown("""   
<ul> 
    <p><strong>Regularization</strong> เป็นเทคนิคที่ใช้ในการลดปัญหาของ Overfitting และปรับปรุงประสิทธิภาพของโมเดล โดยการเพิ่มข้อจำกัดหรือการควบคุมบางอย่างในขณะฝึกโมเดล ทำให้โมเดลมีความสามารถในการเรียนรู้แบบ Generalization ได้ดียิ่งขึ้น</p>
    <h4>ทำไมต้องใช้ Regularization?</h4>
    <p>เมื่อโมเดลมีความซับซ้อนสูง เช่น โมเดลเรียนรู้จากข้อมูลจำนวนมาก และ/หรือจำนวนฟีเจอร์ที่สูง โมเดลอาจเรียนรู้ข้อมูล noise หรือ pattern ที่ไม่สำคัญในข้อมูล ส่งผลให้เกิด Overfitting ซึ่งทำให้โมเดลทำงานได้ดีบนชุดข้อมูลฝึก แต่ทำงานได้ไม่ดีบนชุดข้อมูลทดสอบ</p>
    <h5>1. L1 Regularization (Lasso)</h5>
        <li>L1 Regularization เพิ่มค่าความสูญเสีย (Loss) ด้วยผลรวมของค่าสัมบูรณ์ของน้ำหนัก (coefficients)</li>
</ul>
""", unsafe_allow_html=True)
st.latex(r"Loss = Loss_{\text{original}} + \lambda \sum |w_i|")
st.markdown("""   
<ul> 
    <li>wi คือค่าพารามิเตอร์หรือน้ำหนัก</li>
    <li>λ คือพารามิเตอร์ที่ควบคุมความแรงของการ Regularization</li>
    <li>คุณสมบัติ: ทำให้บางพารามิเตอร์กลายเป็น 0 ซึ่งมีผลในการคัดเลือกฟีเจอร์อัตโนมัติ (Feature Selection)</li>
    <h5>2. L2 Regularization (Ridge)</h5>
        <li>L2 Regularization เพิ่มค่าความสูญเสีย (Loss) ด้วยผลรวมของค่ากำลังสองของน้ำหนัก</li>   
</ul>
""", unsafe_allow_html=True)
st.latex(r"Loss = Loss_{\text{original}} + \lambda \sum w_i^2")
st.markdown("""   
<ul> 
    <li>คุณสมบัติ: ลดขนาดของพารามิเตอร์ทั้งหมดให้น้อยลง แต่ไม่ทำให้ค่าพารามิเตอร์เป็น 0</li>
    <h5>3. Elastic Net</h5>
        <li>Elastic Net เป็นการรวม L1 และ L2 Regularization</li>
</ul>
""", unsafe_allow_html=True)
st.latex(r"Loss = Loss_{\text{original}} + \lambda_1 \sum |w_i| + \lambda_2 \sum w_i^2")
st.markdown("""   
<ul> 
    <li>คุณสมบัติ: ใช้สำหรับข้อมูลที่มีจำนวนฟีเจอร์มากและมีความสัมพันธ์กันสูง (multicollinearity)</li>
    <h5>การใช้ Regularization ใน Machine Learning</h5>
    <h6>การใช้งานใน Linear Models</h6>
    <p>Linear Regression และ Logistic Regression สามารถเพิ่ม Regularization ได้ผ่านโมเดล Ridge (L2), Lasso (L1) หรือ ElasticNet</p>
</ul>
""", unsafe_allow_html=True)

code = ("""
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error

# โหลดข้อมูล
X, y = load_diabetes(return_X_y=True)

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล Ridge, Lasso, ElasticNet
ridge = Ridge(alpha=1.0)  # L2 Regularization
lasso = Lasso(alpha=0.1)  # L1 Regularization
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)  # Elastic Net

# ฝึกโมเดลและประเมินผล
models = {"Ridge": ridge, "Lasso": lasso, "ElasticNet": elastic_net}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name} MSE: {mse:.2f}")

""")
st.code(code,language='python')
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error

# โหลดข้อมูล
X, y = load_diabetes(return_X_y=True)

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล Ridge, Lasso, ElasticNet
ridge = Ridge(alpha=1.0)  # L2 Regularization
lasso = Lasso(alpha=0.1)  # L1 Regularization
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)  # Elastic Net

# ฝึกโมเดลและประเมินผล
models = {"Ridge": ridge, "Lasso": lasso, "ElasticNet": elastic_net}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"{name} MSE: {mse:.2f}")


st.markdown("""   
<ul>
    <h5>ผลกระทบของ Regularization</h5>
    <p>1. L1 Regularization</p>
        <li>ลดจำนวนฟีเจอร์ที่โมเดลใช้ (Feature Selection) เพราะทำให้ค่าพารามิเตอร์บางตัวเป็น 0</li>
        <li>ใช้ได้ดีเมื่อมีฟีเจอร์ที่ไม่สำคัญจำนวนมาก</li>
    <p>2. L2 Regularization</p>
        <li>ลดความซับซ้อนของโมเดลโดยการลดขนาดพารามิเตอร์</li>
        <li>เหมาะสำหรับข้อมูลที่มีความสัมพันธ์สูงระหว่างฟีเจอร์ (multicollinearity)</li>
    <p>3. Elastic Net</p>
        <li>รวมข้อดีของ L1 และ L2 Regularization</li>
        <li>เหมาะสำหรับข้อมูลที่มีฟีเจอร์จำนวนมากและมี multicollinearity</li>
    <h5>Hyperparameter Tuning</h5>
    <p>พารามิเตอร์ λ หรือ alpha ต้องกำหนดค่าให้เหมาะสม โดยสามารถใช้ Cross-Validation เพื่อเลือกค่าที่ดีที่สุด:</p>
</ul>
""", unsafe_allow_html=True)

code = ("""
st.code(code,language='python')
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

# กำหนดพารามิเตอร์ที่ต้องการค้นหา
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge = Ridge()
grid = GridSearchCV(ridge, param_grid, scoring='neg_mean_squared_error', cv=5)
grid.fit(X_train, y_train)

# ดูค่าที่ดีที่สุด
print("Best alpha:", grid.best_params_)

""")
st.code(code,language='python')
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

# กำหนดพารามิเตอร์ที่ต้องการค้นหา
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge = Ridge()
grid = GridSearchCV(ridge, param_grid, scoring='neg_mean_squared_error', cv=5)
grid.fit(X_train, y_train)

# ดูค่าที่ดีที่สุด
st.write("Best alpha:", grid.best_params_)

st.markdown("""   
<ul>
    <h5>สรุป</h5>
    <p>1.Regularization ช่วยลด Overfitting และปรับปรุง Generalization ของโมเดล</p>
    <p>2.มี 3 ประเภทหลัก: L1 (Lasso), L2 (Ridge), และ Elastic Net</p>
    <p>3.การเลือกประเภท Regularization และค่าพารามิเตอร์ที่เหมาะสมขึ้นอยู่กับลักษณะของข้อมูลและปัญหา</p>
</ul>
""", unsafe_allow_html=True)


