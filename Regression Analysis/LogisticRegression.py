import streamlit as st


st.title('Logistic Regression')

st.write('Logistic Regression เป็นเทคนิคทางสถิติที่ใช้สำหรับการจำแนกประเภท (Classification) โดยเฉพาะกรณีที่ผลลัพธ์ (Dependent Variable) มีลักษณะเป็นค่าจำกัด เช่น 1 หรือ 0, ใช่หรือไม่ใช่, หรือ กลุ่มใดกลุ่มหนึ่ง')

st.markdown("""   
<ul> 
    <p> 
       <strong>หลักการเบื้องหลัง</strong>
    </p>    
    <p> 
       แม้ว่าจะมีคำว่า "Regression" ในชื่อ แต่ Logistic Regression ใช้สำหรับงานจำแนกประเภท แทนที่จะประมาณค่าต่อเนื่องแบบ 
        Linear Regression โดยใช้ สมการโลจิสติก เพื่อสร้างผลลัพธ์ที่อยู่ในช่วงระหว่าง 0 ถึง 1 ซึ่งเหมาะสำหรับการจำแนกประเภท
    </p>    
    <p>
        สมการพื้นฐานของ Logistic Regression คือ:
    </p>

                     
</ul>
""", unsafe_allow_html=True)

st.latex(r'''
P(y=1 \mid x) = \frac{1}{1 + e^{-(b_0 + b_1x_1 + b_2x_2 + \dots + b_nx_n)}}
''')

st.markdown("""   
<ul> 
    <li> 
       P(y=1|x):ความน่าจะเป็นของ y เท่ากับ 1 เมื่อมีตัวแปรต้น
    </li> 
    <li> 
        b0,b1,...,bn : ค่าคงที่และค่าสัมประสิทธิ์
    </li>     
    <li> 
        e (Euler's number) : ค่าคงที่ (ประมาณ 2.718)
    </li>    
    <p>
        ค่าความน่าจะเป็นนี้จะถูกแปลงให้เป็น 1 หรือ 0 โดยใช้เกณฑ์ เช่น หากความน่าจะเป็นมากกว่า 0.5 จะจัดกลุ่มเป็น 1 มิฉะนั้นจัดเป็น 0
    </p>
              
</ul>
""", unsafe_allow_html=True)


st.markdown("""   
<ul> 
    <p><strong>ประเภทของ Logistic Regression</strong></p>
    <li> 
       Binary Logistic Regression: จำแนกเป็น 2 กลุ่ม (เช่น ใช่/ไม่ใช่, ผ่าน/ไม่ผ่าน)
    </li> 
    <li> 
        Multinomial Logistic Regression: จำแนกมากกว่า 2 กลุ่มแบบไม่มีลำดับ (เช่น สีแดง/สีน้ำเงิน/สีเขียว)
    </li>     
    <li> 
        Ordinal Logistic Regression: จำแนกมากกว่า 2 กลุ่มที่มีลำดับ (เช่น ระดับความพึงพอใจ: ต่ำ/กลาง/สูง)
    </li>                 
</ul>
""", unsafe_allow_html=True)

st.markdown("""   
<ul> 
    <p><strong> ตัวอย่างในชีวิตจริง </strong></p>
    <li> 
       การจำแนกโรค: เป็นโรคหรือไม่
    </li> 
    <li> 
        การวิเคราะห์การตลาด: ลูกค้าจะซื้อสินค้าหรือไม่
    </li>     
    <li> 
        การทำนายคะแนนสอบ: ผ่านหรือไม่ผ่าน
    </li>                 
</ul>
""", unsafe_allow_html=True)


st.subheader("ตัวอย่างใน Python")
st.write("1.การนำเข้าข้อมูล")
st.write("สมมติว่าเรามีชุดข้อมูลเกี่ยวกับการยอมรับเข้ามหาวิทยาลัย โดยขึ้นอยู่กับคะแนนสอบและเกรดเฉลี่ย")


code = """
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ข้อมูลตัวอย่าง
data = {
    'ExamScore': [88, 92, 75, 70, 80, 65, 60, 58, 90, 85],
    'GPA': [3.8, 4.0, 3.5, 3.2, 3.7, 3.0, 2.8, 2.5, 4.0, 3.9],
    'Admitted': [1, 1, 1, 0, 1, 0, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

# แยกตัวแปรต้นและตัวแปรตาม
X = df[['ExamScore', 'GPA']]
y = df['Admitted']
"""

st.code(code, language='python')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ข้อมูลตัวอย่าง
data = {
    'ExamScore': [88, 92, 75, 70, 80, 65, 60, 58, 90, 85],
    'GPA': [3.8, 4.0, 3.5, 3.2, 3.7, 3.0, 2.8, 2.5, 4.0, 3.9],
    'Admitted': [1, 1, 1, 0, 1, 0, 0, 0, 1, 1]
}

df = pd.DataFrame(data)

# แยกตัวแปรต้นและตัวแปรตาม
X = df[['ExamScore', 'GPA']]
y = df['Admitted']

st.write("2.การแบ่งข้อมูล")
code = """
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.code(code, language='python')

st.write("3. การสร้างและฝึก Logistic Regression")
code = """
# สร้าง Logistic Regression
model = LogisticRegression()

# ฝึกโมเดล
model.fit(X_train, y_train)

# แสดงค่าสัมประสิทธิ์
print("Intercept (b0):", model.intercept_)
print("Coefficients (b1, b2):", model.coef_)

"""
# สร้าง Logistic Regression
model = LogisticRegression()

# ฝึกโมเดล
model.fit(X_train, y_train)

# แสดงค่าสัมประสิทธิ์
print("Intercept (b0):", model.intercept_)
print("Coefficients (b1, b2):", model.coef_)
st.code(code, language='python')

st.write("4. การพยากรณ์")
code = """
# พยากรณ์ผลลัพธ์
y_pred = model.predict(X_test)

# ประเมินผล
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
"""
# พยากรณ์ผลลัพธ์
y_pred = model.predict(X_test)

# ประเมินผล
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
st.code(code, language='python')

st.write("5. การแสดงผลลัพธ์")
code = """
import matplotlib.pyplot as plt

plt.scatter(df['ExamScore'], df['GPA'], c=df['Admitted'], cmap='bwr', edgecolor='k')
plt.xlabel('Exam Score')
plt.ylabel('GPA')
plt.title('Admission Decision')
plt.show()
"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
scatter = ax.scatter(df['ExamScore'], df['GPA'], c=df['Admitted'], cmap='bwr', edgecolor='k')
ax.set_xlabel('Exam Score')
ax.set_ylabel('GPA')
ax.set_title('Admission Decision')
st.code(code, language='python')
st.pyplot(fig)


st.markdown("""   
<ul> 
    <p><strong> ผลลัพพ์ </strong></p>
    <li> 
       Intercept (b0): จุดตัดของเส้นโลจิสติกกับแกน y
    </li> 
    <li> 
        Coefficients (b1,b2): ผลกระทบของตัวแปรต้นแต่ละตัวต่อความน่าจะเป็น
    </li>     
    <li> 
        Accuracy Score: ความแม่นยำของโมเดล
    </li>       
    <li> 
        Confusion Matrix: ตารางที่แสดงค่าจริงและค่าพยากรณ์
    </li>
    <li> 
        Classification Report: รายงานค่า Precision, Recall, และ F1-score
    </li>          
    <p><strong> ข้อดี </strong></p>  
    <li> 
        เหมาะสำหรับงานจำแนกประเภท
    </li>
    <li> 
        ใช้งานง่ายและตีความผลลัพธ์ได้ง่าย
    </li>
    <li> 
        รองรับข้อมูลที่ไม่เป็นเชิงเส้นตรง เมื่อแปลงข้อมูล
    </li>
    <p><strong> ข้อจำกัด </strong></p>  
    <li> 
        ต้องการความสัมพันธ์เชิงเส้นระหว่างตัวแปรต้นและ Log Odds
    </li>
    <li> 
        ไวต่อ Outliers
    </li>
    <li> 
        ไม่เหมาะกับข้อมูลที่มีความสัมพันธ์ซับซ้อน
    </li>
</ul>
""", unsafe_allow_html=True)

st.markdown("""   
<ul> 
    <p><strong> เกร็ดความรู้ Log Odds คืออะไร? </strong></p>
    <p>Log Odds (Logarithmic Odds) คือค่าลอการิทึมของอัตราส่วนระหว่าง ความน่าจะเป็นที่จะเกิดเหตุการณ์ (p) และ ความน่าจะเป็นที่จะไม่เกิดเหตุการณ์ (1-p) โดยค่านี้เป็นพื้นฐานของ Logistic Regression ในการวิเคราะห์เชิงสถิติ</p>
</ul>
""", unsafe_allow_html=True)
st.write("การคำนวณ Log Odds")
st.latex(r"""
\text{Log Odds} = \ln\left(\frac{p}{1-p}\right)
""")
st.markdown("""   
<ul> 
    <li>p : ความน่าจะเป็นที่เหตุการณ์จะเกิด (Probability of Success)</li>
    <li>1-p : ความน่าจะเป็นที่เหตุการณ์จะไม่เกิด (Probability of Failure)</li>
    <p><strong>ความหมายในเชิงสถิติ</strong></p>
    <li><strong>Log Odds และความสัมพันธ์เชิงเส้น:</strong></li>
        <p>Logistic Regression ใช้ Log Odds แทนค่าความน่าจะเป็น (p) เพื่อสร้างความสัมพันธ์เชิงเส้นระหว่างตัวแปรต้น (Independent Variables) และผลลัพธ์ (Dependent Variable)</p>
    <li><strong>แปลง Log Odds เป็น Probability:</strong></li>
        <p>คุณสามารถแปลงค่ากลับจาก Log Odds เป็นความน่าจะเป็นด้วยฟังก์ชัน Sigmoid:</p>
</ul>
""", unsafe_allow_html=True)
st.latex(r"""
p = \frac{1}{1 + e^{-\text{Log Odds}}}
""")
st.write("###### ตัวอย่างการคำนวณ Log Odds")
st.latex(r"""
\text{Log Odds} = \ln\left(\frac{1-p}{p}\right) = \ln\left(\frac{0.2}{0.8}\right) = \ln(4) \approx 1.386
""")

st.write("เริ่มจากสมการ")

st.latex(r'''
\text{Log Odds} = \ln\left(\frac{1 - p}{p}\right)
''')
st.write("ใช้สมการ exponential เพื่อย้อนกลับ ln:")
st.latex(r'''
e^{\text{Log Odds}} = \frac{1 - p}{p}
''')

st.latex(r'''
p = \frac{1}{1 + e^{\text{Log Odds}}}
''')


st.write("ใน Logistic Regression, Log Odds มักเขียนในรูปของสมการเชิงเส้น:")
st.latex(r'''
\text{Log Odds} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n
''')
st.write("เมื่อแทน Log Odds ลงใน p")
st.latex(r'''
p = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n)}}
''')