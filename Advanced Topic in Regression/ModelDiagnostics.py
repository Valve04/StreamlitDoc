


import streamlit as st


st.title('Model Diagnostics')
st.markdown("""   
<ul> 
    <p>
        <strong>Model Diagnostics</strong> คือการประเมินและตรวจสอบประสิทธิภาพของโมเดลที่ถูกฝึกฝน (trained model) 
        เพื่อให้มั่นใจว่าโมเดลนั้นทำงานได้ดีและเหมาะสมกับข้อมูลที่ใช้งาน การวิเคราะห์การทำงานของโมเดลสามารถช่วยระบุปัญหาหรือข้อบกพร่องในโมเดล 
        เช่น การ overfitting หรือ underfitting และแนะนำวิธีการปรับปรุงโมเดลให้ดีขึ้น
    </p>
    <p><strong>ขั้นตอนหลักของการทำ Model Diagnostics ได้แก่:</strong></p>
    <p><strong>1. การตรวจสอบความถูกต้องของโมเดล (Model Accuracy)</strong></p>
    <li>
        การประเมินว่าโมเดลสามารถทำนายผลลัพธ์ได้ถูกต้องแค่ไหน โดยการใช้ การแบ่งข้อมูลเป็นชุดการฝึก (training) และ ชุดการทดสอบ (test) เพื่อตรวจสอบว่าผลลัพธ์ที่ได้จากโมเดลตรงกับค่าจริงในชุดทดสอบ
    </li>
    <li>
        ตัวชี้วัดที่ใช้: Accuracy, Precision, Recall, F1-score (สำหรับ classification), MSE (Mean Squared Error) สำหรับ regression models
    </li>
    <p><strong>2. การตรวจสอบการ overfitting และ underfitting</strong></p>
    <li>
        Overfitting: โมเดลเรียนรู้รายละเอียดของข้อมูลในชุดฝึกมากเกินไปจนไม่สามารถ generalize ข้อมูลใหม่ได้ดี
    </li>
    <li>
        Underfitting: โมเดลไม่สามารถเรียนรู้ลักษณะสำคัญในข้อมูลได้ ซึ่งทำให้โมเดลมีความแม่นยำต่ำทั้งในชุดฝึกและชุดทดสอบ
    </li>
    <li>
        วิธีการตรวจสอบ: เปรียบเทียบ Loss Curve หรือ Error Curve ระหว่างชุดฝึกและชุดทดสอบ หาก loss ในชุดฝึกต่ำและในชุดทดสอบสูงแสดงว่าโมเดล overfitting
    </li>
    <p><strong>3. การตรวจสอบการกระจายของ Residuals (สำหรับ Regression Models)</strong></p>
    <li>
        Residual คือ ความแตกต่างระหว่างค่าที่ทำนายได้จากโมเดลกับค่าจริง (จริง - ค่าทำนาย)
    </li>
    <li>
        Residual Plot: การ plot ค่า residuals บนกราฟสามารถช่วยให้เห็นว่าโมเดลนั้นมีปัญหาในการ fitting หรือไม่ หาก residuals กระจายตัวอย่างไม่เป็นระเบียบแสดงว่าโมเดลอาจไม่เหมาะสม
    </li>
    <p><strong>4. การตรวจสอบ Multicollinearity (สำหรับ Linear Models)</strong></p>
    <li>
        Multicollinearity คือ เมื่อคุณสมบัติ (features) หลายๆ ตัวมีความสัมพันธ์กันมาก ซึ่งอาจทำให้โมเดลไม่สามารถแยกแยะความสำคัญของคุณสมบัติแต่ละตัวได้
    </li>
    <li>
        วิธีการตรวจสอบ: ใช้ Variance Inflation Factor (VIF) หรือ Correlation Matrix เพื่อตรวจสอบความสัมพันธ์ระหว่าง features
    </li>
    <p><strong>5. การใช้ Cross-Validation</strong></p>
    <li>
        Cross-Validation เป็นเทคนิคในการประเมินโมเดลโดยการแบ่งข้อมูลเป็นหลายๆ ส่วน และฝึกฝนโมเดลกับชุดข้อมูลที่ต่างกันในแต่ละรอบ การใช้ cross-validation ช่วยให้มั่นใจว่าโมเดลมีการ generalize ที่ดีและไม่ overfit กับชุดข้อมูลบางชุด
    </li>
    <li>
        วิธีการที่นิยม: K-fold cross-validation
    </li>
    <p><strong>6.การตรวจสอบการกระจายของข้อมูล (Data Distribution)</strong></p>
    <li>
        ตรวจสอบว่าข้อมูลที่ใช้ในการฝึกฝนและทดสอบมีการกระจายตัว (distribution) ที่เหมาะสมหรือไม่ เช่น ตรวจสอบว่าค่าของ feature หรือ target อยู่ในช่วงที่เหมาะสม
    </li>
    <li>
        ใช้ Box Plots, Histograms, หรือ Kernel Density Estimation (KDE) เพื่อ visualise ข้อมูล
    </li>      
    <p><strong>7.การใช้ Learning Curves</strong></p>
    <li>
        Learning Curves เป็นกราฟที่แสดงค่า error (หรือ loss) ของโมเดลเมื่อฝึกฝนไปเรื่อยๆ โดยแสดงทั้งค่าจากชุดฝึกและชุดทดสอบ เพื่อดูว่าโมเดลเรียนรู้ได้ดีแค่ไหนและจุดที่อาจเกิดปัญหา overfitting หรือ underfitting
    </li>
    <p><strong>8. การประเมินความสามารถในการทำนาย (Predictive Power)</strong></p>
    <li>
        สำหรับ classification models ใช้ Confusion Matrix เพื่อดูว่าโมเดลสามารถจำแนกข้อมูลแต่ละประเภทได้ถูกต้องมากน้อยแค่ไหน
    </li>
    <li>
        ใช้ ROC Curve และ AUC (Area Under the Curve) ในการประเมินว่าโมเดลสามารถแยกแยะระหว่างกลุ่มได้ดีแค่ไหน
    </li>
</ul>
""", unsafe_allow_html=True)

st.write("### 1. การตรวจสอบความถูกต้องของโมเดล (Model Accuracy)")
st.write("#### 1. สำหรับ Classification")


code = """
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from sklearn.datasets import load_iris
import streamlit as st

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = df['target'].apply(lambda x: iris.target_names[x])

# Select features (X) and target (y)
X = df.drop(columns=['target', 'target_name'])  # Features: All columns except 'target' and 'target_name'
y = df['target']  # Target: 'target'

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=200)  # Increase iterations to ensure convergence
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Weighted average for multi-class classification
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Display results in Streamlit
st.write(f"Accuracy: {accuracy * 100:.2f}%")
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1 Score: {f1:.2f}")

# Display Confusion Matrix with custom column and row names
cm = confusion_matrix(y_test, y_pred)

# Create a DataFrame for confusion matrix with custom column and row labels
cm_df = pd.DataFrame(cm, 
                     index=[name + ' test' for name in iris.target_names],  # Add 'test' to the row labels
                     columns=[name + ' pred' for name in iris.target_names])  # Add 'pred' to the column labels

# Show the confusion matrix
st.write("Confusion Matrix:")
st.write(cm_df)
st.write(y_test.value_counts())
st.write(pd.DataFrame(y_pred).value_counts())

"""

st.code(code, language='python')





from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from sklearn.datasets import load_iris
import streamlit as st

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = df['target'].apply(lambda x: iris.target_names[x])

# Select features (X) and target (y)
X = df.drop(columns=['target', 'target_name'])  # Features: All columns except 'target' and 'target_name'
y = df['target']  # Target: 'target'

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create and train the Logistic Regression model
model = LogisticRegression()  # Increase iterations to ensure convergence
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Weighted average for multi-class classification
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Display results in Streamlit
st.write(f"Accuracy: {accuracy * 100:.2f}%")
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1 Score: {f1:.2f}")

# Display Confusion Matrix with custom column and row names
cm = confusion_matrix(y_test, y_pred)

# Create a DataFrame for confusion matrix with custom column and row labels
# Create confusion matrix DataFrame with 'test' added to the target names
cm_df = pd.DataFrame(cm, 
                     index=[name + ' test' for name in iris.target_names],  # Add 'test' to the row labels
                     columns=[name + ' pred' for name in iris.target_names])  # Add 'pred' to the column labels

# Show the confusion matrix
st.write("Confusion Matrix:")
st.write(cm_df)
st.write('y_test')
st.write(y_test.value_counts())
st.write('y_pred')
st.write(pd.DataFrame(y_pred).value_counts())
# Correct way to concatenate lists of strings

# pd.DataFrame(y_pred)







st.write("#### 2. สำหรับ Regression")

code = """

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_diabetes
import pandas as pd

# Load diabetes dataset
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

# Separate the features (X) and target (y)
X = df.drop(columns='target')  # Features
y = df['target']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics: MSE, RMSE, MAE, R-squared
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R-squared: {r2:.2f}")


"""

st.code(code, language='python')



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_diabetes
import pandas as pd

# Load diabetes dataset
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

# Separate the features (X) and target (y)
X = df.drop(columns='target')  # Features
y = df['target']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics: MSE, RMSE, MAE, R-squared
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
st.write(f"MSE: {mse:.2f}")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"MAE: {mae:.2f}")
st.write(f"R-squared: {r2:.2f}")


st.write("#### 2. การตรวจสอบการ overfitting และ underfitting")



st.markdown("""   
<ul> 
    <p>
        การตรวจสอบการ overfitting และ underfitting คือการวิเคราะห์ผลลัพธ์ของโมเดลเพื่อดูว่าโมเดลทำงานได้ดีเพียงใดในข้อมูลฝึก 
        (training data) และข้อมูลทดสอบ (test data) โดยสามารถใช้กราฟหรือค่าประสิทธิภาพต่างๆ 
        เช่น Accuracy, Loss, MSE เพื่อช่วยในการตัดสินใจ
    </p>
    <p><strong>วิธีการตรวจสอบ Overfitting และ Underfitting:</strong></p>
    <h5><strong>1. Overfitting:</strong></h5>
    <li>
        โมเดลจะทำงานได้ดีมากกับชุดข้อมูลฝึก แต่เมื่อทดสอบกับชุดข้อมูลทดสอบ โมเดลจะไม่สามารถทำงานได้ดี 
        ซึ่งหมายถึงโมเดลเรียนรู้รายละเอียดที่ไม่จำเป็นในชุดข้อมูลฝึกเกินไป (overly complex) 
        โดยที่ไม่สามารถทั่วไปได้ดี (generalize).
    </li>
    <p><strong>2. Underfitting:</strong></p>
    <li>
        โมเดลไม่สามารถจับลักษณะของข้อมูลได้แม้ในชุดข้อมูลฝึก 
        โมเดลนี้มักจะเป็นโมเดลที่ไม่ซับซ้อนหรือไม่สามารถเรียนรู้ข้อมูลได้เพียงพอ.
    </li>
    <h5><strong>การตรวจสอบ Overfitting และ Underfitting ใน Python</strong></h5>
    <p><strong>สามารถทำได้โดยการเปรียบเทียบประสิทธิภาพระหว่างชุดข้อมูลฝึกและชุดข้อมูลทดสอบ</strong></p>
    <h5><strong> ตัวอย่างโค้ดการตรวจสอบ Overfitting และ Underfitting โดยใช้ Linear Regression: </strong></h5>
</ul>
""", unsafe_allow_html=True)

code = ("""
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

# โหลดข้อมูล diabetes
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# แบ่งข้อมูลเป็นชุดฝึก (train) และชุดทดสอบ (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล
model = LinearRegression()

# ฝึกโมเดลกับชุดข้อมูลฝึก
model.fit(X_train, y_train)

# ทำนายค่าผลลัพธ์
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# คำนวณค่า MSE สำหรับชุดข้อมูลฝึกและชุดข้อมูลทดสอบ
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# แสดงผลลัพธ์
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

# การแสดงกราฟเพื่อช่วยตรวจสอบ Overfitting และ Underfitting
plt.plot(y_train, y_train_pred, 'bo', label='Train Data')
plt.plot(y_test, y_test_pred, 'ro', label='Test Data')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.legend()
plt.title('Overfitting and Underfitting Analysis')
plt.show()
""")
st.code(code, language='python')



import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

# โหลดข้อมูล diabetes
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# แบ่งข้อมูลเป็นชุดฝึก (train) และชุดทดสอบ (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล
model = LinearRegression()

# ฝึกโมเดลกับชุดข้อมูลฝึก
model.fit(X_train, y_train)

# ทำนายค่าผลลัพธ์
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# คำนวณค่า MSE สำหรับชุดข้อมูลฝึกและชุดข้อมูลทดสอบ
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_rmse = mean_squared_error(y_train, y_train_pred,squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred,squared=False)

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# แสดงผลลัพธ์
st.write(f"Training MSE: {train_mse:.2f}")
st.write(f"Test MSE: {test_mse:.2f}")

st.write(f"Training rMSE: {train_rmse:.2f}")
st.write(f"Test rMSE: {test_rmse:.2f}")

st.write(f"Training MAE: {train_mae:.2f}")
st.write(f"Test MAE: {test_mae:.2f}")

st.write(f"Training r2: {train_r2:.2f}")
st.write(f"Test r2: {test_r2:.2f}")


# การแสดงกราฟเพื่อช่วยตรวจสอบ Overfitting และ Underfitting
plt.plot(y_train, y_train_pred, 'bo', label='Train Data')
plt.plot(y_test, y_test_pred, 'ro', label='Test Data')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.legend()
plt.title('Overfitting and Underfitting Analysis')
st.pyplot(plt)

st.markdown("""  
<ul>
    <h5><storng>คำอธิบาย</storng></h5>
    <p>
        1. Train MSE: คำนวณค่า Mean Squared Error (MSE) สำหรับชุดข้อมูลฝึก (training set)
    </p>
    <p>
        2. Test MSE: คำนวณค่า Mean Squared Error (MSE) สำหรับชุดข้อมูลทดสอบ (test set)
    </p>
    <p>
        3. กราฟแสดงการทำนาย: กราฟนี้จะแสดงค่าทำนายจากชุดข้อมูลฝึก (สีน้ำเงิน) และชุดข้อมูลทดสอบ 
        (สีแดง) เปรียบเทียบกับค่าจริง (True Values) เพื่อให้เห็นได้ว่าโมเดลทำงานได้ดีในข้อมูลฝึกหรือไม่
    </p>
    <h5><storng>การตรวจสอบ Overfitting และ Underfitting จากผลลัพธ์</storng></h5>
    <li>
        Overfitting: หากค่า Train MSE ต่ำกว่าค่า Test MSE อย่างมีนัยสำคัญ หรือกราฟการทำนายแสดงว่ามีความแม่นยำสูงในข้อมูลฝึกแต่มีความผิดพลาดสูงในข้อมูลทดสอบ แสดงว่าโมเดลอาจจะเกิดการ overfitting
    </li>
    <li>
        Underfitting: หากทั้ง Train MSE และ Test MSE มีค่าสูงและใกล้เคียงกัน หรือกราฟการทำนายแสดงให้เห็นว่าโมเดลไม่สามารถจับลักษณะข้อมูลได้ดีทั้งในข้อมูลฝึกและข้อมูลทดสอบ แสดงว่าโมเดลอาจจะเกิดการ underfitting
    </li>
    <h5>วิธีการแก้ไข</h5>
    <p>Overfitting</p>
        <li>ใช้เทคนิค Regularization เช่น Ridge Regression หรือ Lasso Regression</li>
        <li>ใช้ข้อมูลเพิ่มเติม (data augmentation) หรือการปรับขนาดข้อมูล (feature engineering)</li>
        <li>ลดความซับซ้อนของโมเดล</li>
    <p>Underfitting</p>
        <li>ใช้โมเดลที่ซับซ้อนขึ้น</li>
        <li>พิ่มข้อมูลการฝึกหรือเพิ่มจำนวนฟีเจอร์</li>
</ul>
"""
,unsafe_allow_html=True)

st.write("### 3. การตรวจสอบการกระจายของ Residuals (สำหรับ Regression Models)")

st.markdown("""   
<ul> 
    <p>
        การตรวจสอบการกระจายของ Residuals เป็นขั้นตอนสำคัญใน Regression Analysis 
        เพื่อตรวจสอบว่าโมเดลเหมาะสมกับข้อมูลหรือไม่ Residuals คือค่าความแตกต่างระหว่างค่าจริง (Yture)
        และค่าที่โมเดลทำนาย (Ypred) ซึ่งสามารถแสดงผลได้ในกราฟเพื่อวิเคราะห์ความสัมพันธ์และรูปแบบต่าง ๆ
    </p>
    <h5><strong>จุดประสงค์ของการตรวจสอบ Residuals:</strong></h5>
    <p><strong>1.ตรวจสอบความเป็นสุ่ม</strong>: Residuals ควรมีการกระจายแบบสุ่มรอบค่า 0 หากโมเดลเหมาะสม </p>
    <p><strong>2.ตรวจสอบ Homoscedasticity</strong>: ความแปรปรวนของ Residuals ควรมีความสม่ำเสมอ </p>
    <p><strong>3.ตรวจสอบ Normality</strong>: การกระจายของ Residuals ควรมีลักษณะใกล้เคียงกับการกระจายแบบปกติ (Normal Distribution)</p>         
</ul>
""", unsafe_allow_html=True)

code = ("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# โหลดข้อมูล Diabetes Dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# แบ่งข้อมูลเป็นชุดฝึก (train) และชุดทดสอบ (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างและฝึกโมเดล Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# คำนวณค่าทำนาย (Predicted Values)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# คำนวณ Residuals
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

# การตรวจสอบการกระจายของ Residuals
plt.figure(figsize=(15, 5))

# Plot Residuals vs Fitted Values
plt.subplot(1, 3, 1)
plt.scatter(y_train_pred, train_residuals, alpha=0.7, label="Train Residuals")
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values (Train)")
plt.legend()

# Plot Histogram of Residuals
plt.subplot(1, 3, 2)
plt.hist(train_residuals, bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals (Train)")

# Q-Q Plot for Normality Test
from scipy.stats import probplot
plt.subplot(1, 3, 3)
probplot(train_residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot (Train Residuals)")

plt.tight_layout()
plt.show()

""")
st.code(code, language='python')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# โหลดข้อมูล Diabetes Dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# แบ่งข้อมูลเป็นชุดฝึก (train) และชุดทดสอบ (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างและฝึกโมเดล Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# คำนวณค่าทำนาย (Predicted Values)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# คำนวณ Residuals
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

# การตรวจสอบการกระจายของ Residuals
plt.figure(figsize=(15, 5))

# Plot Residuals vs Fitted Values
plt.subplot(1, 3, 1)
plt.scatter(y_train_pred, train_residuals, alpha=0.7, label="Train Residuals")
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values (Train)")
# plt.legend()


# Plot Histogram of Residuals
plt.subplot(1, 3, 2)
plt.hist(train_residuals, bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals (Train)")

# Q-Q Plot for Normality Test
from scipy.stats import probplot
plt.subplot(1, 3, 3)
probplot(train_residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot (Train Residuals)")

plt.tight_layout()
# plt.show()
st.pyplot(plt)

st.markdown("""   
<ul> 
    <p>
        การตรวจสอบ Multicollinearity เป็นการตรวจสอบว่าฟีเจอร์ในโมเดลมีความสัมพันธ์กันมากเกินไปหรือไม่ 
        ซึ่งอาจส่งผลกระทบต่อความน่าเชื่อถือของค่าประมาณค่าสัมประสิทธิ์ใน Linear Models เช่น Linear Regression
    </p>
    <h5>
        ทำไมต้องตรวจสอบ Multicollinearity? 
    </h5>
        <li>ปัญหาในการตีความโมเดล: หากฟีเจอร์มีความสัมพันธ์กันสูง อาจทำให้การตีความค่าสัมประสิทธิ์ผิดพลาด</li>
        <li>ความไม่เสถียรของโมเดล: Multicollinearity อาจทำให้โมเดลไม่เสถียรเมื่อข้อมูลเปลี่ยนแปลงเล็กน้อย</li>
        <li>ลดประสิทธิภาพของโมเดล: การใช้ฟีเจอร์ที่สัมพันธ์กันมากเกินไปอาจไม่ได้ช่วยเพิ่มความแม่นยำ</li>
    <h5>
        วิธีการตรวจสอบ Multicollinearity
    </h5>
        <li>Correlation Matrix: ใช้ดูความสัมพันธ์ระหว่างฟีเจอร์</li>
        <li>Variance Inflation Factor (VIF): วัดว่าฟีเจอร์หนึ่งสามารถคาดการณ์ได้ดีแค่ไหนโดยใช้ฟีเจอร์อื่น ๆ</li>
</ul>
""", unsafe_allow_html=True)

code = ("""
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt

# โหลดข้อมูล Diabetes Dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

# 1. ตรวจสอบ Correlation Matrix
corr_matrix = X.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()

# 2. คำนวณ Variance Inflation Factor (VIF)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# แสดงผลลัพธ์ VIF
print(vif_data)

""")
st.code(code,language='python')

import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt

# โหลดข้อมูล Diabetes Dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

# 1. ตรวจสอบ Correlation Matrix
corr_matrix = X.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()
st.pyplot(plt)
# 2. คำนวณ Variance Inflation Factor (VIF)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# แสดงผลลัพธ์ VIF
st.write(vif_data)

st.markdown("""   
<ul> 
    <h5>
        การวิเคราะห์ผลลัพธ์:
    </h5>
    <p><storng> 1.Correlation Matrix:</storng></p>
    <li>หากค่าความสัมพันธ์ (Correlation) ระหว่างฟีเจอร์ใด ๆ มีค่า มากกว่า 0.8 หรือ น้อยกว่า -0.8 แสดงว่าฟีเจอร์นั้นมีความสัมพันธ์กันสูง</li>
    <li>ควรพิจารณาลดหรือรวมฟีเจอร์ที่มีความสัมพันธ์กัน</li>
    <p><storng> 2.Variance Inflation Factor (VIF):</storng></p>
    <li>VIF วัดว่า ฟีเจอร์หนึ่งสามารถคาดการณ์ได้จากฟีเจอร์อื่น ๆ มากแค่ไหน</li>
    <li>ค่าของ VIF:
    <ul>
        <li>VIF < 5: ไม่มี Multicollinearity</li>
        <li>5 ≤ VIF < 10: มี Multicollinearity ในระดับปานกลาง</li>
        <li>VIF ≥ 10: มี Multicollinearity สูง (ควรปรับปรุงโมเดล)</li>
    </ul>
    </li>
</ul>
""", unsafe_allow_html=True)

st.write("#### 5.การใช้ Cross-Validation")

st.markdown("""   
<ul> 
    <p>การใช้ Cross-Validation เป็นเทคนิคสำหรับประเมินประสิทธิภาพของโมเดลโดยแบ่งข้อมูลออกเป็นชุดย่อย (folds) เพื่อให้มั่นใจว่าโมเดลไม่ได้ overfitting หรือ underfitting</p>
    <h5>ทำไมต้องใช้ Cross-Validation?</h5>
    <li>1.ช่วยประเมินโมเดลได้อย่างแม่นยำ</li>
    <li>2.ใช้ข้อมูลให้เกิดประโยชน์สูงสุด</li>
    <li>3.ลดความเสี่ยงของการประเมินที่เอนเอียงจากการสุ่มแบ่งข้อมูล</li>
    <h5>ประเภทของ Cross-Validation</h5>
    <p>1.K-Fold Cross-Validation:</p>
    <li>แบ่งข้อมูลเป็น K ส่วน</li>
    <li>ใช้ (K-1) ส่วนสำหรับฝึก และ 1 ส่วนสำหรับทดสอบ</li>
    <li>ทำซ้ำ K รอบ</li>
    <p>2.Stratified K-Fold Cross-Validation:</p>
    <li>คล้าย K-Fold แต่รักษาสัดส่วนของ class (สำหรับ classification)</li>
    <p>Leave-One-Out Cross-Validation (LOOCV):</p>
    <li>ใช้ข้อมูล 1 ตัวอย่างสำหรับทดสอบ และส่วนที่เหลือสำหรับฝึก</li>
    <h5>ตัวอย่างการใช้ K-Fold Cross-Validation ด้วย Python:</h5>
</ul>
""", unsafe_allow_html=True)

code = ("""
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import numpy as np

# โหลดข้อมูล
iris = load_iris()
X = iris.data
y = iris.target

# สร้างโมเดล Logistic Regression
model = LogisticRegression(max_iter=200)

# กำหนด K-Fold Cross-Validation (แบ่งเป็น 5 ส่วน)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# ประเมินโมเดลด้วย Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

# แสดงผลลัพธ์
print("Cross-Validation Scores:", cv_scores)
print(f"Mean Accuracy: {cv_scores.mean():.2f}")
print(f"Standard Deviation: {cv_scores.std():.2f}")
""")
st.code(code,language='python')

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import numpy as np

# โหลดข้อมูล
iris = load_iris()
X = iris.data
y = iris.target

# สร้างโมเดล Logistic Regression
model = LogisticRegression(max_iter=200)

# กำหนด K-Fold Cross-Validation (แบ่งเป็น 5 ส่วน)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# ประเมินโมเดลด้วย Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

# แสดงผลลัพธ์
st.write("Cross-Validation Scores:", cv_scores)
st.write(f"Mean Accuracy: {cv_scores.mean():.2f}")
st.write(f"Standard Deviation: {cv_scores.std():.2f}")



st.markdown("""   
<ul> 
    <h5>ตัวอย่างการใช้ Stratified K-Fold Cross-Validation:</h5>
</ul>
""", unsafe_allow_html=True)

code = ("""
from sklearn.model_selection import StratifiedKFold

# กำหนด Stratified K-Fold Cross-Validation (แบ่งเป็น 5 ส่วน)
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ประเมินโมเดลด้วย Stratified K-Fold
cv_scores = cross_val_score(model, X, y, cv=stratified_kfold, scoring='accuracy')

# แสดงผลลัพธ์
print("Stratified Cross-Validation Scores:", cv_scores)
print(f"Mean Accuracy: {cv_scores.mean():.2f}")

""")
st.code(code,language='python')

from sklearn.model_selection import StratifiedKFold

# กำหนด Stratified K-Fold Cross-Validation (แบ่งเป็น 5 ส่วน)
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ประเมินโมเดลด้วย Stratified K-Fold
cv_scores = cross_val_score(model, X, y, cv=stratified_kfold, scoring='accuracy')

# แสดงผลลัพธ์
st.write("Stratified Cross-Validation Scores:", cv_scores)
st.write(f"Mean Accuracy: {cv_scores.mean():.2f}")

st.markdown("""   
<ul> 
    <h5>การวิเคราะห์ผลลัพธ์:</h5>
    <p>1.Cross-Validation Scores:</p>
    <li>เป็นค่าความแม่นยำของแต่ละ fold</li>
    <li>ใช้สำหรับตรวจสอบว่าคะแนนในแต่ละ fold ใกล้เคียงกันหรือไม่</li>
    <p>2.Mean Accuracy:</p>
    <li>ค่าเฉลี่ยของคะแนนจากทุก fold</li>
    <li>เป็นตัวชี้วัดความแม่นยำโดยรวมของโมเดล</li>
    <p>3.Standard Deviation:</p>
    <li>ค่าความแปรปรวนของคะแนน</li>
    <li>หากมีค่าสูง แสดงว่าโมเดลอาจทำงานไม่เสถียรในแต่ละ fold</li>
    <h5>การปรับปรุงโมเดลด้วย Cross-Validation:</h5>
    <li>เลือกค่าพารามิเตอร์ที่เหมาะสม (Hyperparameter Tuning)</li>
    <li>ตรวจสอบว่าโมเดล overfitting หรือ underfitting</li>
    <li>เลือกประเภทของ Cross-Validation ที่เหมาะสมกับข้อมูล เช่น Stratified K-Fold สำหรับ classification</li>
    <p><strong>Cross-Validation เป็นเครื่องมือสำคัญที่ช่วยเพิ่มความน่าเชื่อถือในการวัดประสิทธิภาพของโมเดล!</strong></p>
</ul>
""", unsafe_allow_html=True)



st.write("#### 6.การตรวจสอบการกระจายของข้อมูล (Data Distribution)")
st.markdown("""   
<ul> 
    <p>การตรวจสอบการกระจายของข้อมูล (Data Distribution) เป็นขั้นตอนสำคัญในกระบวนการวิเคราะห์ข้อมูลเพื่อเข้าใจลักษณะของข้อมูล เช่น ความสมมาตร, ความเบ้ (Skewness), และการกระจายตัว (Dispersion)</p>
</ul>
""", unsafe_allow_html=True)

st.markdown("""   
<ul> 
    <h5>วิธีการตรวจสอบการกระจายของข้อมูล:</h5>
    <li>Histogram: ใช้ดูการกระจายของค่าตัวเลข</li>
    <li>Box Plot: ใช้ดูข้อมูลที่เป็นค่าผิดปกติ (Outliers)</li>
    <li>Density Plot: ใช้ดูการกระจายเชิงความน่าจะเป็น</li>
    <li>Q-Q Plot: ใช้ตรวจสอบว่าข้อมูลมีการกระจายแบบปกติหรือไม่</li>
    <li>Summary Statistics: ใช้ดูค่าเฉลี่ย (Mean), มัธยฐาน (Median), และค่าเบี่ยงเบนมาตรฐาน (Standard Deviation)</li>
    <h5>ตัวอย่างการใช้ Python สำหรับการตรวจสอบการกระจายของข้อมูล</h5>
    <li>1. ตรวจสอบด้วย Histogram</li>
</ul>
""", unsafe_allow_html=True)

code = ("""
import matplotlib.pyplot as plt
import numpy as np

# สร้างข้อมูลตัวอย่าง
data = np.random.normal(loc=50, scale=10, size=1000)

# สร้าง Histogram
plt.hist(data, bins=20, color='blue', alpha=0.7)
plt.title('Histogram of Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
""")
st.code(code,language='python')

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Generate random data
data = np.random.normal(loc=50, scale=10, size=1000)
plt.figure()
# Create a histogram
plt.hist(data, bins=20, color='blue', alpha=0.7)
plt.title('Histogram of Data')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Display plot in Streamlit
st.pyplot(plt)  # No need to use plt.show()

st.markdown("""   
<ul> 
    <li>2. ตรวจสอบด้วย Box Plot</li>
</ul>
""", unsafe_allow_html=True)


code = ("""
import seaborn as sns
plt.figure()
# สร้าง Box Plot
sns.boxplot(x=data, color='cyan')
plt.title('Box Plot of Data')
plt.show()
st.pyplot(plt)
""")
st.code(code,language='python')

import seaborn as sns
plt.figure()
# สร้าง Box Plot
sns.boxplot(x=data, color='cyan')
plt.title('Box Plot of Data')
plt.show()
st.pyplot(plt)


st.markdown("""   
<ul> 
    <li>3. ตรวจสอบด้วย Density Plot</li>
</ul>
""", unsafe_allow_html=True)
code = ("""
st.code(code,language='python')
# สร้าง Density Plot
plt.figure()
sns.kdeplot(data, shade=True, color='green')
plt.title('Density Plot of Data')
plt.xlabel('Value')
plt.show()
st.pyplot(plt)
""")
st.code(code,language='python')
# สร้าง Density Plot
plt.figure()
sns.kdeplot(data, shade=True, color='green')
plt.title('Density Plot of Data')
plt.xlabel('Value')
plt.show()
st.pyplot(plt)


st.markdown("""   
<ul> 
    <li>4. ตรวจสอบด้วย Q-Q Plot</li>
</ul>
""", unsafe_allow_html=True)

code = ("""
import scipy.stats as stats
# สร้าง Q-Q Plot
plt.figure()
stats.probplot(data, dist="norm", plot=plt)

plt.title('Q-Q Plot')
st.pyplot(plt)
""")
st.code(code,language='python')
import scipy.stats as stats
# สร้าง Q-Q Plot
plt.figure()
stats.probplot(data, dist="norm", plot=plt)

plt.title('Q-Q Plot')
st.pyplot(plt)



st.markdown("""   
<ul> 
    <li>5. ตรวจสอบ Summary Statistics</li>
</ul>
""", unsafe_allow_html=True)
code = ("""
import pandas as pd
# คำนวณค่าทางสถิติ
df = pd.DataFrame(data, columns=['Value'])
summary_stats = df.describe()
# แสดงผลลัพธ์
st.write(summary_stats)
""")
st.code(code,language='python')
import pandas as pd
# คำนวณค่าทางสถิติ
df = pd.DataFrame(data, columns=['Value'])
summary_stats = df.describe()
# แสดงผลลัพธ์
st.write(summary_stats)


st.markdown("""   
<ul> 
    <h5>การแปลผล</h5>
    <li>1.Histogram:
        <ul>
            <li>รูปร่างที่สมมาตรบ่งชี้ถึงการกระจายปกติ</li>
            <li>รูปร่างเบ้ซ้ายหรือขวาแสดงถึงการกระจายที่ไม่สมมาตร</li>
        </ul>
    </li>
    <li>2.Box Plot:
        <ul>
            <li>ดูค่าผิดปกติ (Outliers) หากมีจุดที่อยู่นอกกล่อง</li>
        </ul>
    </li>
    <li>3.Density Plot:
        <ul>
            <li>การกระจายที่เรียบและมียอดเดียว (Unimodal) มักสัมพันธ์กับการกระจายปกติ</li>
        </ul>
    </li>
    <li>4.Q-Q Plot:
        <ul>
            <li>จุดที่อยู่ในแนวเส้นทแยงมุมบ่งชี้ว่าข้อมูลมีการกระจายแบบปกติ</li>
        </ul>
    </li>
    <li>5.Summary Statistics:
        <ul>
            <li>ค่าเฉลี่ยและมัธยฐานที่ใกล้เคียงกันแสดงว่าข้อมูลมีการกระจายปกติ</li>
            <li>ค่าเบี่ยงเบนมาตรฐานบ่งบอกถึงการกระจายตัวของข้อมูล</li>
        </ul>
    </li>
    <h5>การใช้งานในสถานการณ์จริง</h5>
    <li>ใช้ตรวจสอบความเหมาะสมของข้อมูลก่อนการสร้างโมเดล</li>
    <li>ใช้กำจัดค่าผิดปกติหรือปรับปรุงการกระจายให้ใกล้เคียงปกติ (เช่น การทำ Log Transformation)</li>
    <p>การวิเคราะห์การกระจายของข้อมูลช่วยให้เราเข้าใจและเตรียมข้อมูลได้ดีขึ้นสำหรับการวิเคราะห์ขั้นสูง!</p>
</ul>
""", unsafe_allow_html=True)


st.write("#### 7.การใช้ Learning Curves")
st.markdown("""   
<ul> 
    <p>การใช้ Learning Curves ช่วยในการประเมินว่าโมเดลมีการเรียนรู้เพียงพอหรือไม่ รวมถึงการตรวจสอบปัญหา Overfitting หรือ Underfitting โดย Learning Curves จะแสดงกราฟที่แสดงความสัมพันธ์ระหว่างประสิทธิภาพของโมเดล (เช่น ค่า Accuracy หรือ Loss) กับจำนวนข้อมูลที่ใช้ในการฝึกฝน</p>

</ul>
""", unsafe_allow_html=True)


code = ("""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# โหลดข้อมูล Iris
data = load_iris()
X, y = data.data, data.target

# แบ่งข้อมูลเป็น train และ test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล
model = LogisticRegression(max_iter=200)

# ใช้ learning_curve
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

# คำนวณค่าเฉลี่ยและค่าเบี่ยงเบนมาตรฐานของ train/test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# สร้างกราฟ Learning Curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Validation Score')

# เพิ่มช่วงค่าเบี่ยงเบนมาตรฐาน (Standard Deviation)
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')

# ปรับแต่งกราฟ
plt.title('Learning Curve')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.show()

""")
st.code(code,language='python')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# โหลดข้อมูล Iris
data = load_iris()
X, y = data.data, data.target

# แบ่งข้อมูลเป็น train และ test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล
model = LogisticRegression(max_iter=200)

# ใช้ learning_curve
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)

# คำนวณค่าเฉลี่ยและค่าเบี่ยงเบนมาตรฐานของ train/test scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# สร้างกราฟ Learning Curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Validation Score')

# เพิ่มช่วงค่าเบี่ยงเบนมาตรฐาน (Standard Deviation)
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')

# ปรับแต่งกราฟ
plt.title('Learning Curve')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid()
plt.show()
st.pyplot(plt)


# Assume train_mean, train_std, test_mean, test_std are already calculated
data = {
    'Train Mean': train_mean,
    'Train Std': train_std,
    'Test Mean': test_mean,
    'Test Std': test_std
}

# Create the DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
st.write(df)
st.write(df.describe())


st.markdown("""   
<ul> 
    <p>1.learning_curve : ฟังก์ชันใน sklearn ที่คำนวณคะแนนของโมเดลในชุดข้อมูลที่มีขนาดต่าง ๆ เพื่อสร้าง Learning Curve</p>
        <ul>
            <li>train_sizes: สัดส่วนของข้อมูลการฝึก</li>
            <li>train_scores และ test_scores: ค่าคะแนน (เช่น Accuracy) ที่ได้จากการฝึกและทดสอบ</li>
        </ul>
    <p>2.train_mean และ test_mean : คำนวณค่าเฉลี่ยของคะแนนเพื่อแสดงผลเป็นเส้น Learning Curve</p>
    <p>3.plt.fill_between : ใช้แสดงช่วงค่าเบี่ยงเบนมาตรฐาน (Standard Deviation) เพื่อแสดงความมั่นใจในคะแนนของโมเดล</p>
    <h5>การวิเคราะห์กราฟ Learning Curve</h5>
    <li>Overfitting : คะแนนการฝึก (Training Score) สูง แต่คะแนนการตรวจสอบ (Validation Score) ต่ำ </li>
    <li>Underfitting : คะแนนการฝึกและตรวจสอบต่ำมาก และมีช่องว่างเล็กน้อยระหว่างสองเส้น </li>
    <li>Good Fit : คะแนนการฝึกและตรวจสอบสูง และสองเส้นเข้าใกล้กัน </li>
</ul>
""", unsafe_allow_html=True)


st.write("#### 8.การประเมินความสามารถในการทำนาย (Predictive Power)")
st.markdown("""   
<ul> 
    <p>การประเมินความสามารถในการทำนาย (Predictive Power) ช่วยวัดว่ารูปแบบการเรียนรู้ของโมเดลสามารถทำนายข้อมูลใหม่ได้ดีเพียงใด ซึ่งใช้เครื่องมือประเมินที่แตกต่างกันสำหรับโมเดลประเภท Regression และ Classification</p>
    <h5>1. สำหรับโมเดล Classification</h5>
        <li>Accuracy: วัดเปอร์เซ็นต์ของการทำนายถูกต้อง</li>
        <li>Precision, Recall, F1-Score: วัดความถูกต้องและประสิทธิภาพของการจำแนกประเภท</li>
        <li>ROC Curve และ AUC: วัดความสามารถของโมเดลในการแยกประเภทที่ถูกต้อง</li>
    
</ul>
""", unsafe_allow_html=True)


code = ("""
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# โหลดข้อมูลตัวอย่าง (Iris Dataset)
data = load_iris()
X, y = data.data, data.target

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ใช้ Logistic Regression
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ทำนายผล
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# ประเมินผล
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# ROC Curve สำหรับ Multi-class
fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)
auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
print(f"AUC Score: {auc_score:.2f}")

# วาด ROC Curve
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()

""")
st.code(code,language='python')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# โหลดข้อมูลตัวอย่าง (Iris Dataset)
data = load_iris()
X, y = data.data, data.target

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ใช้ Logistic Regression
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ทำนายผล
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# ประเมินผล
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
st.write(f"Accuracy: {accuracy:.2f}")
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1-Score: {f1:.2f}")

# ROC Curve สำหรับ Multi-class
fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)
auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
st.write(f"AUC Score: {auc_score:.2f}")

# วาด ROC Curve
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()
st.pyplot(plt)


































































st.markdown("""   
<ul> 
    <h5>1. สำหรับโมเดล Regression</h5>
        <li>Mean Squared Error (MSE): ค่าความผิดพลาดเฉลี่ยยกกำลังสอง</li>
        <li>Mean Absolute Error (MAE): ค่าความผิดพลาดเฉลี่ย</li>
        <li>R-squared (R²): วัดว่าตัวแปรอิสระสามารถอธิบายตัวแปรตามได้ดีแค่ไหน</li>
        <li>Adjusted R-squared: ปรับค่า R² ให้เหมาะสมกับจำนวนตัวแปร</li>
    
</ul>
""", unsafe_allow_html=True)


code = ("""
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes

# โหลดข้อมูล
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ใช้ Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# ทำนายผล
y_pred = model.predict(X_test)

# ประเมินผล
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R-squared: {r2:.2f}")
print(f"Adjusted R-squared: {adj_r2:.2f}")
""")
st.code(code,language='python')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes

# โหลดข้อมูล
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ใช้ Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# ทำนายผล
y_pred = model.predict(X_test)

# ประเมินผล
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

st.write(f"MSE: {mse:.2f}")
st.write(f"MAE: {mae:.2f}")
st.write(f"R-squared: {r2:.2f}")
st.write(f"Adjusted R-squared: {adj_r2:.2f}")


st.markdown("""   
<ul> 
    <h5>การวิเคราะห์ผลการประเมิน</h5>
    <p>1.โมเดล Classification:</p>
        <ul>
            <li>ค่า Precision/Recall สูง แสดงว่าการจำแนกประเภทมีความแม่นยำ</li>
            <li>ค่า AUC ใกล้ 1 หมายถึงโมเดลแยกประเภทได้ดี</li>
        </ul>
    <p>2.โมเดล Regression:</p>
        <ul>
            <li>ค่า MSE/MAE ต่ำแสดงว่าความผิดพลาดในการพยากรณ์น้อย</li>
            <li>ค่า R-squared ใกล้ 1 หมายถึงโมเดลอธิบายข้อมูลได้ดี</li>
        </ul>
</ul>
""", unsafe_allow_html=True)