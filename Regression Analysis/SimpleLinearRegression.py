import streamlit as st


st.title('SimpleLinear Regression')
st.subheader("Simple Linear Regression")

st.markdown("""   
<ul> 
    <p> 
        Simple Linear Regression เป็นเทคนิคทางสถิติและการเรียนรู้ของเครื่อง (Machine Learning) 
        ที่ใช้สำหรับการสร้างแบบจำลองความสัมพันธ์เชิงเส้นระหว่าง ตัวแปรต้น (Independent Variable) และ ตัวแปรตาม (Dependent Variable)
    </p>                   
    <p>
        <strong>วัตถุประสงค์ของ Simple Linear Regression</strong>
        <li>
            1.หาเส้นตรงที่เหมาะสมที่สุด (Best-fit Line) ซึ่งลดความแตกต่างระหว่างค่าจริง (y) และค่าที่คาดการณ์ได้
        </li>
        <li>
            2.ใช้เส้นตรงนี้ในการพยากรณ์ค่าของ y สำหรับค่า x ใหม่ ๆ
        </li>
    </p>
    <p>
        <strong>การนำไปใช้</strong>
        <p>Simple Linear Regression ใช้ในกรณีที่ต้องการพยากรณ์หรืออธิบายความสัมพันธ์</p>
        <li> 1.การคาดการณ์ราคาบ้านจากพื้นที่ </li>
        <li> 2.การวิเคราะห์ความสัมพันธ์ระหว่างการโฆษณากับยอดขาย </li>
        <li> 3.การคาดการณ์คะแนนสอบจากชั่วโมงการอ่านหนังสือ </li>
    </p>
            
</ul>
""", unsafe_allow_html=True)

st.write(" ")


st.markdown("""   
<ul>                    
    <li>
            Simple Linear Regression คือการวิเคราะห์ความสัมพันธ์ระหว่างตัวแปร 2 ตัว
        <ul>
            <li>
                ตัวแปรอิสระ (Independent Variable, X) 
            </li>
            <li>
                ตัวแปรตาม (Dependent Variable, Y)
            </li>
        </ul>
    </li>    
    <li>
        จุดประสงค์หลักของมันคือการหาความสัมพันธ์ในรูปแบบสมการเชิงเส้น:
        <ul>
            <li>
                Y=β0+β1X+ϵ
            </li>
            <li>
                Y : ค่าที่ต้องการทำนาย (ตัวแปรตาม)
            </li>
            <li>
                X : ตัวแปรอิสระ
            </li>
            <li>
                β0 : ค่าจุดตัดแกน Y (Intercept)
            </li>
            <li>
                β1 : ค่าสัมประสิทธิ์ความชัน (Slope)
            </li>
        </ul>
    </li> 
</ul>
""", unsafe_allow_html=True)



st.markdown("""   
            
<p><strong> ตัวอย่างการทำ Simple Linear Regression ใน Python <strong></p>
<ul>
    <li>สมมติว่ามีข้อมูลเกี่ยวกับชั่วโมงที่ใช้เรียน X และคะแนนสอบ Y เราต้องการสร้างโมเดลเพื่อพยากรณ์คะแนนสอบจากจำนวนชั่วโมงที่เรียน</li>
</ul>
<p><strong> 1.เตรียมข้อมูล <strong></p>
<ul>
    <p>import numpy as np</p>
    <p>import matplotlib.pyplot as plt</p>
    <p>from sklearn.linear_model import LinearRegression</p>
    <p>from sklearn.metrics import mean_squared_error, r2_score</p>
    <p>Example Data</p>
        <li> X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1) </li>
        <li> Y = np.array([2, 4, 5, 4, 5])  </li>
</ul>
            
<p><strong> 2. สร้างโมเดล <strong></p>
<ul>
    <p>model = LinearRegression()</p>
    <p>model.fit(X, Y)</p>
    <p>beta_0 = model.intercept_ </p>
    <p>beta_1 = model.coef_[0] </p>
    <p>print(f'Intercept (β₀): {beta_0}') </p>
    <p>print(f'Coefficient (β₁): {beta_1}') </p>
</ul>    
             
<p><strong> 3. ทำนายค่าและวิเคราะห์ผล <strong></p>
<ul>
    <p>Y_pred = model.predict(X) </p>
    <p>mse = mean_squared_error(Y, Y_pred) </p>
    <p>r2 = r2_score(Y, Y_pred) </p>
    <p>print(f"Mean Squared Error (MSE): {mse}") </p>      
    <p>print(f"R-squared (R²): {r2}") </p>
        <li> MSE (Mean Squared Error): ค่าความคลาดเคลื่อนเฉลี่ยระหว่างค่าจริงและค่าทำนาย </li>
        <li> R² (Coefficient of Determination): ค่าที่แสดงว่าสมการโมเดลอธิบายความแปรปรวนของ Y ได้มากน้อยแค่ไหน (ค่าอยู่ในช่วง 0 ถึง 1) </li>     
</ul>

<p><strong> 4. แสดงผลในกราฟ <strong></p>
<ul>
    <p>plt.scatter(X, Y, color="blue", label="Actual Data") # จุดข้อมูลจริง</p>
    <p>plt.plot(X, Y_pred, color="red", label="Regression Line")  # เส้นทำนาย </p>
    <p>plt.xlabel("Hours Studied") </p>
    <p>plt.ylabel("Exam Score")  </p>
    <p>plt.legend() </p>
    <p>plt.show() </p>
</ul>

            
<p><strong> ผลลัพธ์ที่ได้ <strong></p>
<ul>
    <li>โมเดลจะให้สมการเส้นตรง เช่น Y=2.2+0.6X หมายความว่า คะแนนสอบเพิ่มขึ้นโดยเฉลี่ย 0.6 คะแนนเมื่อเพิ่มชั่วโมงเรียน 1 ชั่วโมง </li>
    <li>ค่า 𝑅2 : แสดงว่าสมการนี้สามารถอธิบายความแปรปรวนของคะแนนสอบได้ ∼80% (ตัวเลขสมมุติ) </li>
</ul>          


<p><strong> ข้อสังเกต <strong></p>   
<ul>
    <p>1. ข้อสมมติของ Linear Regression: </p>
        <li> ความสัมพันธ์ระหว่าง X และ Y ต้องเป็นเชิงเส้น </li>
        <li> ความแปรปรวนของตัวแปรต้องคงที่ (Homoscedasticity) </li>
        <li> ตัวแปรตาม Y ควรมีการแจกแจงแบบปกติ </li>
    <p>2. ข้อจำกัด: </p>
        <li> Simple Linear Regression ใช้ได้เฉพาะตัวแปร X เดียว </li>
        <li> หากมีตัวแปรอิสระหลายตัว ควรใช้ Multiple Linear Regression </li>
</ul>   
""", unsafe_allow_html=True)

