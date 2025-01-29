import streamlit as st


st.title('MultipleLinear Regression')


st.markdown("""   
<ul> 
    <p> 
        Multiple Linear Regression (MLR)  เป็นเทคนิคทางสถิติที่ขยายจาก <strong>Simple Linear Regression</strong> โดยใช้ ตัวแปรต้น (Independent Variables) หลายตัวเพื่ออธิบายความสัมพันธ์และพยากรณ์ ตัวแปรตาม (Dependent Variable)
    </p>    
    <p>
        สมการของ Multiple Linear Regression มีลักษณะดังนี้:
    </p>                       
</ul>
""", unsafe_allow_html=True)
st.latex(r'''
y = b_0 + b_1x_1 + b_2x_2 + \dots + b_nx_n + \epsilon
''')
st.markdown("""   
<ul> 
    <li> 
        y : ตัวแปรตาม (Dependent Variable)
    </li>    
    <li> 
        x1,x2,x3, ... , xn : ตัวแปรต้น (Independent Variables)
    </li> 
    <li> 
        b0 : ค่าคงที่ : (Intercept)
    </li> 
    <li> 
        b1,b2,b3, ... , bn : ค่าสัมประสิทธิ์ (Coefficients) ของตัวแปรต้นแต่ละตัว
    </li> 
    <li> 
        e : ความผิดพลาด (Error Term)
    </li> 
                     
</ul>
""", unsafe_allow_html=True)