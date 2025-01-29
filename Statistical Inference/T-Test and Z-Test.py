import streamlit as st


st.title('T-Test and Z-Test')
st.markdown("""
<p> Hypothesis Testing: การทดสอบสมมติฐาน</p>
        
<p>การทดสอบสมมติฐาน (Hypothesis Testing) 
    เป็นขั้นตอนเชิงสถิติที่ช่วยให้เราตัดสินใจจากข้อมูลตัวอย่างว่าข้อสมมติฐานเกี่ยวกับประชากรควรถูกปฏิเสธหรือยอมรับ 
    โดยแบ่งออกเป็นสองประเภทหลัก:
</p>
            
<ul>                    
    <li>
        Null Hypothesis (H0): สมมติฐานเริ่มต้น เช่น ไม่มีความแตกต่างระหว่างสองกลุ่ม
    </li>
    <li>
         Alternative Hypothesis (𝐻𝑎): สมมติฐานทางเลือก เช่น มีความแตกต่างระหว่างสองกลุ่ม
    </li>       
            
</ul> 
""", unsafe_allow_html=True)
st.subheader("ขั้นตอนในการทดสอบสมมติฐาน")
st.markdown("""
    
<ul>                    
    <li>
        1.ตั้งสมมติฐาน H0 และ Ha
    </li>
    <li>
        2.เลือกระดับนัยสำคัญ (α) เช่น 0.05
    </li>     
    <li>
        3.คำนวณค่าสถิติและ p-value จากข้อมูล
    </li>   
    <li>
        4.เปรียบเทียบ p-value กับ α:
        <ul>
            <li>
                ถ้า p-value < α: ปฏิเสธ H0
            </li>
            <li>
                ถ้า p-value >= α: ปฏิเสธ H0
            </li>
        </ul>
    </li>
            
</ul> 
""", unsafe_allow_html=True)
st.subheader("ประเภทของ Hypothesis Testing")
st.markdown("""
<p><strong> One-Sample t-test <strong></p>
<ul>                    
    <li>
        ใช้เพื่อตรวจสอบว่าค่าเฉลี่ยของตัวอย่างหนึ่งมีค่าแตกต่างจากค่าคงที่ (Population Mean) หรือไม่
    </li>       
</ul> 
""", unsafe_allow_html=True)

st.write("from scipy.stats import ttest_1samp")
from scipy.stats import ttest_1samp
st.write("data = [68, 70, 72, 65, 76, 74, 72, 68, 70, 69]")
data = [68, 70, 72, 65, 76, 74, 72, 68, 70, 69]

st.write("stat, p_value = ttest_1samp(data, 70)")
stat, p_value = ttest_1samp(data, 70)
st.write(f"t-statistic: {stat:.4f}, p-value: {p_value:.4f}")

if p_value <0.05:
    st.write("p_value < 0.05")
    st.write("Reject H0: น้ำหนักเฉลี่ยแตกต่างจาก 70 กิโลกรัม")
else:
    st.write("p_value >= 0.05")
    st.write("Fail to Reject H0: น้ำหนักเฉลี่ยไม่แตกต่างจาก 70 กิโลกรัม")


st.markdown("""
<p><strong> Independent t-test <strong></p>
<ul>                    
    <li>
        ใช้เปรียบเทียบค่าเฉลี่ยระหว่างสองกลุ่มอิสระกัน
    </li>       
</ul> 
""", unsafe_allow_html=True)


st.write("from scipy.stats import ttest_ind")
from scipy.stats import ttest_ind
st.write("group1 = [82, 85, 87, 90, 88]")
st.write("group2 = [78, 75, 80, 85, 82]")
group1 = [82, 85, 87, 90, 88]
group2 = [78, 75, 80, 85, 82]

st.write("stat, p_value = ttest_ind(group1, group2)")
stat, p_value = ttest_ind(group1, group2)
st.write(f"t-statistic: {stat:.4f}, p-value: {p_value:.4f}")

if p_value <0.05:
    st.write("p_value < 0.05")
    st.write("Reject H0: คะแนนเฉลี่ยของสองกลุ่มแตกต่างกัน")
else:
    st.write("p_value >= 0.05")
    st.write("Fail to Reject H0: คะแนนเฉลี่ยของสองกลุ่มไม่แตกต่างกัน")



st.markdown("""
<p><strong> Paired t-test <strong></p>
<ul>                    
    <li>
        ใช้เปรียบเทียบค่าเฉลี่ยระหว่างสองกลุ่มที่เกี่ยวข้องกัน (Dependent)
    </li>       
</ul> 
""", unsafe_allow_html=True)


st.write("from scipy.stats import ttest_rel")
from scipy.stats import ttest_rel
st.write("before = [65, 70, 72, 68, 74]")
st.write("after = [70, 75, 78, 72, 80]")
before = [65, 70, 72, 68, 74]
after = [70, 75, 78, 72, 80]

st.write("stat, p_value = ttest_rel(before, after)")
stat, p_value = ttest_rel(before, after)
st.write(f"t-statistic: {stat:.4f}, p-value: {p_value:.4f}")

if p_value <0.05:
    st.write("p_value < 0.05")
    st.write("Reject H0: คะแนนก่อนและหลังแตกต่างกัน")
else:
    st.write("p_value >= 0.05")
    st.write("Fail to Reject H0: คะแนนก่อนและหลังไม่แตกต่างกัน")





st.markdown("""
<p><strong> Chi-Square Test <strong></p>
<ul>                    
    <li>
        ใช้ตรวจสอบความสัมพันธ์ระหว่างสองตัวแปรเชิงหมวดหมู่ (Categorical Variables)
    </li>       
</ul> 
""", unsafe_allow_html=True)


st.write("import numpy as np")
st.write("from scipy.stats import chi2_contingency")

import numpy as np
from scipy.stats import chi2_contingency

st.write("data = np.array([[50, 30], [20, 100]])")
data = np.array([[50, 30], [20, 100]])

st.write("stat, p_value, dof, expected = chi2_contingency(data)")
stat, p_value, dof, expected = chi2_contingency(data)
st.write(f"t-statistic: {stat:.4f}, p-value: {p_value:.4f}")

if p_value <0.05:
    st.write("p_value < 0.05")
    st.write("Reject H0: มีความสัมพันธ์ระหว่างตัวแปรทั้งสอง")
else:
    st.write("p_value >= 0.05")
    st.write("Fail to Reject H0: ไม่มีความสัมพันธ์ระหว่างตัวแปรทั้งสอง")






