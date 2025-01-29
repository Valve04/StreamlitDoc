import streamlit as st


st.title('ANOVA')
st.write("from scipy.stats import f_oneway")
from scipy.stats import f_oneway
st.write("group1 = [82, 85, 87, 90, 88]")
st.write("group2 = [78, 75, 80, 85, 82]")
st.write("group3 = [92, 88, 91, 87, 89]")
group1 = [82, 85, 87, 90, 88]
group2 = [78, 75, 80, 85, 82]
group3 = [92, 88, 91, 87, 89]

st.write("stat, p_value = f_oneway(group1, group2, group3)")
stat, p_value = f_oneway(group1, group2, group3)
st.write(f"t-statistic: {stat:.4f}, p-value: {p_value:.4f}")

if p_value <0.05:
    st.write("p_value < 0.05")
    st.write("Reject H0: ค่าเฉลี่ยของสามกลุ่มแตกต่างกัน")
else:
    st.write("p_value >= 0.05")
    st.write("Fail to Reject H0: ค่าเฉลี่ยของสามกลุ่มไม่แตกต่างกัน")



st.markdown("""
<p><strong> ANOVA (Analysis of Variance) <strong></p>
   
<ul>                    
    <li>
        <strong>1. ไม่จำเป็นต้องมีข้อมูลเท่ากันในแต่ละกลุ่ม:<strong>
            <ul>
                <li>
                    ANOVA สามารถทำงานได้แม้ว่าข้อมูลในแต่ละกลุ่มจะมีจำนวนไม่เท่ากัน. 
                </li>
                <li>
                    อย่างไรก็ตาม หากกลุ่มมีจำนวนตัวอย่างแตกต่างกันมาก อาจส่งผลต่อความแม่นยำของผลลัพธ์ โดยเฉพาะอย่างยิ่งเมื่อข้อมูลมี ความแปรปรวนไม่เท่ากัน (Heteroscedasticity).
                </li>
            </ul>
    </li>   
    <li>
        2.สิ่งที่ต้องตรวจสอบก่อนใช้ ANOVA (เพื่อให้ผลลัพธ์เชื่อถือได้ ควรตรวจสอบเงื่อนไขเหล่านี้):
    </li> 
    <li>
        a. Homogeneity of Variance (ความแปรปรวนเท่ากัน):
            <ul>
                <li>
                    ความแปรปรวนของข้อมูลในแต่ละกลุ่มควรใกล้เคียงกัน. 
                </li>
                <li>
                    สามารถทดสอบได้ด้วย Levene's Test หรือ Bartlett's Test.
                </li>
            </ul>
    </li>   
    <li>
        b. Normality (ข้อมูลมีการแจกแจงแบบปกติ)
            <ul>
                <li>
                    ข้อมูลในแต่ละกลุ่มควรมีการแจกแจงแบบปกติ.
                </li>
                <li>
                    สามารถทดสอบด้วย Shapiro-Wilk Test หรือ Kolmogorov-Smirnov Test.
                </li>
            </ul>
    </li>   
    <li>
        3.ข้อควรระวังเมื่อตัวอย่างไม่เท่ากัน:
    </li> 
    <li>
        ความแปรปรวนไม่เท่ากัน (Heteroscedasticity)
            <ul>
                <li>
                    ถ้าข้อมูลมีจำนวนตัวอย่างต่างกันและความแปรปรวนไม่เท่ากัน อาจทำให้ ANOVA ให้ผลลัพธ์ที่ผิดพลาดได้
                </li>
                <li>
                    ในกรณีนี้ ควรใช้ Welch’s ANOVA ซึ่งเป็นเวอร์ชันของ ANOVA ที่ไม่ต้องการเงื่อนไขความแปรปรวนเท่ากัน.
                </li>
            </ul>
    </li>   
</ul> 
""", unsafe_allow_html=True)




st.write("Welch’s ANOVA")
st.write("from pingouin import welch_anova")

from pingouin import welch_anova
import pandas as pd


st.write(" data = pd.DataFrame({ 'score': [82, 85, 87, 90, 88, 78, 75, 80, 85, 82, 92, 88, 91, 87, 89],'group': [group1]*5 + [group2]*5 + [group3]*5})")



data = pd.DataFrame({
    'score': [82, 85, 87, 90, 88, 78, 75, 80, 85, 82, 92, 88, 91, 87, 89],
    'group': ['group1']*5 + ['group2']*5 + ['group3']*5
})
st.write(data)

anova_result = welch_anova(dv='score', between='group', data=data)
st.write(anova_result)

# st.write(f" ANOVA : {anova_result:.4f}")
st.write(anova_result[['p-unc']])


if p_value <0.05:
    st.write("p_value < 0.05")
    st.write("Reject H0: ความแปรปรวนต่างกัน")
else:
    st.write("p_value >= 0.05")
    st.write("Fail to Reject H0: ความแปรปรวนต่างกันไม่แตกต่างกัน")

