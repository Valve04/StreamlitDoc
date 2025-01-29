import streamlit as st
import pandas as pd

st.title('Sampling Methods')

st.write("Sampling methods are strategies used to select a subset (sample) from a larger population to study and make inferences about the population. เป็นกลยุทธ์ที่ใช้ในการศึกษาและสรุปผลเกี่ยวกับประชากร")
st.write("Sampling is critical in statistics, data analysis, and research because studying an entire population is often impractical or impossible. มีความสำคัญเนื่องจากเราไม่สามารถศึกษาประชากรทั้งหมดได้")


st.markdown("""
<p>Importance of Sampling</p>
<ul>
    <li><strong>Feasibility (ความเป็นไปได้) </strong>: It's faster, less expensive, and more efficient than surveying an entire population. เร็วและประหยัดกว่าการใช้ทรัพยากรทั้งหมด</li>
    <li><strong>Accuracy (ความแม่นยำ,ความถูกต้อง) </strong>: Well-chosen samples can provide reliable insights with reduced effort. ตัวอย่างที่เลือกมาได้ดีสามารถวิเคราะห์เชิงลึกได้ดี</li>
    <li><strong>Diversity (ความหลากหลาย) </strong>: Helps ensure representation of different groups within the population. เป็นตัวแทนของกลุ่มต่างๆ</li>
</ul>
""", unsafe_allow_html=True)


st.title('Types of Sampling Methods')
# Types of Sampling Methods

# 1. Probability Sampling
st.subheader('1. Probability Sampling')


st.write("Probability sampling methods ensure that every individual in the population has a known, non-zero chance of being selected. มั่นใจได้ว่าทุกคนมีโอกาศเท่ากันไม่ใช้ 0")
st.write("These methods are generally more statistically rigorous and reduce sampling bias. ใช้หลักการทางสถิติมากกว่าและลดความ Bias")

st.markdown("""
<h5>Simple Random Sampling</h5>
<ul>
    <li><strong> </strong> Each member of the population has an equal chance of being selected. ทุกคนมีโอกาสเท่ากัน</li>
    <li><strong> </strong> Randomly drawing names from a hat. การจับฉลาก</li>

</ul>
""", unsafe_allow_html=True)


st.markdown("""
<h5>Systematic Sampling</h5>
<ul>
    <li><strong> </strong> Selects every k-th individual from a list or sequence after randomly choosing a starting point. เลือกทุกๆ คนที่ k จากรายการหรือลำดับหลังจากเลือกจุดเริ่มต้นแบบสุ่ม</li>
    <li><strong> </strong> Example: Surveying every 10th customer entering a store. เลือกลูกค้าทุกคนที่ 10 ที่เข้าร้าน</li>

</ul>
""", unsafe_allow_html=True)

st.markdown("""
<h5>Stratified Sampling</h5>
<ul>
    <li><strong> </strong> Divides the population into subgroups (strata) based on shared characteristics, then randomly samples from each group. แบ่งประชากรออกเป็นกลุ่มย่อยแล้วสุ่มจากกลุ่มย่อย</li>
    <li><strong> </strong> Sampling men and women separately to ensure proportional representation. สุ่มตัวอย่างจากกลุ่มชายและกลุ่มหญิงให้เหมาะสมกับสัดส่วน</li>

</ul>
""", unsafe_allow_html=True)

st.markdown("""
<h5>Cluster Sampling</h5>
<ul>
    <li><strong> </strong> Divides the population into clusters (usually based on geography or natural groupings), then randomly selects entire clusters for the study. แบ่งประชากรออกเป็นแบบกลุ่ม และเลือกสุ่มกลุ่มและสำรวจข้อมูลจากกลุ่มนั้น</li>
    <li><strong> </strong> Choosing random schools and surveying all students in those schools. เลือกโรงเรียนแบบสุ่ม และสำรวจนักเรียนทุกคนในโรงเรียน </li>
</li>

</ul>
""", unsafe_allow_html=True)




st.subheader('2. Non-Probability Sampling')

st.write('Non-probability sampling does not provide every individual a known or equal chance of being selected.')
st.write('These methods are often easier and cheaper but may introduce bias. ง่ายและประหยัดกว่า แต่อาจจะมีความ Bias')

st.markdown("""
<h5>Convenience Sampling</h5>
<ul>
    <li><strong> </strong> Samples are chosen based on ease of access. ขึ้นอยู่กับความสะดวก </li>
    <li><strong> </strong> Surveying people at a nearby mall. สำรวจผู้คนในห้างสรรพสินค้าใกล้เคียง </li>
</li>

</ul>
""", unsafe_allow_html=True)

st.markdown("""
<h5>Judgmental (Purposive) Sampling</h5>
<ul>
    <li><strong> </strong> The researcher selects individuals based on their judgment about who will provide the best information. เลือกตามความคิดว่าใครให้ข้อมูลดีที่สุด </li>
    <li><strong> </strong>  Interviewing experts in a specific field. การสัมพาทผู้เชียวชาญเฉพาะ </li>
</li>

</ul>
""", unsafe_allow_html=True)

st.markdown("""
<h5>Quota Sampling</h5>
<ul>
    <li><strong> </strong> Ensures that specific subgroups are represented by setting quotas but does not randomly sample within those groups. รับประกันว่ากลุ่มย่อยเฉพาะจะได้รับการแสดงโดยการตั้งโควตา แต่จะไม่สุ่มสุ่มตัวอย่างภายในกลุ่มเหล่านั้น </li>
    <li><strong> </strong>  Ensuring 50% of respondents are female in a survey. </li>
</li>

</ul>
""", unsafe_allow_html=True)

st.markdown("""
<h5>Snowball Sampling</h5>
<ul>
    <li><strong> </strong> Used for hard-to-reach populations, where existing participants recruit others. ใช้สำหรับประชากรที่เข้าถึงได้ยาก โดยผู้เข้าร่วมที่มีอยู่จะคัดเลือกคนอื่นๆ เข้ามา </li>
    <li><strong> </strong>  Researching a niche group, like extreme sports enthusiasts. </li>
</li>

</ul>
""", unsafe_allow_html=True)


st.subheader('Key Factors to Consider in Sampling')
st.markdown("""
<h5> Key Factors to Consider in Sampling </h5>
<ul>
    <li><strong>Population Size </strong>The total number of individuals or items in the group you're studying. จำนวนรวมของบุคคลหรือรายการในกลุ่มที่คุณกำลังศึกษา </li>
    <li><strong>Sample Size </strong> The number of individuals included in the sample. จำนวนบุคคลที่รวมอยู่ในตัวอย่าง </li>
    <li><strong>Representation </strong> Ensuring the sample reflects the diversity of the population. Ensuring the sample reflects the diversity of the population. </li>
    <li><strong>Bias </strong> Minimizing systematic errors that can skew results. Minimizing systematic errors that can skew results. </li>
    
</ul>
""", unsafe_allow_html=True)


st.subheader('Applications of Sampling Methods')
st.markdown("""
<h5> Key Factors to Consider in Sampling </h5>
<ul>
    <li><strong>Market Research </strong>Understanding customer preferences.  </li>
    <li><strong>Epidemiology </strong> Studying disease spread in a population.  </li>
    <li><strong>Quality Control </strong> Inspecting products in manufacturing. </li>
    <li><strong>Social Science Research </strong> Understanding human behavior or opinions. </li>
    
</ul>
""", unsafe_allow_html=True)




data = pd.DataFrame({
    'Type': ['Probability Sampling', 'Non-Probability Sampling'],
    'Subtypes': ['Simple Random, Systematic, Stratified, Cluster','Convenience, Judgmental, Quota, Snowball'],
    'Pros': ['Statistically robust, representative','Quick, easy, cost-effective'],
    'Cons': ['Can be time-consuming or costly','May lead to bias, less generalizable']
})



st.title('Summary Table: Sampling Methods')
st.dataframe(data) 