import streamlit as st


st.title('CentralLimit Theorem')
st.subheader("Central Limit Theorem (CLT)")
st.write("The Central Limit Theorem (CLT) is one of the most fundamental concepts in statistics.")

st.write("It describes how the distribution of sample means (or sums) becomes approximately normal, regardless of the shape of the population distribution, as the sample size increases.")

st.write()
st.write("Statement of the Central Limit Theorem:")
st.write("หากคุณสุ่มตัวอย่างขนาดใหญ่เพียงพอจากประชากรที่มีรูปแบบการแจกแจงใดๆ ก็ตาม (ไม่ว่าจะปกติ เบ้ หรืออื่นๆ) การแจกแจงการสุ่มของค่าเฉลี่ยของกลุ่มตัวอย่างจะเข้าใกล้การแจกแจงแบบปกติเมื่อขนาดของกลุ่มตัวอย่าง  n เพิ่มขึ้น โดยที่ประชากรต้องมีค่าเฉลี่ยและความแปรปรวนจำกัด")

st.subheader("  Key Components of the CLT")
st.write(" ##### 1.Population Distribution")
st.write("การแจกแจงของข้อมูลแต่ละจุด (ประชากร) อาจมีรูปร่างใดก็ได้—ปกติ (normal), เบี่ยงเบน (skewed), สองโหมด (bimodal) หรือรูปร่างอื่นๆ")


st.write(" ##### 2.Sampling Distribution of the Mean")
st.write("หากคุณทำการสุ่มตัวอย่างซ้ำๆ ขนาด n จากประชากรและคำนวณค่าเฉลี่ยของแต่ละตัวอย่าง การแจกแจงของค่าเฉลี่ยตัวอย่างเหล่านั้นเรียกว่า การแจกแจงการสุ่มตัวอย่างของค่าเฉลี่ย (sampling distribution of the mean)")


st.write(" ##### 3.As Sample Size Increases")
st.write("- เมื่อขนาดตัวอย่าง n เพิ่มขึ้น การแจกแจงการสุ่มตัวอย่างของค่าเฉลี่ยตัวอย่างจะเข้าใกล้การแจกแจงแบบปกติ (normal distribution)")
st.write("- สิ่งนี้เป็นจริงแม้ว่าการกระจายตัวของประชากรดั้งเดิมจะไม่ปกติก็ตาม")
st.write("- ยิ่งขนาดตัวอย่างมีขนาดใหญ่เท่าใด การกระจายตัวอย่างของค่าเฉลี่ยก็จะใกล้เคียงกับการแจกแจงแบบปกติมากขึ้นเท่านั้น")


st.write(" ##### 4.Mean and Standard Error")
st.write("- ค่าเฉลี่ยของการกระจายการสุ่มตัวอย่างจะเท่ากับค่าเฉลี่ยของประชากร นั่นคือ μ X_bar = μ")
st.write("- ค่าเบี่ยงเบนมาตรฐานของการแจกแจงการสุ่มตัวอย่าง ซึ่งเรียกอีกอย่างว่าข้อผิดพลาดมาตรฐาน คือ ค่าเบี่ยงเบนมาตรฐานของประชากร  σ หารด้วยรากที่สองของขนาดตัวอย่าง n:")
st.latex(r'''
SE = \frac{\sigma}{\sqrt{n}}
''')
st.write("- เมื่อ n เพิ่มขึ้น ข้อผิดพลาดมาตรฐานจะลดลง ทำให้การกระจายการสุ่มมีความแคบลงตามค่าเฉลี่ยของประชากร")



st.subheader("  Formula for Sampling Distribution of the Mean")

st.write("If X1 , X2,...Xn are random variables from a population with mean μ and standard deviation σ, the sampling distribution of the sample mean X bar for sample size n will have")
st.write("μ X_bar = μ")
st.latex(r'''
\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}
''')


st.subheader( "Why is the Central Limit Theorem Important?")
st.write("##### 1.Normal Approximation")
st.write("- CLT ทำให้เราสามารถอนุมานเกี่ยวกับค่าเฉลี่ยของประชากรได้โดยใช้เทคนิคการแจกแจงแบบปกติ แม้ว่าประชากรนั้นจะไม่มีการแจกแจงแบบปกติก็ตาม")
st.write("- สิ่งนี้มีประโยชน์อย่างยิ่ง เนื่องจากวิธีการทางสถิติหลายวิธี (เช่น ช่วงความเชื่อมั่นและการทดสอบสมมติฐาน) ถือว่าข้อมูลมีการแจกแจงแบบปกติ")

st.write("##### 2.Statistical Inference")

# หัวข้อหลัก

st.markdown("""
<p> The CLT enables the use of the normal distribution for estimation and testing even when the original data distribution is unknown. It is the foundation for methods like </p>
<ul>
    <li>Confidence intervals for the population mean. </li>
    <li>Hypothesis tests involving the population mean. </li>

</ul>
""", unsafe_allow_html=True)

st.write("##### 3.Practical Application:")
st.write("ในสถานการณ์จริง มักไม่สามารถทำได้จริงในการทราบการกระจายตัวที่แน่นอนของประชากร อย่างไรก็ตาม ตราบใดที่ขนาดของกลุ่มตัวอย่างมีขนาดใหญ่เพียงพอ เราสามารถถือได้ว่าค่าเฉลี่ยของกลุ่มตัวอย่างจะกระจายตัวตามปกติ โดยอาศัย CLT วิธีนี้ช่วยให้สามารถอนุมานทางสถิติได้อย่างน่าเชื่อถือ แม้ในสถานการณ์ที่ไม่เหมาะสม")

st.subheader("When Does the CLT Apply?")
st.write("##### Large Sample Sizes:")
st.markdown("""
<p>  </p>
<ul>
    <li>ยิ่งขนาดตัวอย่าง 𝑛 n มีขนาดใหญ่ขึ้น การประมาณค่าแบบปกติก็จะดีขึ้น โดยทั่วไป ขนาดตัวอย่าง 𝑛 ≥ 30 n≥30 ถือว่ามีขนาดใหญ่เพียงพอสำหรับการใช้ CLT แม้ว่าเกณฑ์ที่แน่นอนอาจขึ้นอยู่กับการกระจายตัวของประชากรก็ตาม </li>
    <li>หากประชากรมีการเบ้มาก อาจต้องใช้กลุ่มตัวอย่างที่ใหญ่กว่า (เช่น 𝑛 ≥ 50 n≥50) เพื่อให้ CLT ยังคงอยู่ </li>
</ul>
""", unsafe_allow_html=True)



st.write("##### Finite Mean and Variance:")
st.markdown("""
<p> </p>
<ul>
    <li>ประชากรต้องมีค่าเฉลี่ยและความแปรปรวนที่จำกัด (finite mean and finite variance) หากประชากรมีความแปรปรวนไม่จำกัด (เช่น การแจกแจงแบบ Cauchy) หลักทฤษฎี CLT (Central Limit Theorem) จะไม่สามารถใช้ได้ </li>
</ul>
""", unsafe_allow_html=True)




