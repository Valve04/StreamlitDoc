import streamlit as st


st.title('Discrete Distributions')

st.subheader("Concept of Discrete Distributions")
st.write("A discrete distribution describes the probability distribution of a discrete random variable—a variable that can take on only specific, countable values (e.g., integers).")
st.write("ตัวแปรที่สามารถรับค่าที่นับได้เฉพาะเจาะจงเท่านั้น จำนวนเต็ม")
st.write("These distributions are fundamental in probability and statistics, particularly for modeling scenarios where outcomes are finite or can be enumerated.")

st.subheader("Key Characteristics of Discrete Distributions")

st.write("1.Countable Outcomes (ผลลัพธ์ที่นับได้):")
st.markdown("""
<ul>
    <li>The variable takes on a finite or countably infinite set of values. ตัวแปรจะมีค่าจำนวนจำกัดหรือจำนวนอนันต์ที่นับได้ </li>
    <li>Example: Rolling a die (x ∈ {1,2,3,4,5,6})</li>

</ul>
""", unsafe_allow_html=True)

st.write("2.Probability Mass Function (PMF) (ฟังก์ชันมวลความน่าจะเป็น):")
st.markdown("""
<ul>
    <li>Defines the probability of each specific outcome. กำหนดความน่าจะเป็นของผลลัพธ์ที่เฉพาะเจาะจงแต่ละอย่าง</li>
    <li> P(X=x) = f(x) </li>
    <li> For a fair die P(X=x) = 1/6 for x = 1,2,3,4,5,6 </li>

</ul>
""", unsafe_allow_html=True)

st.write("3.Sum of Probabilities (ผลรวมของความน่าจะเป็น):")
st.markdown("""
<ul>
    <li>The total probability over all possible outcomes equals 1</li>
    <li> sum(P(X=x)) = 1 </li>

</ul>
""", unsafe_allow_html=True)

st.write("4.Cumulative Distribution Function (CDF) (ฟังก์ชันการแจกแจงสะสม):")
st.markdown("""
<ul>
    <li>Represents the cumulative probability up to a value x. </li>
    <li> F(x) = P(X <= x) </li>

</ul>
""", unsafe_allow_html=True)


st.subheader("Common Examples of Discrete Distributions")

st.write("1.Bernoulli Distribution:")
st.write("Describes a random variable with two possible outcomes: success (1) and failure (0).")

st.latex(r'''
P(X = x) = p^x (1 - p)^{1 - x}, \quad x \in \{0, 1\}
''')



st.write("Flipping a coin (p = 0.5)")
st.image("images/BernoulliDistribution.png", caption="This is an example image.")


st.write("2.Binomial Distribution:")
st.write("Models the number of successes in n independent trials of a Bernoulli process.")
st.write("สร้างแบบจำลองจำนวนความสำเร็จในการทดลองแบบอิสระ n ครั้งของกระบวนการเบอร์นูลลี")

st.latex(r'''
P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}, \quad k = 0, 1, 2, \dots, n

''')

# st.write("""
# - \(p\): Probability of success (\(0 \leq p \leq 1\))
# - \(x\): Random variable with values \(0\) or \(1\)
# """)


st.write("Tossing a coin 10 times and counting heads. โยนเหรียญ 10 ครั้งและนับหัว")
st.image("images/Binomial Distribution.png", caption="This is an example image.")






st.write("3.Geometric Distribution:")
st.write("Models the number of successes in n independent trials of a Bernoulli process.")
st.write("สร้างแบบจำลองจำนวนความสำเร็จในการทดลองแบบอิสระ n ครั้งของกระบวนการเบอร์นูลลี")

st.latex(r'''
P(X = k) = (1 - p)^{k-1} p, \quad k = 1, 2, 3, \dots
''')

# st.write("""
# - \(p\): Probability of success (\(0 \leq p \leq 1\))
# - \(x\): Random variable with values \(0\) or \(1\)
# """)


st.write("Rolling a die until you get a 6. ทอยลูกเต๋าจนกระทั่งได้เลข 6")
st.image("images/Geometric Distribution.png", caption="This is an example image.")


st.write("Poisson Distribution:")
st.write("Describes the number of events occurring in a fixed interval of time or space, assuming events occur independently at a constant rate.")
st.write("อธิบายจำนวนเหตุการณ์ที่เกิดขึ้นในช่วงเวลาหรือพื้นที่คงที่ โดยถือว่าเหตุการณ์ต่างๆ เกิดขึ้นอย่างอิสระและอัตราคงที่")

st.latex(r'''
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \dots
''')

# st.write("""
# - \(p\): Probability of success (\(0 \leq p \leq 1\))
# - \(x\): Random variable with values \(0\) or \(1\)
# """)


st.write("Number of emails received in an hour. จำนวนอีเมล์ที่ได้รับในหนึ่งชั่วโมง ")
st.image("images/Poisson Distribution.png", caption="This is an example image.")





st.subheader("Applications of Discrete Distributions")


st.write("""
- \(Quality Control\) : Modeling the number of defective items in a batch (binomial). การสร้างแบบจำลองจำนวนสินค้าที่ชำรุดในหนึ่งชุด (ทวินาม)
- \(Customer Behavior\) : Number of purchases made by a customer in a month (Poisson). จำนวนการซื้อที่ลูกค้าทำในหนึ่งเดือน (ปัวซอง)
- \(Risk Analysis\) : Number of claims filed in an insurance portfolio (Poisson). จำนวนการเรียกร้องในพอร์ตโฟลิโอประกันภัย (ปัวซอง)
- \(Game Theory\) : Outcomes of dice rolls or coin flips in games (binomial, geometric). ผลลัพธ์ของการทอยลูกเต๋าหรือโยนเหรียญในเกม (ทวินาม, เรขาคณิต)
""")