import streamlit as st


st.title('Probability Basic')

st.subheader("Concept of Probability: The Basics")
st.write("Probability is a branch of mathematics that measures the likelihood of an event occurring. ความน่าจะเป็นเป็นสาขาหนึ่งของคณิตศาสตร์ที่ใช้วัดความน่าจะเป็นที่เหตุการณ์จะเกิดขึ้น")
st.write("It is foundational in statistics, data analysis, and everyday decision-making. เป็นพื้นฐานในสถิติ การวิเคราะห์และการตัดสินใจในชีวิตประจำวัน")
st.write("Probability is expressed as a value between 0 (impossible event) and 1 (certain event), or in percentages from 0% to 100%.")



st.subheader('Key Terms in Probability')

st.markdown("""
<h5></h5>
<ul>
    <li><strong> Experiment (การทำลอง)</strong>: 
    <ul>
            <li><strong> A process or activity that generates observable results. กระบวนการหรือกิจกรรมที่สร้างผลลัพธ์ที่สังเกตได้ </strong>  </li>
            <li><strong> Example: Rolling a dice, flipping a coin. การทอยลูกเต๋า และ การโยนเหรียญ </strong>  </li>
    </ul>
    </li>

</ul>
""", unsafe_allow_html=True)

st.markdown("""
<p></p>
<ul>
    <li><strong> Outcome (ผลลัพธ์)</strong>: 
    <ul>
            <li><strong> A single possible result of an experiment. ผลลัพธ์ที่เป็นไปได้ เพียงผลลัพธ์เดียว </strong>  </li>
            <li><strong> Example: Rolling a 4 on a six-sided die. ทอยลูกเต็า 6 ด้าน ได้เลข 4 </strong>  </li>
    </ul>
    </li>

</ul>
""", unsafe_allow_html=True)

st.markdown("""
<p></p>
<ul>
    <li><strong> Sample Space (S)</strong>: 
    <ul>
            <li><strong> The set of all possible outcomes of an experiment. ชุดผลลัพธ์ที่เป็นไปได้ทั้งหมด  </strong>  </li>
            <li><strong> Example: For a six-sided die. S = {1,2,3,4,5,6}  </strong>  </li>
    </ul>
    </li>

</ul>
""", unsafe_allow_html=True)

st.markdown("""
<p></p>
<ul>
    <li><strong> Event (E)</strong>: 
    <ul>
            <li><strong> A subset of the sample space that includes outcomes of interest. ซับเซ็ตที่สนใจ  </strong>  </li>
            <li><strong> Example: Rolling an even number. E = {2,4,6}  </strong>  </li>
    </ul>
    </li>

</ul>
""", unsafe_allow_html=True)

st.markdown("""
<h5>Calculating Probability</h5>
<ul>
    <li><strong> The probability of an event (E) occurring is defined as </strong>: </li>

</ul>
""", unsafe_allow_html=True)

st.latex(r'''
P(E) = \frac{\text{Number of favorable outcomes}}{\text{Total number of outcomes}}
''')

st.write(" ")
st.subheader('Types of Probability')


st.markdown("""
<h5>Theoretical Probability</h5>
<ul>
    <li>Based on mathematical reasoning without conducting experiments </li>
    <li>Example: The probability of flipping a coin and getting heads is P(Head) = 1/2 </li>

</ul>
""", unsafe_allow_html=True)

st.write("##### Experimental Probability")
st.write(" Determined by conducting an experiment and recording outcomes.")
st.latex(r'''
P(E) = \frac{\text{Number of favorable outcomes}}{\text{Total number of outcomes}}
''')
st.write("Example: Flipping a coin 100 times and observing 48 heads:")
st.latex(r'''
P(\text{Head}) = \frac{48}{100} = 0.48
''')


st.write("##### Subjective Probability")
st.write("  Based on personal judgment or experience rather than exact calculations.")
st.write("  Example: Estimating a 70% chance of rain based on how the sky looks.")


st.write("##### Probability Rules")
st.write("1.Rule of Complements:")
st.latex(r''' P(\text{NOT E}) = \ 1 - P(\text{E})  ''')
st.write("Example: If the probability of rain is 0.3, the probability of no rain is 1 - 0.3 = 0.7")
st.write("##### 2.Addition Rule (Union of Events)")
st.write("For two events A and B, the probability that either A or B occurs is")

st.latex(r''' P(\text{A U B}) = P(\text{A}) +  P(\text{B}) -  P(\text{A ∩ B})  ''')
st.write('If A and B are mutually exclusive')
st.latex(r''' P(\text{A U B}) = P(\text{A}) +  P(\text{B})  ''')

st.write("##### 3.Multiplication Rule (Intersection of Events)")
st.write('For two events A and B, the probability that both A and B occur is')

st.latex(r''' P(\text{A U B}) = P(\text{A}) *  P(\text{B|A})  ''')
st.write('if A and B independent A และ B เป็นอิสระกัน')
st.latex(r''' P(\text{A U B}) = P(\text{A}) *  P(\text{B})  ''')


st.write("##### Types of Events")
st.write("1.Independent Events:")

st.markdown("""
<ul>
    <li>The occurrence of one event does not affect the other. การเกิดขึ้นของเหตุการณ์หนึ่งไม่ส่งผลกระทบต่ออีกเหตุการณ์หนึ่ง</li>
    <li>Example: Rolling a dice and flipping a coin. การทอยลูกเต๋าและการโยนเหรียญ</li>

</ul>
""", unsafe_allow_html=True)

st.write("2.Dependent Events:")

st.markdown("""
<ul>
    <li>The occurrence of one event affects the probability of the other. การเกิดขึ้นของเหตุการณ์หนึ่งจะส่งผลต่อความน่าจะเป็นของเหตุการณ์อื่น</li>
    <li>Example: Drawing two cards from a deck without replacement. การจั่วไพ่ 2 ใบจากสำรับโดยไม่เปลี่ยนไพ่</li>

</ul>
""", unsafe_allow_html=True)

st.write("3.Mutually Exclusive Events:")

st.markdown("""
<ul>
    <li>Two events cannot occur simultaneously. เหตุการณ์สองเหตุการณ์ไม่สามารถเกิดขึ้นพร้อมๆ กันได้ </li>
    <li>Rolling a 4 and rolling a 5 on a single dice roll. ทอยลูกเต๋าได้เลข 4 และ 5 บนลูกเต๋าหนึ่งลูก</li>

</ul>
""", unsafe_allow_html=True)


# P(A∪B)=P(A)+P(B)−P(A∩B)

st.subheader('Applications of Probability')


st.markdown("""
<h5>Theoretical Probability</h5>
<ul>
    <li><strong>Risk Analysis</strong>: Estimating the likelihood of events like financial losses.  
            การประเมินความน่าจะเป็นของเหตุการณ์ต่างๆ เช่น การสูญเสียทางการเงิน </li>
    <li><strong>Games and Gambling</strong>: Calculating odds and expected returns. 
            การคำนวณอัตราต่อรองและผลตอบแทนที่คาดหวัง </li>
    <li><strong>Machine Learning</strong>: Building predictive models (e.g., Bayesian models). 
            การสร้างแบบจำลองการคาดการณ์ (เช่น โมเดลแบบเบย์) </li>
    <li><strong>Quality Control</strong>: Ensuring defect rates remain within acceptable limits. 
            การทำให้แน่ใจว่าอัตราข้อบกพร่องยังคงอยู่ในขอบเขตที่ยอมรับได้ </li>

</ul>
""", unsafe_allow_html=True)