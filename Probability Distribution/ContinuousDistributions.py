import streamlit as st
import pandas as pd

st.title('Continuous Distributions')

st.subheader("Concept of Continuous Distributions")
st.write("A continuous distribution describes the probability distribution of a continuous random variable—a variable that can take any value within a given range.")
st.write("การแจกแจงต่อเนื่องอธิบายถึงการแจกแจงความน่าจะเป็นของตัวแปรสุ่มต่อเนื่อง ซึ่งเป็นตัวแปรที่สามารถรับค่าใดก็ได้ภายในช่วงที่กำหนด")
st.write(" Unlike discrete distributions, where outcomes are countable, continuous distributions deal with uncountable, infinitely many possible values (e.g., real numbers).")
st.write("ไม่เหมือนกับการแจกแจงแบบแยกส่วน ซึ่งผลลัพธ์สามารถนับได้ การแจกแจงแบบต่อเนื่องจะเกี่ยวข้องกับค่าที่เป็นไปได้ที่นับไม่ได้และมีจำนวนไม่สิ้นสุด (เช่น จำนวนจริง)")


st.subheader("Key Characteristics of Continuous Distributions")

st.write("##### 1.Uncountable Outcomes:")
st.write("A continuous random variable can take any value in an interval or range, such as [a,b] or (-∞,∞)")
st.write("Example: Heights of people, time taken to complete a task, temperature ความสูงของผู้คน เวลาที่ใช้ในการทำงานหนึ่งๆ และอุณหภูมิ")

st.write("##### 2.Probability Density Function (PDF):")
st.write("The PDF , f(x) , describes the relative likelihood of a random variable taking a specific value.")
st.write("The probability of the random variable falling within an interval is the area under the curve of the PDF over that interval:")

st.latex(r'''
P(\text{a<= X <=b}) = \int_{-\infty}^{\infty} f(x) \,
''')

st.write("The value of f(x) is not a probability but a density.")

st.write("##### 3.Total Probability:")
st.write("The value of f(x) is not a probability but a density.")
st.latex(r'''
\int_{-\infty}^\infty f(x) \, dx = 1
''')

st.write("##### 4.Cumulative Distribution Function (CDF):")
st.write(" Represents the probability that the random variable is less than or equal to a specific value x:")
st.latex(r'''
F(x) = P(X \leq x) = \int_{-\infty}^x f(t) \, dt
''')

st.write("##### 5.Probability at a Specific Value:")
st.write(" The probability of a continuous random variable taking an exact value is zero:")
st.latex(r'''
P(X=x) = 0
''')
st.write("Probabilities are always calculated over intervals.")

st.subheader("Common Examples of Continuous Distributions")

st.write("##### 1.Uniform Distribution:")
st.write(" All values within a range are equally likely.")
st.latex(r'''
f(x) = \frac{1}{b - a}, \quad a \leq x \leq b
''')
st.write("""
Where:
- \(f(x)\): Probability Density Function (PDF) of the uniform distribution
- \(a\): Lower bound of the interval
- \(b\): Upper bound of the interval
- The probability is uniformly distributed between \(a\) and \(b\)
""")
st.image("images/Uniform Distribution.jpg", caption="This is an example image.")



st.write("##### 2.Normal Distribution (Gaussian):")
st.write(" The 'bell curve' shape; values near the mean are more likely.")

st.latex(r'''
f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
''')
st.write("""
Where:
- \(f(x)\): Probability Density Function (PDF) of the normal distribution
- \(\mu\): Mean (average) of the distribution
- \(\sigma\): Standard deviation (measure of spread)
- \(x\): The random variable
- \(\exp\): Exponential function
""")

st.write("Example: Heights, test scores.")
st.image("images/Normal Distribution (Gaussian).png", caption="This is an example image.")



st.write("##### 3.Exponential Distribution:")
st.write("Models the time between events in a Poisson process.")
st.latex(r'''
f(x) = \lambda e^{-\lambda x}, \quad x \geq 0
''')

st.write("""
Where:
- \(f(x)\): Probability Density Function (PDF) of the exponential distribution
- \(\lambda\): Rate parameter (the rate at which events occur)
- \(x\): The random variable (the time between events or number of occurrences)
- The function describes the time between events in a Poisson process.
""")
st.write("Example: Time between arrivals at a bus stop.")
st.image("images/Exponential Distribution.png", caption="This is an example image.")





st.write("##### 4.Beta Distribution:")
st.write("Defined on [0,1] and commonly used in Bayesian statistics.")
st.latex(r'''
f(x; \alpha, \beta) = \frac{x^{\alpha-1} (1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad 0 \leq x \leq 1
''')

st.write("""
Where:
- \(f(x; \ alpha, \ beta)\): Probability Density Function (PDF) of the Beta distribution
- \(\ alpha\) and \(\ beta\): Shape parameters of the Beta distribution
- \(B(\ alpha, \ beta)\): Beta function, which normalizes the PDF
- \(x\): The random variable, with values between 0 and 1
""")
st.write("Example: Probabilities of success rates.")
st.image("images/Beta Distribution.png", caption="This is an example image.")





st.write("##### 5.Chi-Squared Distribution:")
st.write("Used in hypothesis testing and confidence intervals.")
st.write("Variance analysis.")
st.image("images/Chi-Squared Distribution.png", caption="This is an example image.")


st.subheader("Applications of Continuous Distributions")

st.write("##### 1.Natural Phenomena:")
st.write("Modeling real-world measurements like rainfall, temperatures, and weights (normal distribution).")
st.write("##### 2.Finance:")
st.write("Stock price movements (log-normal or normal distribution).")

st.write("##### 3.Engineering:")
st.write("Time-to-failure models (exponential distribution).")

st.write("##### 4.Statistics:")
st.write("Confidence intervals and hypothesis testing (chi-squared, t-distribution).")

st.subheader("Visualizing Continuous Distributions")
st.write("##### 1.Probability Density Function (PDF):")
st.write("Smooth curve showing the density of probabilities.")
st.write("X-axis: Range of values; Y-axis: Probability density.")
st.write("##### 2.Cumulative Distribution Function (CDF):")
st.write("Increasing curve showing cumulative probabilities.")

# st.title("omparison with Discrete Distributions")

data = pd.DataFrame({
    'Aspect': ['Nature of Outcomes', 'Examples','Probability Function','Probability at a Specific Point','Total Probability','Probability Calculation','Cumulative Distribution Function (CDF)','Visualization','Examples of Distributions','Applicability','Calculation of Probability'],
    'Discrete Distribution': ['Countable outcomes, finite or countably infinite.','Rolling a die, number of defective items, coin tosses.','Probability Mass Function (PMF): Specifies probability for each value.','Probability of a specific value can be non-zero.','Sum of probabilities of all outcomes equals 1.','P(X=x) for specific values.','The cumulative probability is computed by summing PMFs.','Bar chart (discrete values on x-axis and probability on y-axis).','Binomial, Poisson, Geometric.','Used for countable events, or finite sample spaces.','P(X=x) gives direct probability for each outcome.'],
    'Continuous Distribution': ['Uncountable outcomes, usually any value within an interval or range.','Height of a person, temperature, time taken for an event.','Probability Density Function (PDF): Specifies probability density over a range.','Probability of a specific value is always zero.','Area under the PDF curve equals 1 (integral over all outcomes).','P(a<= X <= b) for intervals, calculated using the area under the curve.','The cumulative probability is computed by integrating the PDF.','Smooth curve (continuous values on x-axis and density on y-axis).','Normal, Exponential, Uniform, Beta.','Used for modeling quantities that vary smoothly over a range.','P(X=x) is zero, but probabilities are computed over intervals.']
})



st.title('Comparison with Discrete Distributions')
st.dataframe(data) 