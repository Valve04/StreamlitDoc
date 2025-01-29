import streamlit as st

st.title('Estimation')

st.subheader("Concept of Estimation")
st.write("The Central Limit Theorem (CLT) is one of the most fundamental concepts in statistics.")


st.subheader("1. Point Estimation")

st.markdown("""
<p> A point estimate is a single value that serves as an estimate of a population parameter. For example:  </p>
<ul>
    <li>The sample mean (X bar) can be used as a point estimate of the population mean μ  </li>
    <li>The sample proportion (p) can be used as a point estimate of the population proportion (P) </li>

</ul>
<p> Point estimates provide a specific value but do not indicate how close the estimate is to the true population parameter. </p>      

<p> Example </p>:
<ul>
    <li> If we want to estimate the average height of people in a city, we take a random sample of 100 people and find that the sample mean height is 5.8 feet. The point estimate of the population mean height is 5.8 feet. </li>
</ul>

""", unsafe_allow_html=True)


st.subheader("2. Interval Estimation (Confidence Intervals)")

st.markdown("""
<p class=""> An interval estimate provides a range of values within which the population parameter is likely to lie. This range is associated with a level of confidence, such as 95%, which quantifies the degree of certainty that the interval contains the true parameter.  </p>
<p><strong> Confidence Interval: </strong></p>   
<p> A confidence interval (CI) is an estimate of the population parameter expressed as an interval, with an associated probability (confidence level). For example: </p>
<ul>
    <li> A 95% confidence interval for a population mean μ might look like: [μ1 ,μ2 ] meaning that we are 95% confident that the true population mean lies between μ1 and μ2  </li>
</ul>     
<p>The wider the interval, the less precise the estimate; the narrower the interval, the more precise.</p>
<p><strong> Example: </strong></p>   
<p> Based on the sample of 100 people with a sample mean height of 5.8 feet and a standard deviation of 0.3 feet, we may calculate a 95% confidence interval for the population mean height as: </p>

""", unsafe_allow_html=True)

st.latex(r'''  \text{CI} = \bar{x} \pm Z_{\frac{\alpha}{2}} \times \frac{\sigma}{\sqrt{n}} ''')


st.latex(r'''
5.8 \pm 1.96 \times \frac{0.3}{\sqrt{100}} = [5.74, 5.86]
''')




st.markdown("""
<p> This means we are 95% confident that the true average height of the population lies between 5.74 feet and 5.86 feet. </p>
""", unsafe_allow_html=True)


st.subheader("3. Types of Estimators")

st.markdown("""
<p> There are different types of estimators used in statistical estimation, which can be characterized based on certain properties:  </p>

<ul>
    <li><strong> 1.Unbiased Estimators:  
        <ul>
            <li> An estimator is unbiased if its expected value equals the true parameter value. For example, the sample mean X_bar is an unbiased estimator of the population mean (μ) </li>
            <li> Property: The expected value of the estimator equals the true value of the parameter. </li>
        </ul>
    <strong></li>
</ul>
<ul>
    <li><strong> 2.Biased Estimators:  
        <ul>
            <li> An estimator is biased if its expected value does not equal the true parameter value. For example, if you use the sample variance formula without correcting for degrees of freedom, it can be biased. </li>
            <li> Property: The expected value of the estimator is not equal to the true parameter value. </li>
        </ul>
    <strong></li>
</ul>
<ul>       
    <li><strong> 3.Consistent Estimators:  
        <ul>
            <li> An estimator is consistent if, as the sample size increases, the estimator gets closer to the true population parameter. </li>
            <li> Property: The estimator becomes more accurate as the sample size grows. </li>
        </ul>
    <strong></li>
</ul>
<ul>                    
    <li><strong> 4.Efficient Estimators:  
        <ul>
            <li> An estimator is efficient if it has the smallest variance among all unbiased estimators. Efficiency is related to how much information the estimator captures from the data. </li>
            <li> Property: The estimator with the smallest variance is considered the most efficient. </li>
        </ul>
    <strong></li>
</ul> 
""", unsafe_allow_html=True)




st.subheader("4. Methods of Estimation")


st.markdown("""
<p> </p>

<ul>
    <li><strong> 1.Method of Moments:  
        <ul>
            <li> In this method, sample moments (such as sample mean, variance) are set equal to the corresponding population moments (mean, variance, etc.) to estimate the parameters of the population. </li>
            <li> For example, you can estimate the population mean by setting the sample mean equal to the population mean. </li>
        </ul>
    <strong></li>
</ul>

<ul>
    <li><strong> 2.Maximum Likelihood Estimation (MLE):  
        <ul>
            <li> In MLE, the parameters of the population are estimated by maximizing the likelihood function, which measures the likelihood of observing the given sample data under different parameter values. </li>
            <li> MLE is a widely used method in many fields because it provides estimators that are often unbiased, consistent, and efficient, especially for large sample sizes. </li>
        </ul>
    <strong></li>
</ul>
""", unsafe_allow_html=True)


st.subheader("5. Factors Affecting Estimation")

st.markdown("""
<p>Several factors influence the quality of an estimate: </p>

<ul>
    <li><strong> 1.Sample Size:  
        <ul>
            <li> A larger sample size generally leads to more accurate estimates, as it reduces the standard error of the estimate and narrows the confidence interval. </li>
        </ul>
    <strong></li>
</ul>

<ul>
    <li><strong> 2.Sample Variability:  
        <ul>
            <li> The more variability (or spread) there is in the sample data, the less precise the estimate is. Lower variability results in more reliable estimates. </li>
        </ul>
    <strong></li>
</ul>
            
<ul>
    <li><strong> 3.Confidence Level:  
        <ul>
            <li> A higher confidence level (e.g., 99% instead of 95%) results in a wider confidence interval, making the estimate less precise but more reliable. </li>
        </ul>
    <strong></li>
</ul>
""", unsafe_allow_html=True)



st.subheader("6. Hypothesis Testing and Estimation")

st.markdown("""
<p>Several factors influence the quality of an estimate: </p>

<ul>
    <li>Estimation and hypothesis testing are closely related. While estimation focuses on determining the value of a population parameter, 
            hypothesis testing involves evaluating whether a certain hypothesis about the parameter is true.</li>
</ul>

<ul>
    <li>For example, you might use an estimate of the population mean to test a hypothesis like </li>
    <li> H0 : μ = 50 (null hypothesis) </li>
    <li> H1 : μ <> 50 (alternative hypothesis) </li>
    <li> The estimation helps provide evidence for or against the hypothesis. </li>
</ul>  

""", unsafe_allow_html=True)




