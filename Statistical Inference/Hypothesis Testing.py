import streamlit as st


st.title('Hypothesis Testing')

st.subheader("1. Key Terms in Hypothesis Testing")

st.markdown("""
<p> 1.Key Terms in Hypothesis Testing  </p>

<ul>
    <li><strong> 1.Null Hypothesis (H0):  
        <ul>
            <li> The default assumption or claim we aim to test. </li>
            <li> It usually represents no effect, no difference, or the status quo. </li>
            <li> Example: H0:μ=50 (the population mean is 50) </li>
        </ul>
    <strong></li>
</ul>
<ul>
    <li><strong> 2.Alternative Hypothesis (HA) :  
        <ul>
            <li> Represents what we want to investigate or the opposing claim to the null hypothesis. </li>
            <li> Example: H0:μ<>50 (the population mean is not 50) </li>
        </ul>
    <strong></li>
</ul>
<ul>       
    <li><strong> 3.Significance Level (α):  
        <ul>
            <li> The probability of rejecting the null hypothesis when it is true (Type I error). </li>
            <li> Common values are 0.05 (5%) or 0.01 (1%). </li>
        </ul>
    <strong></li>
</ul>
<ul>                    
    <li><strong> 4.Test Statistic:  
        <ul>
            <li> A value calculated from the sample data to evaluate the null hypothesis. Examples include 
                <ul>
                    <li><strong>z-score (for large samples or known variance).</li>
                    <li><strong>t-score (for small samples or unknown variance).</li>
                    <li><strong>χ2 -statistic (for categorical data).</li>
                 </ul>
            </li>
        </ul>
    <strong></li>
</ul> 
            
<ul>                    
    <li><strong> 5.P-value:  
        <ul>
            <li> 
                The probability of obtaining the observed sample results (or more extreme results) if the null hypothesis is true. 
            </li>
            <li> 
                A small p-value (less than α) suggests evidence against H0.
            </li>
        </ul>
    <strong></li>
</ul> 
            
<ul>                    
    <li><strong> 6.Critical Region :  
        <ul>
            <li> 
                The range of values for the test statistic that leads to rejecting the null hypothesis.
            </li>
        </ul>
    <strong></li>
</ul> 
""", unsafe_allow_html=True)


st.subheader("2. Steps in Hypothesis Testing")

st.markdown("""
<p> 2.Steps in Hypothesis Testing  </p>
       
<ul>                    
    <li><strong> 1.State the Hypotheses:  
        <ul>
            <li> 
                Define the null hypothesis (H0) and the alternative hypothesis (HA)
            </li>
        </ul>
    <strong></li>
</ul> 
            
<ul>                    
    <li><strong> 2.Set the Significance Level (α) :  
        <ul>
            <li> 
                Decide on the threshold for rejecting H0 (e.g.,α = 0.05 )
            </li>
        </ul>
    <strong></li>
</ul> 
            

<ul>                    
    <li><strong> 3.Choose the Appropriate Test :  
        <ul>
            <li> 
                Based on the type of data and sample size, select a statistical test (e.g., Z-test , t-test ,χ2 test )
            </li>
        </ul>
    <strong></li>
</ul> 
            
<ul>                    
    <li><strong> 4.Calculate the Test Statistic :  
        <ul>
            <li> 
                Use the sample data to compute the value of the test statistic.
            </li>
        </ul>
    <strong></li>
</ul> 
            
<ul>                    
    <li><strong> 5.Determine the P-value or Critical Value :  
        <ul>
            <li> 
                Compare the test statistic to the critical value(s) or calculate the p-value.
            </li>
        </ul>
    <strong></li>
</ul> 
            
<ul>                    
    <li><strong> 6.Make a Decision :  
        <ul>
            <li> 
                Reject H0 if the test statistic falls in the critical region or if the p-value is less than α.
            </li>
             <li> 
                Fail to reject H0 if otherwise.
            </li>
        </ul>
    <strong></li>
</ul> 
            
<ul>                    
    <li><strong> 7.Interpret the Results :  
        <ul>
            <li> 
                State whether there is enough evidence to support HA and its implications in the context of the problem.
            </li>
        </ul>
    <strong></li>
</ul> 
""", unsafe_allow_html=True)



st.subheader("3. Types of Hypothesis Tests")

st.markdown("""
<p> 2.Steps in Hypothesis Testing  </p>
       
<ul>                    
    <li><strong> 1.One-Tailed Test:  
        <ul>
            <li> 
                Tests for an effect in one direction (e.g. Ha: μ > 50 or Ha: μ < 50)
            </li>
            <li> 
                Used when the research question is directional.
            </li>
        </ul>
    <strong></li>
</ul> 
            
<ul>                    
    <li><strong> 1.Two-Tailed Test :  
        <ul>
            <li> 
                Tests for an effect in both directions (e.g. Ha: μ <> 50 )
            </li>
            <li> 
                Used when the research question is non-directional.
            </li>
        </ul>
    <strong></li>
</ul> 
""", unsafe_allow_html=True)



st.subheader("4. Errors in Hypothesis Testing")

st.markdown("""
<p> In hypothesis testing, two types of errors can occur when making decisions about the null hypothesis H0:
    Type I Error and Type II Error. 
    These errors arise because of the inherent uncertainty in drawing conclusions from sample data rather than the entire population. 
</p>
        
<p><strong>1.Type I Error (False Positive) (α) <strong></p>
<ul>                    
    <li>
        Rejecting H0 when it is actually true (false positive).
    </li>
    <li>
         Controlled by the significance level.
    </li>       
            

</ul> 
<p><strong>2.Type II Error (False Negative) (β)<strong></p>          
<ul>                    
    <li>
        Failing to reject H0 when HA is ture (false negative).
    </li>  
    <li>
        Inversely related to the power of the test (1-β).
    </li>    
</ul>      
""", unsafe_allow_html=True)
st.subheader("5. Example of Hypothesis Testing")

st.markdown("""
<p><strong> Problem <strong>
            A company claims the average weight of its product is 50 kg. 
            A sample of 30 products has a mean weight of 48 kg and a standard deviation of 3 kg. 
            Test this claim at α = 0.05.
</p>
       
<p><strong> Steps <strong></p>
<ul>                    
    <li>
        1. State the Hypotheses:
            <ul>
                <li>
                    H0 : μ = 50 (the population mean weight is 50 kg). 
                </li>
                <li>
                    H1 : μ <> 50 (the population mean weight is not 50 kg).
                </li>
            </ul>
    </li>     
    <li>
        2. Set Significance Level:
            <ul>
                <li>
                    α = 0.05 
                </li>
            </ul>
    </li>  
    <li>
        3. Choose the Test:
            <ul>
                <li>
                    Use a t-test (small sample size, unknown population variance).
                </li>
            </ul>
    </li>   
    <li>
        4. Calculate the Test Statistic:
    </li>  
</ul> 
""", unsafe_allow_html=True)

st.latex(r'''
t = \frac{\frac{s}{n}}{\overline{x} - \mu} = \frac{\frac{3}{30}}{48 - 50} = -3.65
''')

st.markdown("""
    
<ul>                    
    <li>
        5. Determine the Critical Value: For a two-tailed test with
    </li>     
</ul> 
    
""", unsafe_allow_html=True)
st.latex(r'''
\text{} \alpha = 0.05 \text{ and } df = 29, t_{\text{critical}} = \pm 2.045
''')

st.markdown("""
    
<ul>                    
    <li>
        6. Decision
    </li>     
</ul> 
""", unsafe_allow_html=True)
st.latex(r'''
\text{Since } |t| = 3.65 > 2.045, \text{ reject } H_0.
''')
st.markdown("""
    
<ul>                    
    <li>
        7. Interpretation
            <ul>
                <li>
                    There is strong evidence that the mean weight is different from 50 kg.
                </li>
            </ul>
    </li>     
</ul> 
""", unsafe_allow_html=True)