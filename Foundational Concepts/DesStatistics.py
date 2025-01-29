import streamlit as st


st.title('Descriptive Statistics')

st.write('Descriptive Statistics is a branch of statistics that focuses on summarizing and describing the main features of a dataset. Instead of drawing conclusions beyond the data at hand (which is the role of inferential statistics), descriptive statistics helps in understanding the basic characteristics of the data.')

# st.latex(r'''
#     \mu = \frac{1}{n} \sum_{i=1}^{n} x_i
         
# ''')

st.write('\n\n' )
st.markdown("# Key Elements of Descriptive Statistics:")
# st.markdown("""
# These measures indicate where the center of the data lies.

# Common examples include:

# - **Mean (Average)**: Sum of all values divided by the number of values.
# - **Median**: The middle value when the data is ordered.
# - **Mode**: The most frequently occurring value(s).
# """)


st.write('\n\n' )
st.markdown("## 1.Measures of Central Tendency:")
st.write('\n\n' )
st.markdown("""
<p>These measures indicate where the center of the data lies.</p>
<p>Common examples include:</p>
<ul>
    <li><strong>Mean (Average)</strong>: Sum of all values divided by the number of values.</li>
    <li><strong>Median</strong>: The middle value when the data is ordered.</li>
    <li><strong>Mode</strong>: The most frequently occurring value(s).</li>
</ul>
""", unsafe_allow_html=True)

st.latex(r'''\mu = \frac{1}{n} \sum_{i=1}^{n} x_i''')
st.latex(r'''
\text{Median} = 
\begin{cases} 
x_{\left(\frac{n+1}{2}\right)} & \text{if } n \text{ is odd} \\
\frac{x_{\left(\frac{n}{2}\right)} + x_{\left(\frac{n}{2}+1\right)}}{2} & \text{if } n \text{ is even}
\end{cases}
''')
st.latex( r'''\text{Mode} = \arg\max_{x_i \in X} \, \text{Frequency}(x_i)''')


st.markdown("## 2.Measures of Dispersion (Spread):")
st.write('\n\n' )
st.markdown("""
<p>These measures describe the variability or spread in the dataset.</p>
<p>Common examples include:</p>
<ul>
    <li><strong>Range:  (Average)</strong>:The difference between the maximum and minimum values.</li>
    <li><strong>Variance</strong>: The average of the squared differences from the mean.</li>
    <li><strong>Standard Deviation:</strong> The square root of the variance, indicating how much the data deviates from the mean.</li>
    <li><strong>Interquartile Range (IQR):</strong> The difference between the 75th percentile (Q3) and the 25th percentile (Q1).</li>
</ul>
""", unsafe_allow_html=True)
st.write('\n\n' )

# LaTeX formula for Range
st.latex( r'\text{Range} = \max(x) - \min(x)')
st.latex( r'\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2')
st.latex( r's^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2')
st.latex( r'\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}')
st.latex( r's = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2}')
st.latex( r'\text{IQR} = Q3 - Q1')



st.write('\n\n' )

st.markdown("## 3.Shape of the Data Distribution:")
st.write('\n\n' )
st.markdown("""
<p>Descriptive statistics also helps describe the shape of the data's distribution.</p>
<p>Key aspects include:</p>
<ul>
    <li><strong>Skewness:</strong> Measures asymmetry of the data distribution.
        <ul>
            <li><strong>Positive skew:</strong> Longer tail on the right.</li>
            <li><strong>Negative skew:</strong> Longer tail on the left.</li>
        </ul>
    </li>
    <li><strong>Kurtosis:</strong>: Measures the "tailedness" of the distribution..
        <ul>
            <li><strong>High kurtosis:</strong> Heavy tails.</li>
            <li><strong>Low kurtosis:</strong> Light tails.</li>
        </ul>
    </li>
</ul>
""", unsafe_allow_html=True)
st.write('\n\n' )

st.latex( r'''
\text{Skewness} = \frac{n}{(n-1)(n-2)} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{s} \right)^3
''')
st.latex( r'''
\text{Kurtosis} = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{s} \right)^4 
- \frac{3(n-1)^2}{(n-2)(n-3)}
''')




st.markdown("## 4.Data Visualization:")
st.write('\n\n' )
st.markdown("""
<p>Visual representations complement numerical summaries, making patterns and trends easier to identify. Examples include:</p>
<ul>
    <li><strong>Histograms:  (Average)</strong>Show the frequency of data within specified intervals.</li>
    <li><strong>Box Plots:</strong> Highlight the central tendency and dispersion along with outliers.</li>
    <li><strong>Bar Charts:</strong> Display categorical data.</li>
    <li><strong>Scatter Plots:</strong> Show relationships between two variables.</li>
</ul>
""", unsafe_allow_html=True)


st.write('\n\n' )
st.markdown("## 5.Summary Statistics:")
st.write('\n\n' )
st.markdown("""
<p>Often, datasets are summarized using a combination of the above metrics:</p>
<ul>
    <li><strong></strong>Count, mean, median, standard deviation, minimum, maximum, and quartiles.</li>
 
</ul>
""", unsafe_allow_html=True)

st.write('\n\n' )
st.image("images/A.jpg", caption="This is an example image.")