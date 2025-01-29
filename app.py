# import streamlit as st


# st.set_page_config(page_title="My Custom App", page_icon=":guardsman:", layout="wide", initial_sidebar_state="expanded")

# # Rename the sidebar menu
# st.sidebar.title("Custom Sidebar Menu")

# # Main content
# st.title('Welcome to My Streamlit App')

# # def Hello():
# #     name = st.text_input("What's your name?")
# #     if st.button('Greet'):
# #         st.write(f"Hello, {name}! ")

# # def test():
# #     st.title("test")

# # st.line_chart({"data":[1,2,3,4,5]})


# if __name__ == '__main__':
#     None

#     # Hello()
#     # test()


import streamlit as st

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    if st.button("Log in"):
        st.session_state.logged_in = True
        st.rerun()

def logout():
    if st.button("Log out"):
        st.session_state.logged_in = False
        st.rerun()

login_page = st.Page(login, title="Log in", icon=":material/login:")
logout_page = st.Page(logout, title="Log out", icon=":material/logout:")

DescriptiveStatistics = st.Page("Foundational Concepts/DesStatistics.py", title="Descriptive Statistics", icon=":material/dashboard:", default=True)
ProbabilityBasic = st.Page("Foundational Concepts/Probability.py", title="Probability Basic", icon=":material/dashboard:", default=False)
SamplingMethods = st.Page("Foundational Concepts/SamplingMet.py", title="Sampling Methods", icon=":material/dashboard:", default=False)

CentralLimitTheorem = st.Page("Probability Distribution/CentralLimitTheorem.py", title="CentralLimit Theorem", icon=":material/dashboard:", default=False)
ContinuousDistributions = st.Page("Probability Distribution/ContinuousDistributions.py", title="Continuous Distributions", icon=":material/dashboard:", default=False)
DiscreteDistributions = st.Page("Probability Distribution/DiscreteDistributions.py", title="Discrete Distributions", icon=":material/dashboard:", default=False)

Estimation = st.Page("Statistical Inference/Estimation.py", title="Estimation", icon=":material/dashboard:", default=False)
Hypothesis = st.Page("Statistical Inference/Hypothesis Testing.py", title="Hypothesis Testing", icon=":material/dashboard:", default=False)
T_and_Z_Test = st.Page("Statistical Inference/T-Test and Z-Test.py", title="T-Test and Z-Test", icon=":material/dashboard:", default=False)
ANOVA = st.Page("Statistical Inference/ANOVA.py", title="ANOVA", icon=":material/dashboard:", default=False)

SimpleLinearRegression = st.Page("Regression Analysis/SimpleLinearRegression.py", title="SimpleLinear Regression", icon=":material/dashboard:", default=False)
MultipleLinearRegression = st.Page("Regression Analysis/MultipleLinearRegression.py", title="MultipleLinear Regression", icon=":material/dashboard:", default=False)
LogisticRegression = st.Page("Regression Analysis/LogisticRegression.py", title="Logistic Regression", icon=":material/dashboard:", default=False)


ModelDiagnostics = st.Page("Advanced Topic in Regression/ModelDiagnostics.py", title="Model Diagnostics", icon=":material/dashboard:", default=False)
Regularization = st.Page("Advanced Topic in Regression/Regularization.py", title="Regularization", icon=":material/dashboard:", default=False)
GLM = st.Page("Advanced Topic in Regression/GLM.py", title="Generalized Linear Models", icon=":material/dashboard:", default=False)

PCA = st.Page("Multivariate Statistics/PCA.py", title="PCA", icon=":material/dashboard:", default=False)
Clustering = st.Page("Multivariate Statistics/Clustering.py", title="Clustering", icon=":material/dashboard:", default=False)
FactorAnalysis = st.Page("Multivariate Statistics/FactorAnalysis.py", title="FactorAnalysis", icon=":material/dashboard:", default=False)

TrendAnalysisandSeasonality = st.Page("TimeSeries Analysis/TrendAnalysisandSeasonality.py", title="TrendAnalysis and Seasonalit", icon=":material/dashboard:", default=False)
AutoregressiveModels = st.Page("TimeSeries Analysis/AutoregressiveModels.py", title="Autoregressive Models", icon=":material/dashboard:", default=False)
ExponentialSmoothing = st.Page("TimeSeries Analysis/ExponentialSmoothing.py", title="Exponential Smoothing", icon=":material/dashboard:", default=False)

Chi_Squaretest = st.Page("Non-Parametric Methods/Chi_Squaretest.py", title="Chi Square test", icon=":material/dashboard:", default=False)
Mann_WhitneyUandWilcoxin = st.Page("Non-Parametric Methods/Mann_WhitneyUandWilcoxin.py", title="Mann-Whitney U and Wilcoxon Test", icon=":material/dashboard:", default=False)

SupervisedLearning = st.Page("Machine Learning Techniques/SupervisedLearning.py", title="Supervised Learning", icon=":material/dashboard:", default=False)
UnsupervisedLearning = st.Page("Machine Learning Techniques/UnsupervisedLearning.py", title="Unsupervised Learning", icon=":material/dashboard:", default=False)
ModelEvaluations = st.Page("Machine Learning Techniques/ModelEvaluations.py", title="Model Evaluations", icon=":material/dashboard:", default=False)

BayesianStatistics = st.Page("Special Topics/BayesianStatistics.py", title="Bayesian Statistics", icon=":material/dashboard:", default=False)
ExperimentalDesign = st.Page("Special Topics/ExperimentalDesign.py", title="Experimental Design", icon=":material/dashboard:", default=False)
BootstrappingandResampling = st.Page("Special Topics/BootstrappingandResampling.py", title="Bootstrapping and Resampling", icon=":material/dashboard:", default=False)



search = st.Page("tools/search.py", title="Search", icon=":material/search:")
history = st.Page("tools/history.py", title="History", icon=":material/history:")

if st.session_state.logged_in:
    pg = st.navigation(
        {
            "Account": [logout_page],
            "FoundationalConcepts": [DescriptiveStatistics, ProbabilityBasic, SamplingMethods],
            "ProbabilityDistribution": [DiscreteDistributions,ContinuousDistributions,CentralLimitTheorem],
            "StatisticalInference":[Estimation,Hypothesis,T_and_Z_Test,ANOVA],
            "RegressionAnalysis":[SimpleLinearRegression,MultipleLinearRegression,LogisticRegression],
            "Advanced Topic in Regression":[ModelDiagnostics,Regularization,GLM],
            "Multivariate Statistics":[PCA,Clustering,FactorAnalysis],
            "TimeSeries Analysis":[TrendAnalysisandSeasonality,AutoregressiveModels,ExponentialSmoothing],
            "Non-Parametric Methods":[Chi_Squaretest,Mann_WhitneyUandWilcoxin],
            "Machine Learning Techniques":[SupervisedLearning,UnsupervisedLearning,ModelEvaluations],
            "Special Topics":[BayesianStatistics,ExperimentalDesign,BootstrappingandResampling],
            "Tools": [search, history],
            
        }
    )
else:
    pg = st.navigation([login_page])

# st.title('Topic Statistics')


pg.run()





