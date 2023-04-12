import streamlit as st
import reliability 
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reliability.Fitters import Fit_Beta_2P, Fit_Everything, Fit_Exponential_1P, Fit_Exponential_2P, Fit_Gamma_2P, Fit_Gumbel_2P, Fit_Loglogistic_2P, Fit_Loglogistic_3P, Fit_Lognormal_2P, Fit_Lognormal_3P, Fit_Normal_2P, Fit_Weibull_2P, Fit_Weibull_3P, Fit_Gamma_3P
from reliability.Distributions import Weibull_Distribution, Lognormal_Distribution, Exponential_Distribution, Normal_Distribution, Gamma_Distribution, Beta_Distribution, Loglogistic_Distribution, Gumbel_Distribution
from reliability.Probability_plotting import plot_points, plotting_positions
from reliability.ALT_fitters import Fit_Weibull_Exponential, Fit_Weibull_Eyring, Fit_Weibull_Power, \
Fit_Weibull_Dual_Exponential, Fit_Weibull_Power_Exponential, Fit_Weibull_Dual_Power, Fit_Lognormal_Exponential, Fit_Lognormal_Eyring, \
Fit_Lognormal_Power, Fit_Lognormal_Dual_Exponential, Fit_Lognormal_Power_Exponential, Fit_Lognormal_Dual_Power, Fit_Normal_Exponential, \
Fit_Normal_Eyring, Fit_Normal_Power, Fit_Normal_Dual_Exponential, Fit_Normal_Dual_Power, Fit_Exponential_Exponential, \
Fit_Exponential_Eyring, Fit_Exponential_Power, Fit_Exponential_Dual_Exponential, Fit_Exponential_Power_Exponential, Fit_Exponential_Dual_Power
import pickle

#st.set_page_config(page_title="Accelerated Life Testing",page_icon="📈",layout="wide", initial_sidebar_state="expanded")

def show():
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    #ReportStatus {visibility: hidden;}

    </style>

    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    #href_homepage = f'<a href="https://reliability.ceerma.com/" style="text-decoration: none; color :black;" > <button kind="primary" class="css-qbe2hs edgvbvh1">Go to Homepage</button></a>'
    #st.markdown(href_homepage, unsafe_allow_html=True)

    updatemenus_log = [
        dict(
            buttons=list([
                dict(
                    args=[{'xaxis.type': '-', 'yaxis.type': '-'}],
                    label='Linear',
                    method='relayout'

                ),
                dict(
                    args=[{'xaxis.type': 'log', 'yaxis.type': 'log'}],
                    label='Log-xy',
                    method='relayout'

                ),
            ]),
            direction="right",
            type="buttons",
            pad={"r": 10, "t": 10},
            x=0.0,
            xanchor="left",
            y=1.115,
            yanchor="top"                       

        )
    ]

    def st_tonumlist(txt_input):
        txt_input = txt_input.rsplit(sep=",")
        num_list =[]
        for i in txt_input:
            try:
                num_list.append(float(i))
            except:
                pass
        return num_list

    single_stress_ALT_models_list = [
        "Weibull_Exponential",
        "Weibull_Eyring",
        "Weibull_Power",
        "Lognormal_Exponential",
        "Lognormal_Eyring",
        "Lognormal_Power",
        "Normal_Exponential",
        "Normal_Eyring",
        "Normal_Power",
        "Exponential_Exponential",
        "Exponential_Eyring",
        "Exponential_Power",
    ]


    dual_stress_ALT_models_list = [
        "Weibull_Dual_Exponential",
        "Weibull_Power_Exponential",
        "Weibull_Dual_Power",
        "Lognormal_Dual_Exponential",
        "Lognormal_Power_Exponential",
        "Lognormal_Dual_Power",
        "Normal_Dual_Exponential",
        "Normal_Power_Exponential",
        "Normal_Dual_Power",
        "Exponential_Dual_Exponential",
        "Exponential_Power_Exponential",
        "Exponential_Dual_Power",
    ]



    st.title("Accelerated Life Testing")
    st.write("In this module, you can provide your Accelerated Life Testing (ALT) data (complete or incomplete) and fit the most common probability distributions in reliability ")

    with st.expander(label='Help'):
        st.write('When using this module, please take into consideration the following points:')
        st.write('- There is no need to sort the data in any particular order as this is all done automatically;')
        st.write('- For each stress level, number of failure points must be at least three, for single stress, or four, for dual stress.')
        st.write(" ")
        st.write("""Accelerated Life Testing is implemented on Exponential, Weibull, 
        Normal and Lognormal distributions. One of the parameters of the distribution will
        be changed into a Life Model function L(S), which models the behaviour of the 
        stress changes. The parameter changed is as follows:
        """)
        st.latex(r'\text{Exponential } \lambda = \frac{1}{L(S)}')
        st.latex(r'\text{Weibull } \alpha = L(S)')
        st.latex(r'\text{Normal } \mu = L(S)')
        st.latex(r'\text{Lognormal } \mu = \ln\left(L(S)\right)')

        st.write("L(S) function options are shown in Equation Information tab.")
        
    equation = st.expander("Equation Information")
    equation.write("Single-stress models available:")
    equation.latex(r'''\text{Exponential (or Arrhenius): } L(S)=b*\exp\left(\frac{a}{S}\right)''')
    equation.latex(r"""\text{Eyring: } L(S)=\frac{1}{S}*\exp\left(-\left(c-\frac{a}{S}\right)\right)""")
    equation.latex(r"""\text{Power (or Inverse Power Law): } L(S)=a*S^n""")
    equation.write("Dual-stress models available:")
    equation.latex(r"""\text{Dual-Exponential (or Temperature-Humidity): } L(S_1,S_2)=c*\exp\left(\frac{a}{S_1}+\frac{b}{S_2}\right)""")
    equation.latex(r"""\text{Dual-Power: } L(S_1,S_2)=c*S_1^m*S_2^n""")
    equation.latex(r"""\text{Power-Exponential (or Thermal-Nonthermal): } L(S_1,S_2)=c*\exp\left(\frac{a}{S_1}\right)*S_2^n""")
    equation.info("""Although named Power-Exponential, this Life-Stress Model actually 
    applies Exponential to stress 1 and Power to stress 2. You can simply shift the order 
    of your input columns to switch which stress is affected by each function.""")

    expander = st.expander("Data format")
    expander.info('Upload an excel file thar contains the following columns: failure or right-censored time ("Time"), \
        the time type, if failure or right censored ("Type"), and the stress level (only "Stress1" or also "Stress2" for dual stress models).')
    df_show = {'Time': [620,632,685,822,380,416,460,596,216,146,332,400],
               'Type': ['F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F', 'F'],
               'Stress1': [348,348,348,348,348,348,348,348,378,378,378,378],
               'Stress2': [3,3,3,3,5,5,5,5,3,3,3,3],}
    df_show = pd.DataFrame.from_dict(df_show)
    expander.write(df_show, width=50)
    expander.info('The use level stress parameter is optional. If single stress model, enter only one value. For example:')
    expander.write('323')
    expander.info('If dual stress model, enter two values separated by ",". For example: ')
    expander.write('323, 2')

    col2_1, col2_2 = st.columns(2)
    uploaded_file = col2_1.file_uploader("Upload a XLSX file", \
        type="xlsx", accept_multiple_files=False)

    dual = False

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        # sdf = df.style.format('{:10n}'.format)
        col2_2.dataframe(df)
        fdata = df[df['Type'] == 'F']
        ftime = np.array(fdata.iloc[:,0])
        fstress_1 = np.array(fdata.iloc[:,2])
        cdata = df[df['Type'] == 'C']
        ctime = np.array(cdata.iloc[:,0])
        cstress_1 = np.array(cdata.iloc[:,2])
        use_level = st.text_input("Use level stress (optional)")
        use_level = st_tonumlist(use_level)
        if len(df.columns) == 4:
            fstress_2 = np.array(fdata.iloc[:,3])
            cstress_2 = np.array(cdata.iloc[:,3])
            dual = True
            if use_level:
                if len(use_level) != 2:
                    st.error('Enter two use level stresses')
        elif len(df.columns) == 3:
            if use_level:
                if len(use_level) != 1:
                    st.error('Enter one use level stress')
                else:
                    use_level = use_level[0]        
        #     include = st.multiselect('Choose which distribution(s) you want to fit to your data', dual_stress_ALT_models_list)
        # else:
        #     include = st.multiselect('Choose which distribution(s) you want to fit to your data', single_stress_ALT_models_list)

    if dual == False:
        include = st.multiselect('Choose which distribution(s) you want to fit to your data', single_stress_ALT_models_list)
    else:
        include = st.multiselect('Choose which distribution(s) you want to fit to your data', dual_stress_ALT_models_list)

    method = st.radio('Choose the optimizer', ('TNC', 'L-BFGS-B'))
    st.info('The optimizers are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails, the initial guess (using least squares) will be returned.')
    metric = st.radio('Choose a goodness of fit criteria', ('BIC', 'AICc', 'Log-likelihood'))

    IC = 0.9
    print_results = False
    show_probability_plot = True
    show_life_stress_plot = True

    st.write(" ")

    if st.button("Fit ALT model"):
        
        if use_level == 0:
            use_level = None

        # plt.savefig('test.png')

        # st.write('## Results of all fitted ALT models')
        # st.write(results.results)

        # st.write('## Results of the best fitted ALT model')
        # distribution_name = results.best_model_name

        best_BIC, best_AICc, best_loglik = np.inf, np.inf, np.inf
        best_model = None
        best_model_name = None
        results = []

        if include:
            if dual == False:
                # create empty dataframe to append results
                results = pd.DataFrame(
                    columns=[
                        "ALT_model",
                        "a",
                        "b",
                        "c",
                        "n",
                        "beta",
                        "sigma",
                        "Log-LH",
                        "AICc",
                        "BIC",
                    ]
                )
                if 'Weibull_Exponential' in include:
                    res = Fit_Weibull_Exponential(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Weibull_Exponential",
                            "a": f'{res.a:0.4f}',
                            "b": f'{res.b:0.4f}',
                            "c": "",
                            "n": "",
                            "beta": f'{res.beta:0.4f}',
                            "sigma": "",
                            "Log-LH": f'{res.loglik:0.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Weibull_Exponential'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Weibull_Exponential'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Weibull_Exponential'


                if 'Weibull_Eyring' in include:
                    res = Fit_Weibull_Eyring(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Weibull_Eyring",
                            "a": f'{res.a:0.4f}',
                            "b": "",
                            "c": f'{res.c:0.4f}',
                            "n": "",
                            "beta": f'{res.beta:0.4f}',
                            "sigma": "",
                            "Log-LH": f'{res.loglik:0.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Weibull_Eyring'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Weibull_Eyring'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Weibull_Eyring'


                if 'Weibull_Power' in include:
                    res = Fit_Weibull_Power(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Weibull_Power",
                            "a": f'{res.a:0.4f}',
                            "b": "",
                            "c": "",
                            "n": f'{res.n:0.4f}',
                            "beta": f'{res.beta:0.4f}',
                            "sigma": "",
                            "Log-LH": f'{res.loglik:0.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Weibull_Power'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Weibull_Power'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglikAICc
                        best_model = res
                        best_model_name = 'Weibull_Power'

                if 'Lognormal_Exponential' in include:
                    res = Fit_Lognormal_Exponential(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Lognormal_Exponential",
                            "a": f'{res.a:0.4f}',
                            "b": f'{res.b:0.4f}',
                            "c": "",
                            "n": "",
                            "beta": "",
                            "sigma": f'{res.sigma:0.4f}',
                            "Log-LH": f'{res.loglik:0.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Lognormal_Exponential'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Lognormal_Exponential'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Lognormal_Exponential'

                if 'Lognormal_Eyring' in include:
                    res = Fit_Lognormal_Eyring(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Lognormal_Eyring",
                            "a": f'{res.a:0.4f}',
                            "b": "",
                            "c": f'{res.c:0.4f}',
                            "n": "",
                            "beta": "",
                            "sigma": f'{res.sigma:0.4f}',
                            "Log-LH": f'{res.loglik:0.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Lognormal_Eyring'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Lognormal_Eyring'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Lognormal_Eyring'

                if 'Lognormal_Power' in include:
                    res = Fit_Lognormal_Power(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Lognormal_Power",
                            "a": f'{res.a:0.4f}',
                            "b": "",
                            "c": "",
                            "n": f'{res.n:0.4f}',
                            "beta": "",
                            "sigma": f'{res.sigma:0.4f}',
                            "Log-LH": f'{res.loglik:0.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Lognormal_Power'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Lognormal_Power'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Lognormal_Power'

                if 'Normal_Exponential' in include:
                    res = Fit_Normal_Exponential(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Normal_Exponential",
                            "a": f'{res.a:0.4f}',
                            "b": f'{res.b:0.4f}',
                            "c": "",
                            "n": "",
                            "beta": "",
                            "sigma": f'{res.sigma:0.4f}',
                            "Log-LH": f'{res.loglik:0.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Normal_Exponential'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Normal_Exponential'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Normal_Exponential'

                if 'Normal_Eyring' in include:
                    res = Fit_Normal_Eyring(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Normal_Eyring",
                            "a": f'{res.a:0.4f}',
                            "b": "",
                            "c": f'{res.c:0.4f}',
                            "n": "",
                            "beta": "",
                            "sigma": f'{res.sigma:0.4f}',
                            "Log-LH": f'{res.loglik:0.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Normal_Eyring'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Normal_Eyring'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Normal_Eyring'

                if 'Normal_Power' in include:
                    res = Fit_Normal_Power(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Normal_Power",
                            "a": f'{res.a:0.4f}',
                            "b": "",
                            "c": "",
                            "n": f'{res.n:0.4f}',
                            "beta": "",
                            "sigma": f'{res.sigma:0.4f}',
                            "Log-LH": f'{res.loglik:0.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Normal_Power'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Normal_Power'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Normal_Power'

                if 'Exponential_Exponential' in include:
                    res = Fit_Exponential_Exponential(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Exponential_Exponential",
                            "a": f'{res.a:0.4f}',
                            "b": f'{res.b:0.4f}',
                            "c": "",
                            "n": "",
                            "beta": "",
                            "sigma": "",
                            "Log-LH": f'{res.loglik:0.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Exponential_Exponential'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Exponential_Exponential'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Exponential_Exponential'

                if 'Exponential_Eyring' in include:
                    res = Fit_Exponential_Eyring(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Exponential_Eyring",
                            "a": f'{res.a:0.4f}',
                            "b": "",
                            "c": f'{res.c:0.4f}',
                            "n": "",
                            "beta": "",
                            "sigma": "",
                            "Log-LH": f'{res.loglik:0.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Exponential_Eyring'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Exponential_Eyring'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Exponential_Eyring'

                if 'Exponential_Power' in include:
                    res = Fit_Exponential_Power(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Exponential_Power",
                            "a": f'{res.a:0.4f}',
                            "b": "",
                            "c": "",
                            "n": f'{res.n:0.4f}',
                            "beta": "",
                            "sigma": "",
                            "Log-LH": f'{res.loglik:0.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Exponential_Power'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Exponential_Power'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Exponential_Power'

                # # Recreate plt figure in plotly
                # lines = best_model.probability_plot

                # fig = go.Figure()
                # x_min,x_max = lines.get_xlim()
                # y_min,y_max = lines.get_ylim()

                # for line in lines.get_lines():

                #     if line.get_linestyle() == '--':
                #         dash = 'dash'
                #     else:
                #         dash = None

                #     if line.get_label() in list(set(map(str, fstress_1))):
                #         label = line.get_label()
                #     else:
                #         label = label

                #     fig.add_trace(go.Scatter(x=line.get_xdata(), y=line.get_ydata(), name = label,  line=dict(color = line.get_color(), dash=dash), visible = True))

                # fig.update_xaxes(range=[x_min, x_max])
                # fig.update_yaxes(range=[y_min, y_max])
                # fig.update_layout(width = 600, height = 600, title = 'Probability plot', yaxis=dict(tickformat='.2e'), xaxis=dict(tickformat='.2e'), updatemenus=updatemenus_log, title_text='Probability plot') #- {} - a = {}, b = {}, beta = {}'.format('Weibull Exponential', results.a, results.b, results.beta))
                # st.plotly_chart(fig)

            else:
                # create empty dataframe to append results
                results = pd.DataFrame(
                    columns=[
                        "ALT_model",
                        "a",
                        "b",
                        "c",
                        "m",
                        "n",
                        "beta",
                        "sigma",
                        "Log-LH",
                        "AICc",
                        "BIC",
                    ]
                )

                if 'Weibull_Dual_Exponential' in include:
                    res = Fit_Weibull_Dual_Exponential(failures=ftime, failure_stress_1=fstress_1, failure_stress_2=fstress_2, \
                    right_censored=ctime, right_censored_stress_1=cstress_1, right_censored_stress_2=cstress_2, use_level_stress=use_level, \
                    print_results=print_results, show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, \
                    CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Weibull_Dual_Exponential",
                            "a": f'{res.a:0.4f}',
                            "b": f'{res.b:0.4f}',
                            "c": f'{res.c:0.4f}',
                            "m": "",
                            "n": "",
                            "beta": f'{res.beta:0.4f}',
                            "sigma": "",
                            "Log-LH": f'{res.loglik:0.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Weibull_Dual_Exponential'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Weibull_Dual_Exponential'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Weibull_Dual_Exponential'

                if 'Weibull_Power_Exponential' in include:
                    res = Fit_Weibull_Power_Exponential(failures=ftime, failure_stress_1=fstress_1, failure_stress_2=fstress_2, \
                    right_censored=ctime, right_censored_stress_1=cstress_1, right_censored_stress_2=cstress_2, use_level_stress=use_level, \
                    print_results=print_results, show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, \
                    CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Weibull_Power_Exponential",
                            "a": f'{res.a:0.4f}',
                            "b": "",
                            "c": f'{res.c:0.4f}',
                            "m": "",
                            "n": f'{res.n:0.4f}',
                            "beta": f'{res.beta:0.4f}',
                            "sigma": "",
                            "Log-LH": f'{res.loglik:0.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Weibull_Power_Exponential'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Weibull_Power_Exponential'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Weibull_Power_Exponential'

                if 'Weibull_Dual_Power' in include:
                    res = Fit_Weibull_Dual_Power(failures=ftime, failure_stress_1=fstress_1, failure_stress_2=fstress_2, \
                    right_censored=ctime, right_censored_stress_1=cstress_1, right_censored_stress_2=cstress_2, use_level_stress=use_level, \
                    print_results=print_results, show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, \
                    CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Weibull_Dual_Power",
                            "a": "",
                            "b": "",
                            "c": f'{res.c:0.4f}',
                            "m": res.m,
                            "n": f'{res.n:0.4f}',
                            "beta": f'{res.beta:0.4f}',
                            "sigma": "",
                            "Log-LH": f'{res.loglik:0.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Weibull_Dual_Power'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Weibull_Dual_Power'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Weibull_Dual_Power'

                if 'Lognormal_Dual_Exponential' in include:
                    res = Fit_Lognormal_Dual_Exponential(failures=ftime, failure_stress_1=fstress_1, failure_stress_2=fstress_2, \
                    right_censored=ctime, right_censored_stress_1=cstress_1, right_censored_stress_2=cstress_2, use_level_stress=use_level, \
                    print_results=print_results, show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, \
                    CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Lognormal_Dual_Exponential",
                            "a": f'{res.a:0.4f}',
                            "b": f'{res.b:0.4f}',
                            "c": f'{res.c:0.4f}',
                            "m": "",
                            "n": "",
                            "beta": "",
                            "sigma": f'{res.sigma:0.4f}',
                            "Log-LH": f'{res.loglik:0.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Lognormal_Dual_Exponential'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Lognormal_Dual_Exponential'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Lognormal_Dual_Exponential'
                        
                if 'Lognormal_Power_Exponential' in include:
                    res = Fit_Lognormal_Power_Exponential(failures=ftime, failure_stress_1=fstress_1, failure_stress_2=fstress_2, \
                    right_censored=ctime, right_censored_stress_1=cstress_1, right_censored_stress_2=cstress_2, use_level_stress=use_level, \
                    print_results=print_results, show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, \
                    CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Lognormal_Power_Exponential",
                            "a": f'{res.a:0.4f}',
                            "b": "",
                            "c": f'{res.c:0.4f}',
                            "m": "",
                            "n": f'{res.n:0.4f}',
                            "beta": "",
                            "sigma": f'{res.sigma:0.4f}',
                            "Log-LH": f'{res.loglik:0.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Lognormal_Power_Exponential'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Lognormal_Power_Exponential'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Lognormal_Power_Exponential'

                if 'Lognormal_Dual_Power' in include:
                    res = Fit_Lognormal_Dual_Power(failures=ftime, failure_stress_1=fstress_1, failure_stress_2=fstress_2, \
                    right_censored=ctime, right_censored_stress_1=cstress_1, right_censored_stress_2=cstress_2, use_level_stress=use_level, \
                    print_results=print_results, show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, \
                    CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Lognormal_Dual_Power",
                            "a": "",
                            "b": "",
                            "c": f'{res.c:0.4f}',
                            "m": res.m,
                            "n": f'{res.n:0.4f}',
                            "beta": "",
                            "sigma": f'{res.sigma:0.4f}',
                            "Log-LH": f'{res.loglik:0.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Lognormal_Dual_Power'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Lognormal_Dual_Power'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Lognormal_Dual_Power'

                if 'Normal_Dual_Exponential' in include:
                    res = Fit_Normal_Dual_Exponential(failures=ftime, failure_stress_1=fstress_1, failure_stress_2=fstress_2, \
                    right_censored=ctime, right_censored_stress_1=cstress_1, right_censored_stress_2=cstress_2, use_level_stress=use_level, \
                    print_results=print_results, show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, \
                    CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Normal_Dual_Exponential",
                            "a": f'{res.a:0.4f}',
                            "b": f'{res.b:0.4f}',
                            "c": f'{res.c:0.4f}',
                            "m": "",
                            "n": "",
                            "beta": "",
                            "sigma": f'{res.sigma:0.4f}',
                            "Log-LH": f'{res.loglik:0.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Normal_Dual_Exponential'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Normal_Dual_Exponential'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Normal_Dual_Exponential'
                        
                if 'Normal_Dual_Power' in include:
                    res = Fit_Normal_Dual_Power(failures=ftime, failure_stress_1=fstress_1, failure_stress_2=fstress_2, \
                    right_censored=ctime, right_censored_stress_1=cstress_1, right_censored_stress_2=cstress_2, use_level_stress=use_level, \
                    print_results=print_results, show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, \
                    CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Normal_Dual_Power",
                            "a": "",
                            "b": "",
                            "c": f'{res.c:0.4f}',
                            "m": res.m,
                            "n": f'{res.n:0.4f}',
                            "beta": "",
                            "sigma": f'{res.sigma:0.4f}',
                            "Log-LH": f'{res.loglik:0.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Normal_Dual_Power'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Normal_Dual_Power'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Normal_Dual_Power'

                if 'Exponential_Dual_Exponential' in include:
                    res = Fit_Exponential_Dual_Exponential(failures=ftime, failure_stress_1=fstress_1, failure_stress_2=fstress_2, \
                    right_censored=ctime, right_censored_stress_1=cstress_1, right_censored_stress_2=cstress_2, use_level_stress=use_level, \
                    print_results=print_results, show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, \
                    CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Exponential_Dual_Exponential",
                            "a": f'{res.a:0.4f}',
                            "b": f'{res.b:0.4f}',
                            "c": f'{res.c:0.4f}',
                            "m": "",
                            "n": "",
                            "beta": "",
                            "sigma": "",
                            "Log-LH": f'{res.loglik:0.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Exponential_Dual_Exponential'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Exponential_Dual_Exponential'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Exponential_Dual_Exponential'

                if 'Exponential_Power_Exponential' in include:
                    res = Fit_Exponential_Power_Exponential(failures=ftime, failure_stress_1=fstress_1, failure_stress_2=fstress_2, \
                    right_censored=ctime, right_censored_stress_1=cstress_1, right_censored_stress_2=cstress_2, use_level_stress=use_level, \
                    print_results=print_results, show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, \
                    CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Exponential_Power_Exponential",
                            "a": f'{res.a:0.4f}',
                            "b": "",
                            "c": f'{res.c:0.4f}',
                            "m": "",
                            "n": f'{res.n:0.4f}',
                            "beta": "",
                            "sigma": "",
                            "Log-LH": f'{res.loglik:0.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Exponential_Power_Exponential'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Exponential_Power_Exponential'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Exponential_Power_Exponential'

                if 'Exponential_Dual_Power' in include:
                    res = Fit_Exponential_Dual_Power(failures=ftime, failure_stress_1=fstress_1, failure_stress_2=fstress_2, \
                    right_censored=ctime, right_censored_stress_1=cstress_1, right_censored_stress_2=cstress_2, use_level_stress=use_level, \
                    print_results=print_results, show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, \
                    CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Exponential_Dual_Power",
                            "a": "",
                            "b": "",
                            "c": f'{res.c:0.4f}',
                            "m": res.m,
                            "n": f'{res.n:0.4f}',
                            "beta": "",
                            "sigma": "",
                            "Log-LH": f'{res.loglik:0.4f}',
                            "AICc": f'{res.AICc:.4f}',
                            "BIC": f'{res.BIC:.4f}',
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Exponential_Dual_Power'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Exponential_Dual_Power'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Exponential_Dual_Power'

            st.write('## Results of all fitted ALT models')
            results = pd.DataFrame.from_dict(results)
            st.write(results)

            st.write('## Results of the best fitted ALT model')
            st.write(best_model_name)
            st.write(best_model.results)
            st.write(f'The mean life of best model is {best_model.mean_life:.4f}\n')

            probability_plot = best_model.probability_plot.figure
            life_stress_plot = best_model.life_stress_plot.figure
            col1, col2 = st.columns(2)
            col1.pyplot(probability_plot)
            col2.pyplot(life_stress_plot)

        else:
            st.error('Please, choose at least one model to fit.')
            st.stop()