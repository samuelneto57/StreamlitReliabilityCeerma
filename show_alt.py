import streamlit as st
import numpy as np
import pandas as pd

import functions
import distributions


def show():
    st.write("""
    In this module, you can provide your Accelerated Life Testing (ALT)
    data (complete or incomplete) and fit the most common probability
    distributions in reliability.
    """)

    with st.expander('Short Guide'):
        st.write('When using this module, please take into consideration \
                 the following points:')
        st.info("""
        - There is no need to sort the data in any particular order as this
        is all done automatically;
        - For single-stress, number of failure points must be at least three
        for each stress level;
        - For dual-stress, number of failure points must be at least four for
        each combination of stress levels.
        """)
        st.write("""
        Accelerated Life Testing is implemented on Exponential, Weibull,
        Normal and Lognormal distributions. One of the parameters of the
        distribution will be changed into a Life Model function L(S), which
        models the behaviour of the stress changes. The parameter changed
        is as follows:
        """)
        for item in distributions.alt_substitution_equations:
            st.latex(item)

        st.info("- L(S) function options are shown in Equation Information tab.")

    with st.expander("Equation Information"):
        st.write("Life Model functions available:")
        st.write("- Single-stress:")
        for item in distributions.alt_single_equations:
            st.latex(item)
        st.write("- Dual-stress:")
        for item in distributions.alt_dual_equations:
            st.latex(item)
        st.info("""Although named Power-Exponential, this Life-Stress Model
        actually applies Exponential to stress 1 and Power to stress 2. You
        can simply shift the order of your input columns to switch which
        stress is affected by each function.""")


    with st.expander("Data format"):
        st.info("""
        Upload an excel file that contains the following columns:
        * Time - failure or right-censored time;
        * Type - 'F' for failure time, 'C' for right-censored time;
        * Stress1 - stress level associated with this failure;
        * Stress2 (optional) - second stress level, for dual stress models.
        """)
        df_show = {
            'Time': [620,632,685,822,380,416,460,596,216,146,332,400],
            'Type': ['F','F','F','F','F','F','F','F','F','F','F','F'],
            'Stress1':[348,348,348,348,348,348,348,348,378,378,378,378],
            'Stress2': [3,3,3,3,5,5,5,5,3,3,3,3],
        }
        df_show = pd.DataFrame.from_dict(df_show)
        st.write(df_show)
        st.info('The use level stress parameter is optional. \
                If single stress model, enter only one value. For example:')
        st.write('323')
        st.info('If dual stress model, enter two values \
                separated by ",". For example:')
        st.write('323, 2')

    header = st.checkbox("Does your data contain header?", value=True)
    head = 0 if header else None

    col2_1, col2_2 = st.columns(2)
    uploaded_file = col2_1.file_uploader("Upload a XLSX file",
                                         type="xlsx",
                                         accept_multiple_files=False,
                                         label_visibility="collapsed")

    dual = False
    if uploaded_file:
        df = pd.read_excel(uploaded_file, header=head)
        n_cols = len(df.columns)
        if n_cols <= 2 or n_cols >= 5:
            st.error('Please enter data according to the \
                      "Data format" example!')
        else:
            col2_2.dataframe(df)
            df['Type'] = df['Type'].str.upper()
            fdata = df[df['Type'] == 'F']
            cdata = df[df['Type'] == 'C']
            ftime = np.array(fdata.iloc[:,0])
            ctime = np.array(cdata.iloc[:,0])
            fstress_1 = np.array(fdata.iloc[:,2])
            cstress_1 = np.array(cdata.iloc[:,2])
            if n_cols > 3:
                dual = True
                fstress_2 = np.array(fdata.iloc[:,3])
                cstress_2 = np.array(cdata.iloc[:,3])
            use_level = st.text_input("Use level stress (optional)")
            if use_level:
                use_level = use_level.strip().split(sep=',')
                use_level = [float(x or 0) for x in use_level]
                if dual and len(use_level) != 2:
                    st.error('Please enter two use level stresses!')
                elif not dual and len(use_level) != 1:
                    st.error('Please enter one use level stress only!')
            else:
                use_level = None

    if dual:
        distr = distributions.alt_dual_distributions
    else:
        distr = distributions.alt_single_distributions

    include = st.multiselect('Choose which distribution(s) you want to \
                              fit your data to:', list(distr))

    metric, method, ic = functions.fit_options(alt=True, include_CI=True)

    if st.button("Fit ALT model"):

        if not uploaded_file:
            st.error('Please upload a file first!')
            st.stop()

        print_results = False
        probability_plot = True
        life_stress_plot = True

        if dual:
            function_parameters = [
                ftime, fstress_1, fstress_2, ctime, cstress_1, cstress_2,
                use_level, ic, method,
                probability_plot, life_stress_plot, print_results
            ]
        else:
            function_parameters = [
                ftime, fstress_1, ctime, cstress_1,
                use_level[0], ic, method,
                probability_plot, life_stress_plot, print_results
            ]

        best_BIC, best_AICc, best_loglik = np.inf, np.inf, np.inf
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

        for item in include:
            res = distr[item](*function_parameters)
            results = results.append(
                {
                    "ALT_model": item,
                    "a": f'{res.a:.4f}' if 'a' in dir(res) else '',
                    "b": f'{res.b:.4f}' if 'b' in dir(res) else '',
                    "c": f'{res.c:.4f}' if 'c' in dir(res) else '',
                    "n": f'{res.n:.4f}' if 'n' in dir(res) else '',
                    "beta": f'{res.beta:.4f}' if 'beta' in dir(res) else '',
                    "sigma": f'{res.sigma:.4f}' if 'sigma' in dir(res) else '',
                    "Log-LH": f'{res.loglik:.4f}',
                    "AICc": f'{res.AICc:.4f}',
                    "BIC": f'{res.BIC:.4f}',
                },
                ignore_index=True,
            )
            if (res.BIC < best_BIC and metric == 'BIC') or \
               (res.AICc < best_AICc and metric == 'AICc') or \
               (-res.loglik < best_loglik and metric == 'Log-likelihood'):
                best_BIC = res.BIC
                best_AICc = res.AICc
                best_loglik = -res.loglik
                best_model = res
                best_model_name = item

        st.write('## Results of all fitted ALT models')
        results = pd.DataFrame.from_dict(results)
        st.write(results)

        st.write('## Results of the best fitted ALT model')
        if use_level:
            st.write(f'**{best_model_name}**, whose mean life \
                       is **{best_model.mean_life:.4f}** time units.')
        else:
            st.write(f'**{best_model_name}**')
        st.write(best_model.results)

        probability_plot = best_model.probability_plot.figure
        life_stress_plot = best_model.life_stress_plot.figure
        col1, col2 = st.columns(2)
        col1.write(probability_plot)
        col2.write(life_stress_plot)
