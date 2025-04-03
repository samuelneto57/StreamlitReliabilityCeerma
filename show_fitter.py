import streamlit as st
import numpy as np
import pandas as pd
from reliability.Fitters import Fit_Everything

import distributions
import functions


def show():
    st.write("""
    In this module, you can provide your data (complete or incomplete) and fit
    either the most common parametric probability distributions in reliability
    or some non-parametric models.
    """)

    with st.expander("Data format"):
        st.info("""
        Upload an excel file that contains the following columns:
        * Time - failure or right-censored time;
        * Type - 'F' for failure time, 'C' for right-censored time.
        """)
        df = {'Time': [10, 15, 8, 20, 21, 12, 13, 30, 5],
            'Type': ['F', 'F', 'C', 'F', 'C', 'C', 'F', 'F', 'C']}
        df = pd.DataFrame.from_dict(df)
        st.write(df)

    header = st.checkbox("Does your data contain header?", value=True)
    head = 0 if header else None

    col2_1, col2_2 = st.columns(2)
    uploaded_file = col2_1.file_uploader("Upload a XLSX file:",
                                         type="xlsx",
                                         accept_multiple_files=False,
                                         label_visibility="collapsed")
    if uploaded_file:
        df = pd.read_excel(uploaded_file, names=['Time', 'Type'], header=head)
        col2_2.dataframe(df, use_container_width=True)
        df['Type'] = df['Type'].str.upper()
        fdata = df[df['Type'] == 'F']
        cdata = df[df['Type'] == 'C']
        fdata = np.array(fdata.iloc[:,0])
        cdata = np.array(cdata.iloc[:,0])

    mod = st.selectbox("Which type of models would you like to fit?",
                       ("Parametric", "Non-parametric"))

    if mod == "Parametric":

        distr = distributions.distributions
        fitdistr = distributions.fit_distributions

        include = st.multiselect('Choose which distribution(s) you want to \
                                 fit your data to:', list(distr))

        metric, method = functions.fit_options()

        plot_params = functions.plot_parameters(CI=True, IC_plot=['CDF', 'SF'])

        exclude = list(set(list(distr))-set(include))

        exc = []
        for dis in exclude:
            for fitdis in fitdistr:
                if fitdis[:4] in dis:
                    exc.append(fitdis)

    elif mod == "Non-parametric":

        distr = distributions.non_parametric_distributions

        col1, col2 = st.columns([1,1])

        include = col1.selectbox('Choose which distribution you want to \
                                 fit your data to:', list(distr))

        ci = col2.number_input('Confidence Interval:',
                               min_value=0.0, max_value=1.0, value=0.95)

        par = st.checkbox('Do you want to compare the results with \
                          a parametric distribution?')

        if par == True:
            metric, method = functions.fit_options()

        plot_params = functions.plot_parameters(n_points=False, stats=False,
                                                IC_plot=['CDF', 'SF', 'CHF'],
                                                np=True)

    if st.button("Fit distribution"):

        if not uploaded_file:
            st.error('Please upload a file first!')
            st.stop()

        if mod == 'Parametric':

            if not include:
                st.error('Please, choose at least one distribution to fit.')
                st.stop()

            exc = exc if exc else None

            results = Fit_Everything(failures=fdata,
                                     right_censored=cdata,
                                     exclude=exc,
                                     sort_by=metric, method=method,
                                     print_results=False,
                                     show_histogram_plot=False,
                                     show_PP_plot=False,
                                     show_probability_plot=False)

            st.write('### Results of all fitted distributions')
            results_table = results.results.set_index('Distribution')
            results_table = results_table.mask(results_table=='').astype(float)
            results_table = results_table.fillna('')
            for i in range(len(results_table.iloc[:,0])):
                for j in range(len(results_table.iloc[0,:])):
                    try:
                        results_table.iloc[i,j] = f'{results_table.iloc[i,j]:.{plot_params["decimals"]}f}'
                    except ValueError:
                        results_table.iloc[i,j] = ''
            st.dataframe(results_table, use_container_width=True)

            dist = results.best_distribution
            distribution_name = results.best_distribution_name
            st.write(f'### Results of the best fitted distribution: {distribution_name}')

            percentiles = np.linspace(1, 99, num=99)

            new_fit = fitdistr[distribution_name][0](
                failures=fdata, right_censored=cdata,
                show_probability_plot=False, print_results=False,
                percentiles=percentiles, method=method,
                CI=plot_params['confidence_interval']
            )

            if 'Exponential' in distribution_name:
                new_fit.results = new_fit.results.drop(1, axis=0)

            new_fit.results = new_fit.results.set_index('Parameter')

            params_upper = list(new_fit.results.iloc[:,3])
            params_lower = list(new_fit.results.iloc[:,2])
            fit_up = fitdistr[distribution_name][1]['distribution'](
                *params_upper
            )
            fit_lw = fitdistr[distribution_name][1]['distribution'](
                *params_lower
            )

            functions.plot_distribution(dist, plot_params,
                                        title='Evaluated data',
                                        sidetable=new_fit.results,
                                        plot_dists_up_lw=plot_params['up_lw'],
                                        dist_upper=fit_up,
                                        dist_lower=fit_lw,
                                        failure_data=fdata,
                                        censored_data=cdata)

        elif mod == "Non-parametric":

            dist = distr[include](failures=fdata, right_censored=cdata, CI=ci,
                                  show_plot=False, print_results=False)

            # Parametric comparison
            if par == True:
                results = Fit_Everything(failures=fdata,
                                         right_censored=cdata,
                                         sort_by=metric, method=method,
                                         print_results=False,
                                         show_histogram_plot=False,
                                         show_PP_plot=False,
                                         show_probability_plot=False)
                par_params = [results]
            else:
                par_params = None

            functions.plot_distribution(dist, plot_params,
                                        title='Evaluated data',
                                        plot_dists=['CDF', 'SF', 'CHF'],
                                        plot_dists_up_lw=plot_params['up_lw'],
                                        non_parametric=True, par=par_params,
                                        failure_data=fdata,
                                        censored_data=cdata)
