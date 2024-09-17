import streamlit as st

import distributions
import functions


def show():
    st.write("""
    In this module, you can select and analyze the most common
    probability distributions in reliability analysis.
    """)

    distr = distributions.distributions
    distribution_name = st.selectbox('Select a distribution:',
                                     list(distr))

    with st.expander('Equation Information'):
        for equat in distr[distribution_name]['equations']:
            st.latex(equat)

    dist_print = ''
    var = []
    variables = distr[distribution_name]['variables']
    cols = st.columns([1]*len(variables))
    for i, variable in enumerate(variables):
        to_append = cols[i].number_input(
            variable,
            value=variables[variable][0],
            min_value=variables[variable][1],
            step=0.1,
            format='%0.5f'
        )
        var.append(to_append)
        dist_print = dist_print + f'{variable[-1]}: {to_append}\n'

    plot_params = functions.plot_parameters()

    if st.button("Plot distribution"):

        dist = distr[distribution_name]['distribution'](*var)

        st.write(dist_print)

        if distribution_name == "Beta Distribution" \
            and (var[0] <=1 or var[1] <=1):
            plot_params['plot_mean'] = False
        if distribution_name == "Normal Distribution":
            plot_params['plot_median'] = False
            plot_params['plot_mode'] = False

        if distribution_name == "Loglogistic Distribution" and var[1] <=1:
            functions.plot_distribution(
                dist, plot_params,
                title=dist.param_title_long,
                plot=False
            )
            st.write("No plot when beta is less than or equal to 1")
        else:
            functions.plot_distribution(
                dist, plot_params, title=dist.param_title_long
            )
