import streamlit as st
from reliability.Distributions import Mixture_Model, Competing_Risks_Model

import distributions
import functions


def show():
    st.write("""
    In this module, you can combine two or more parametric distributions.
    """)

    with st.expander('Short Guide'):
        st.write("""
        Mixture models are a combination of two or more distributions by
        multiplying each of the combining distributions' PDFs (and,
        equivalently, CDFs) by a proportion, and the sum of all the proportions
        must be equal to 1. In simple terms, its a weighted sum of
        distributions' PDFs (and CDFs). If all the proportions do not
        sum to 1, each weight is divided by the sum to normalize them.

        Competing risks models are a combination of two or more distributions
        by multiplying the SFs (survival, or reliability, functions). This is
        similar to Mixture models, but the components here are instead
        "competing" to the end of the life of the system being modelled.

        In both cases, those models are useful when there is more than one
        failure mode that is generating the failure data. This can be
        recognised by the shape of the PDF, CDF and/or SF being outside of what
        any single distribution can accurately model. You should not use them
        just because it can fit almost anything really well, but instead if you
        suspect that there are multiple failure modes contributing to the
        failure data you are observing. To judge whether a mixture model is
        justified, look at the goodness of fit criterion (AICc or BIC) which
        penalises the score based on the number of parameters in the model. The
        closer the goodness of fit criterion is to zero, the better the fit.
        Using AD or log-likelihood for this check is not appropriate as these
        goodness of fit criterions do not penalise the score based on the
        number of parameters in the model and are therefore prone to
        overfitting.
        """)

    col1, col2 = st.columns([1,1])

    Model_selected = col1.selectbox('Select method:',
                                    ('Mixture Model', 'Competing Risk Model'))

    number_distributions = col2.number_input("N° of distributions:", value=1,
                                             min_value=1, max_value=20)
    number_distributions =  int(number_distributions)

    init_val= 1/number_distributions

    distr = distributions.distributions

    selected_weights = []
    selected_dists = []

    for i in range(number_distributions):
        if Model_selected == 'Mixture Model':
            cols = st.columns([2,1,1,1,1])
        else:
            cols = st.columns([3,1,1,1])
        distribution_name = cols[0].selectbox('Select distribution:',
                                                list(distr),
                                                key=f"dist_{i}")

        var = []
        variables = distr[distribution_name]['variables']
        for j, variable in enumerate(variables):
            item = variables[variable]
            to_append = cols[j+1].number_input(variable[-1],
                                                value=item[0],
                                                min_value=item[1],
                                                step=0.1,
                                                format='%0.2f',
                                                key=f'var_{j}_{i}')
            var.append(to_append)

        if Model_selected == 'Mixture Model':
            selected_weights.append(cols[4].number_input("Proportion",
                                                         value=init_val,
                                                         min_value=0.0,
                                                         max_value=1.0,
                                                         key=f'weigth_{i}'))

        selected_dists.append(distr[distribution_name]['distribution'](*var))

    if Model_selected == 'Mixture Model':
        # Fix rounding problems by normalizing weights
        total = 0.0
        if number_distributions > 1:
            sum_weights = sum(selected_weights)
            for i in range(1, len(selected_weights)):
                new = selected_weights[i] / sum_weights
                selected_weights[i] = new
                total = total + new
        selected_weights[0] = 1 - total

    plot_params = functions.plot_parameters()

    if st.button("Plot distribution"):

        if Model_selected == 'Mixture Model':
            dist = Mixture_Model(distributions=selected_dists,
                                 proportions=selected_weights)
        if Model_selected == 'Competing Risk Model':
            dist = Competing_Risks_Model(distributions=selected_dists)

        functions.plot_distribution(dist, plot_params, title='Combined model')
