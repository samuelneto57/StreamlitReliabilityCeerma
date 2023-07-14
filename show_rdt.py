import streamlit as st
import numpy as np
from math import comb, gamma
from scipy.optimize import fsolve
from functools import partial

import distributions


help_text = 'Please use the same time measurement unit for all time entries'


def combination(n, i):
    if isinstance(n, int):
        return comb(n, i)
    else:
        return gamma(n + 1) / (gamma(i + 1) * gamma(n - i + 1))


def risk(x, *, t_test=None, n_units=None, reliab_target, t_mission,
         allowed_failures, ac_factor, beta, lower_confidence):

    if t_test:
        n_units = x
    elif n_units:
        t_test = x
    else:
        raise 'At least one of t_test or n_units must be passed'

    # Initial risk is the lower level confidence minus 1 (shift to right side)
    risk = lower_confidence - 1

    # Sum risk based on maximum number of failures allowed in the test
    for i in range(allowed_failures+1):

        term = np.exp(
            np.log(reliab_target) * (t_test * ac_factor / t_mission) ** beta
        )

        risk += combination(n_units, i) * (1 -  term)**i * term**(n_units - i)

    return risk


def rdt_eq(*, t_test=None, n_units=None, reliab_target, t_mission,
           allowed_failures, ac_factor, beta, lower_confidence):

    # Assert exactly one of t_test or n_units must be given
    assert ((t_test is not None) ^ (n_units is not None))

    # Write function keyword-only parameters as dict
    params = dict(reliab_target=reliab_target, t_mission=t_mission,
                  allowed_failures=allowed_failures, ac_factor=ac_factor,
                  beta=beta, lower_confidence=lower_confidence)

    # Freeze function parameters
    risk_freeze = partial(risk, t_test=t_test, n_units=n_units, **params)

    # Solve non-frozen parameter (the one that is not none)
    return fsolve(risk_freeze, 0)[0]


def show():
    st.write("""
    In this module, you can better design reliability tests by finding
    the optimal test time or number of units to be tested, given certain
    defined test conditions. This
    """)

    with st.expander('Short Guide'):
        st.write("""
        The objective of Reliability Demonstration Tests (RDTs) is to
        demonstrate a certain reliability, with a specific confidence
        level, based on an allowed number of failures, in general
        zero or a few. If, for the real test when using the same
        variables used in the model, no more than the allowed number of
        failures occur, then the target reliability is demonstrated.

        Under the RDT in this module, it is possible to consider an
        accelerated testing scenario, in which the Stress Level of the
        test is higher than the Use-level Stress. This way, failures are
        expected to occur faster, which allows for a shorter testing
        window.

        This module implements RDT using the Binomial test design
        method and considering a Weibull distribution for the data.
        You could validade if your data comes from a Weibull
        distribution using the Fit Distribution module of this APP.
        """)

    with st.expander("Acceleration Factor Information"):
        st.info('The Acceleration Factor ($A_F$) can be calculated by:')
        st.latex(r'A_F = \frac{L_{use}}{L_{accelerated}}')
        st.info('Given a model, $A_F$ can be calculated by stress levels:')
        for item in distributions.acceleration_factor_equations:
            st.latex(item)
        st.info('Where B is a parameter that controls the behaviour of models')

    params = {}

    cols = st.columns([1,1])

    options = ('Test Time (under accelerated condition)',
               'Number of Specimens to be tested')

    approach = cols[0].radio('Which test variable you want to estimate?',
                             options)

    if approach == options[0]:
        n_test = cols[1].number_input('Fixed number of specimens to be tested',
                                      min_value=0, value=1, step=1)
        params['n_units'] = n_test
    elif approach == options[1]:
        t_test = cols[1].number_input('Fixed accelerated test time',
                                      min_value=0.1, value=1.0,
                                      step=1.0, format='%0.2f',
                                      help=help_text)
        params['t_test'] = t_test

    mission_time = st.number_input('Mission time (under nominal condition)',
                    min_value=0.0, value=87600.0, step=1.0, format='%0.2f',
                    help=help_text)
    params['t_mission'] = mission_time

    cols = st.columns([1,1])

    Rt = cols[0].number_input('Reliability target for mission',
                              min_value=0.0, max_value=1.0, value=0.9,
                              step=0.01, format='%0.2f')
    params['reliab_target'] = Rt

    llc = cols[1].number_input('Lower level confidence for mission',
                               min_value=0.0, max_value=1.0, value=0.8,
                               step=0.01, format='%0.2f')
    params['lower_confidence'] = llc

    cols = st.columns([1,1])

    n_fail = cols[0].number_input('Allowed number of failures in accelerated test',
                             min_value=0, step=1)
    params['allowed_failures'] = n_fail

    beta = cols[1].number_input('Shape parameter of Weibull distribution',
                           min_value=0.0, max_value=10.0, value=1.0,
                           step=1.0, format='%0.2f')
    params['beta'] = beta

    use_af = st.checkbox('Use Acceleration Factor?')

    if use_af:
        cols = st.columns([1,1])

        model = cols[0].radio('Acceleration factor model to be used:',
                            ('Arrhenius', 'Eyring', 'Inverse Power Law', 'Other'))

        if model != 'Other':
            v_use = cols[1].number_input('Use-level stress',
                                        min_value=0.1, value=1.0,
                                        step=1.0, format='%.02f')
            v_acc = cols[1].number_input('Accelerated stress',
                                        min_value=0.1, value=2.0,
                                        step=1.0, format='%.02f')
            v_coe = cols[1].number_input('Model parameter',
                                        min_value=0.0, value=1.0,
                                        step=1.0, format='%.02f')
            if model == 'Arrhenius':
                ac_factor = np.exp(v_coe * (1/v_use - 1/v_acc))
            elif model == 'Eyring':
                ac_factor = np.exp(v_coe * (1/v_use - 1/v_acc)) * v_acc / v_use
            elif model == 'Inverse Power Law':
                ac_factor = (v_acc / v_use) ** v_coe
        else:
            ac_factor = cols[1].number_input('Acceleration factor',
                                            min_value=0.01, value=2.0,
                                            step=1.0, format='%0.2f')
    else:
        ac_factor = 1.0

    params['ac_factor'] = ac_factor

    if st.button('Calculate'):

        rdt = rdt_eq(**params)

        if approach == options[0]:
            st.write(f'The optimal test time for this scenario is {rdt:.2f}')
        elif approach == options[1]:
            st.write(f'The optimal number of speciments to be tested is {rdt:.1f}')

