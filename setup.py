from setuptools import setup
import versioneer

setup(
    name='Reliability CEERMA',
    author='CEERMA UFPE',
    url='https://reliabilityceerma.streamlit.app/',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
)
