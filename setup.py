from setuptools import setup, find_packages

setup(
    name='atm_retrieval',
    version='0.1.0',
    description='A package for atmospheric retrieval',
    author='Dario Gonzalez Picos',
    author_email='picos@strw.leidenuniv.nl',
    packages=find_packages(include=['atm_retrieval', 'atm_retrieval.*']),
    install_requires=[
        # Add your project's dependencies here
        # For example:
        # 'numpy',
        # 'petitRADTRANS',
        # 'pymultinest',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
)