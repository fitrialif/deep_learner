from setuptools import setup, find_packages

setup(
    name='Deep Learner',
    version='0.1.0',
    description='Deep Learner wrapper for Keras',
    long_description=readme,
    author='Angelo Cardellicchio',
    author_email='a.cardellicchio@aeflab.net',
    url='https://github.com/anhelus',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)