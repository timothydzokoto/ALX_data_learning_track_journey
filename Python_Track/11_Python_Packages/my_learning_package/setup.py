from setuptools import setup, find_packages

setup(
    name='mypackage',
    version='0.2',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='EDSA example python package',
    long_description=open('README.md').read(),
    install_requires=['numpy'],
    url='https://github.com/timothydzokoto/my_learning_package',
    author='Timothy Dzokoto',
    author_email='timdzok@gmail.com',

    entry_points= {
        'console_scripts':[
            'mypackage = mypackage:myModule' 
        ]
    }
)

