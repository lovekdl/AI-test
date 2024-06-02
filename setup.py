from setuptools import setup, find_packages
# print(find_packages(where='src'))
setup(
    name='llm_coding_test',
    version='0.0.1',
    author='Renjia Deng',
    author_email='drjrenjia@gmail.com',
    license='MIT License',
    description='THU llm test.',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.10',
    install_requires=[line.strip() for line in open('requirements.txt', encoding='utf-8')],
    entry_points={
        "console_scripts": [],
    },
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
    ],
)
