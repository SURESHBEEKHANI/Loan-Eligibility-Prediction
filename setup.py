from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]  # Use strip() to remove whitespace and newlines

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name="Loan Approval Prediction",
    version="1.0.0",
    packages=find_packages(),
    # Package metadata
    author="Suresh Beekhani",
    author_email="sureshbeekhani26@gmail.com",
    description="A tool for predicting loan approval based on various financial and demographic factors.",
    long_description="This package provides a comprehensive model for predicting loan approval using data analysis and machine learning techniques.",
    long_description_content_type="text/markdown",
    url="https://github.com/SURESHBEEKHANI/LOAN-APPROVAL-PREDICTION.git",  # Updated URL to match your project
    # Read dependencies from requirements.txt
    install_requires=get_requirements('requirements.txt'),
    # License and classification
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
 