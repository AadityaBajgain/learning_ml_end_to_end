from setuptools import find_packages, setup
from typing import List


HYPHEN_DOT_E = "-e ."

def get_requirements(requirement_dir) -> List[str]:
    """
    Docstring for get_requirements
    
    :param requirement_dir: path of the requirements.txt
    :return: list of string
    :rtype: List[str]
    """

    requirements = []
    with open (requirement_dir) as file_obj:
        requirements = file_obj.readlines()

        requirements = [req.replace("\n","") for req in requirements]
        
        if HYPHEN_DOT_E in requirements:
            requirements.remove(HYPHEN_DOT_E)
    
    return requirements



setup(
    name="ML_END_TO_END",
    version="0.0.1",
    author="Aaditya Bajgain",
    author_email="adityabajgain@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements("requirements.txt"),
)