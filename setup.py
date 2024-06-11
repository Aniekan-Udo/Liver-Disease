from setuptools import find_packages,setup
from typing import List


hyphenE="-e ."

def get_requirement(file_path:str)->List[str]:
    """
    This will return the list of requirements
    """
    requirements=[]
    
    with open(file_path) as file_obj:
        requirements=file_obj.readline()
        requirements=[req.replace("\n", "") for req in requirements]

    if hyphenE in requirements:
        requirements.remove(hyphenE)

        return requirements



setup(
    name="Liver Disease",
    version='0.0.1',
    author='Aniekan Udo',
    author_email='aniekanudo476@gmail.com',
    packages=find_packages(),
    install_requires=get_requirement("requirements.txt")
)