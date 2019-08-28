from setuptools import setup, find_packages
import os


with open(os.path.join(os.path.dirname(__file__), "swdb_2019_tools", "VERSION.txt"), "r") as version_file:
    version = version_file.read().strip()


def find_all_packages(*roots):
    out = []
    for root in roots:
        for subdir, _, _ in os.walk(root):
            out.extend([subdir.strip(".").replace("/", ".")])
    return out


setup(
    version = version,
    name = 'swdb_2019_tools',
    author = 'Michael Buice',
    author_email = 'mabuice@alleninstitute.org',
    packages = find_all_packages("swdb_2019_tools"),
    package_data={'': ['*.conf', '*.cfg', '*.md', '*.json', '*.dat', '*.env', '*.sh', '*.txt', 'bps', 'Makefile', 'LICENSE', '*.hoc'] },
    description = '',
    setup_requires=["setuptools"],
    url=f'https://github.com/AllenInstitute/swdb_2019_tools/tree/v{version}',
    download_url = f'https://github.com/AllenInstitute/swdb_2019_tools/tarball/v{version}',
    keywords = ['neuroscience', 'bioinformatics', 'scientific' ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License', # Allen Institute Software License
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6', 
        'Topic :: Scientific/Engineering :: Bio-Informatics'
        ])