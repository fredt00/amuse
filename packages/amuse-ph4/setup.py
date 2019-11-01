import sys
import os
from support.version import version, main_version
from support.classifiers import classifiers

from setuptools import setup

import support
support.use("system")
from support.setup_codes import setup_commands

name = 'amuse-ph4'
author = 'The AMUSE team'
author_email = 'info@amusecode.org'
license_ = "Apache License 2.0"
url = 'http://www.amusecode.org/'
install_requires = [
    'amuse-framework>=%s' % (main_version),
]
description = 'The Astrophysical Multipurpose Software Environment - ph4'
with open("README.md", "r") as fh:
    long_description = fh.read()
long_description_content_type = "text/markdown"

extensions = []

all_data_files = []

packages = ['amuse.community.ph4']

package_data = {
}

mapping_from_command_name_to_command_class=setup_commands()

setup(
    name=name,
    version=version,
    classifiers=classifiers,
    url=url,
    author_email=author_email,
    author=author,
    license=license_,
    description=description,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    install_requires=install_requires,
    cmdclass=mapping_from_command_name_to_command_class,
    ext_modules=extensions,
    package_dir={'amuse.community.ph4': 'src/amuse/community/ph4'},
    packages=packages,
    package_data=package_data,
    data_files=all_data_files,
)
