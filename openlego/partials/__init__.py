#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright 2018 D. de Vries

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file contains constants for the partials package.
"""
from __future__ import absolute_import, division, print_function

import os

from lxml import etree

dir_path = os.path.dirname(os.path.abspath(__file__))
xsd_file_path = os.path.join(dir_path, 'partials.xsd')
xsi_schema_location = 'file:///' + xsd_file_path

schema = etree.XMLSchema(file=xsi_schema_location)
parser = etree.XMLParser(schema=schema)
