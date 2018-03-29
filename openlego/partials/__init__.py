from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from lxml import etree

dir_path = os.path.dirname(os.path.abspath(__file__))
xsd_file_path = os.path.join(dir_path, 'partials.xsd')
xsi_schema_location = 'file:///' + xsd_file_path

schema = etree.XMLSchema(file=xsi_schema_location)
parser = etree.XMLParser(schema=schema)
