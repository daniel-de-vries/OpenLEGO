"""
This file represents the G2 discipline of the Sellar problem:

    g2 = 1 - y2/24

Call using:
    python g2.py input.xml output.xml

The script is written for Python 2.7!

Copyright 2018, DLR, Jasper Bussemaker.
Licence: MIT
"""
import argparse
from lxml import etree
from openlego.utils.xml_utils import xml_safe_create_element


def run(input_xml_file, output_xml_file):
    # Load input file
    tree = etree.parse(input_xml_file)
    root = tree.getroot()

    # Load input parameters for this discipline
    y2 = float(root.find('./analyses/y2').text)

    # "Run" the discipline calculations
    g2 = 1. - (y2/24.)

    # Write the output
    xml_safe_create_element(tree, '/dataSchema/output/g2', str(g2))

    tree.write(output_xml_file)


if __name__ == '__main__':
    # Define the command line arguments
    parser = argparse.ArgumentParser('Sellar G2')
    parser.add_argument('input_xml', help='The input XML file containing the Sellar central data schema')
    parser.add_argument('output_xml', help='The output XML file that will be written by this script')
    args = parser.parse_args()

    # Run the discipline
    run(args.input_xml, args.output_xml)
