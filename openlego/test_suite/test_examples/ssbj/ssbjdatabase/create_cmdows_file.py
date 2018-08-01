from kadmos.cmdows import CMDOWS


def create_cmdows_file():

    cmdows = CMDOWS()

    dcs = ['StructuralAnalysis', 'AeroAnalysis', 'PropulsionAnalysis', 'PerformanceAnalysis']


    cmdows.add_header('Imco van Gent', 'CMDOWS file for the SSBJ database.', '2018-05-23T11:19:56.287006',
                      fileVersion='1.0')

    cmdows.add_contact('Imco van Gent', 'i.vangent@tudelft.nl', 'imcovangent',
                       company='TU Delft',
                       department='Flight Performance and Propulsion',
                       function='PhD Student',
                       address='Kluyverweg 1, 2629 HS, Delft',
                       country='The Netherlands',
                       telephone='0031 6 53 89 42 75',
                       roles=['architect', 'integrator'])
    cmdows.add_contact('Remi Lafage', 'Remi.Lafage@onera.fr', 'remilafage',
                       company='ONERA - French Aerospace Lab',
                       country='France',
                       roles=['tool_specialist'])
    cmdows.add_contact('Sylvain Dubreuil', 'Sylvain.Dubreuil@onera.fr', 'sylvaindubreuil',
                       company='ONERA - French Aerospace Lab',
                       country='France',
                       roles=['tool_specialist'])

    for dc in dcs:
        cmdows.add_dc(dc, dc, 'main', instance_id=1, version='1.0', label=dc)

        cmdows.add_dc_general_info(dc,
                                   description='{} discipline of the SSBJ tool set.'.format(dc),
                                   status='Available',
                                   owner_uid='remilafage',
                                   creator_uid='sylvaindubreuil',
                                   operator_uid='imcovangent')
        cmdows.add_dc_licensing(dc,
                                license_type='open-source',
                                license_specification='Apache License 2.0',
                                license_info='https://www.apache.org/licenses/LICENSE-2.0')
        cmdows.add_dc_sources(dc,
                              repository_link='https://bitbucket.org/imcovangent/ssbj-kadmos/src/master/',
                              download_link='https://bitbucket.org/imcovangent/ssbj-kadmos/downloads/',
                              references=['ONERA SSBJ-OpenMDAO GitHub repository: https://github.com/OneraHub/SSBJ-OpenMDAO',
                                          'NASA report using SSBJ: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19980234657.pdf'])
        dc_folder = dc.lower() if dc not in dcs[4:6] else 'outputfunctions'
        dc_py_file = dc + '.py'
        cmdows.add_dc_execution_details(dc,
                                        operating_system='Windows',
                                        integration_platform='Optimus',
                                        command='cd %PROJECTDIR%\SSBJ\\n'
                                                '"ssbjkadmos\\tools\\{}\\{}"  -i "cpacsInputUpdated.xml" -o "cpacsOutput.xml"\\n'
                                                'echo doing some analysis\\n'
                                                'copy %PROJECTDIR%\\SSBJ\\cpacsOutput.xml %METHODDIR%\\cpacsOutput.xml'.format(dc_folder, dc_py_file),
                                        description='Details for the command line execution of the {} Python tool in Windows using an Optimus integration.'.format(dc),
                                        software_requirements=['Python 2.7.11 or higher installed',
                                                               'kadmos Python package version 0.8 or higher installed'],
                                        hardware_requirements=None)

    cmdows.save('__cmdows__SSBJ.xml', pretty_print=True)