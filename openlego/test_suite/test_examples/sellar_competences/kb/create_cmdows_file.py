from kadmos.cmdows import CMDOWS

cmdows = CMDOWS()

dcs = ['D1', 'D2', 'F', 'G1', 'G2']


cmdows.add_header('Imco van Gent', 'CMDOWS file for the Sellar database.', '2018-05-23T11:19:56.287006',
                  fileVersion='1.0')

cmdows.add_contact('Imco van Gent', 'i.vangent@tudelft.nl', 'imcovangent',
                   company='TU Delft',
                   department='Flight Performance and Propulsion',
                   function='PhD Student',
                   address='Kluyverweg 1, 2629 HS, Delft',
                   country='The Netherlands',
                   telephone='0031 6 53 89 42 75',
                   roles=['architect', 'integrator'])
cmdows.add_contact('Daniel de Vries', 'danieldevries6@gmail.com', 'danieldevries',
                   company='TU Delft',
                   country='USA',
                   roles=['tool_specialist'])

for dc in dcs:
    cmdows.add_dc(dc, dc, 'main', instance_id=1, version='1.0', label=dc)

    cmdows.add_dc_general_info(dc,
                               description='{} discipline of the Sellar tool set.'.format(dc),
                               status='Available',
                               owner_uid='danieldevries',
                               creator_uid='imcovangent',
                               operator_uid='imcovangent')
    cmdows.add_dc_licensing(dc,
                            license_type='open-source',
                            license_specification='Apache License 2.0',
                            license_info='https://www.apache.org/licenses/LICENSE-2.0')

cmdows.save('__cmdows__Sellar.xml', pretty_print=True)