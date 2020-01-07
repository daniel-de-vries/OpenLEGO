import os
from openlego.core.abstract_discipline import AbstractDiscipline
from openlego.test_suite.test_examples.sellar_competences_virtual.sellar_code import d1, d2, d12, f1, g1, g2

__all__ = ['D1Discipline', 'D2Discipline', 'D12Discipline', 'F1Discipline', 'G1Discipline', 'G2Discipline']


class AbstractSellarDiscipline(AbstractDiscipline):

    _io_path = os.path.join(os.path.dirname(__file__), 'io')

    @staticmethod
    def _read_xml(path):
        with open(path, 'rb') as fp:
            return fp.read()

    def generate_input_xml(self):
        return self._read_xml(os.path.join(self._io_path, self.name+'-input.xml'))

    def generate_output_xml(self):
        return self._read_xml(os.path.join(self._io_path, self.name+'-output.xml'))

    @staticmethod
    def execute(in_file, out_file):
        raise NotImplementedError


class D1Discipline(AbstractSellarDiscipline):

    @property
    def name(self):
        return 'D1'

    @staticmethod
    def execute(in_file, out_file):
        d1.run(in_file, out_file)


class D2Discipline(AbstractSellarDiscipline):

    @property
    def name(self):
        return 'D2'

    @staticmethod
    def execute(in_file, out_file):
        d2.run(in_file, out_file)


class D12Discipline(AbstractSellarDiscipline):

    @property
    def name(self):
        return 'D12'

    @staticmethod
    def execute(in_file, out_file):
        d12.run(in_file, out_file)


class F1Discipline(AbstractSellarDiscipline):

    @property
    def name(self):
        return 'F1'

    @staticmethod
    def execute(in_file, out_file):
        f1.run(in_file, out_file)


class G1Discipline(AbstractSellarDiscipline):

    @property
    def name(self):
        return 'G1'

    @staticmethod
    def execute(in_file, out_file):
        g1.run(in_file, out_file)


class G2Discipline(AbstractSellarDiscipline):

    @property
    def name(self):
        return 'G2'

    @staticmethod
    def execute(in_file, out_file):
        g2.run(in_file, out_file)
