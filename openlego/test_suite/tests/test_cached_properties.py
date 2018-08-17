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

This file contains the definition of the test case for the cached property functionality.
"""
from __future__ import absolute_import, division, print_function

import unittest
from cached_property import cached_property


class SimpleCachedPropertyClass(object):

    def __init__(self, an_integer):
        # type: (int) -> None
        super(SimpleCachedPropertyClass, self).__init__()
        self._my_integer = an_integer

    def invalidate(self):
        # type: () -> None
        __dict__ = self.__class__.__dict__.copy()
        __dict__.update(SimpleCachedPropertyClass.__dict__)
        for name, value in __dict__.items():
            if isinstance(value, cached_property):
                if name in self.__dict__:
                    del self.__dict__[name]

    @cached_property
    def my_integer(self):
        # type: () -> int
        return self._my_integer


class SubCachedPropertyClass(SimpleCachedPropertyClass):

    def __init__(self, an_integer, a_string):
        # type: (int, str) -> None
        super(SubCachedPropertyClass, self).__init__(an_integer)
        self._my_string = a_string

    @cached_property
    def my_string(self):
        # type: () -> str
        return self._my_string


class TestCachedProperty(unittest.TestCase):

    def test_cached_properties(self):
        one = SimpleCachedPropertyClass(1)
        self.assertEqual(one.my_integer, one._my_integer)
        self.assertEqual(one.my_integer, 1)
        one.invalidate()
        self.assertEqual(one.my_integer, 1)
        one._my_integer = 2
        self.assertEqual(one.my_integer, 1)
        one.invalidate()
        self.assertEqual(one.my_integer, 2)

        two = SimpleCachedPropertyClass(2)
        self.assertEqual(two.my_integer, two._my_integer)
        self.assertEqual(two.my_integer, 2)

        three = SubCachedPropertyClass(3, 'three')
        self.assertEqual(three.my_integer, 3)
        self.assertEqual(three.my_string, 'three')
        three._my_integer = 4
        self.assertEqual(three.my_integer, 3)
        self.assertEqual(three.my_string, 'three')
        three.invalidate()
        self.assertEqual(three.my_integer, 4)
        self.assertEqual(three.my_string, 'three')
        three._my_string = 'four'
        self.assertEqual(three.my_string, 'three')
        three.invalidate()
        self.assertEqual(three.my_string, 'four')


if __name__ == '__main__':
    unittest.main()
