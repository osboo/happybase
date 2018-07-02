"""
HappyBase utility tests.
"""

from codecs import decode, encode

import pytest

import happybase.util as util

examples = [('foo', 'Foo', 'foo'),
            ('fooBar', 'FooBar', 'foo_bar'),
            ('fooBarBaz', 'FooBarBaz', 'foo_bar_baz'),
            ('fOO', 'FOO', 'f_o_o')]


@pytest.mark.parametrize("lower_cc,upper_cc,correct", examples)
def test_camel_case_to_pep8(lower_cc, upper_cc, correct):
    x1 = util.camel_case_to_pep8(lower_cc)
    x2 = util.camel_case_to_pep8(upper_cc)
    assert correct == x1
    assert correct == x2


@pytest.mark.parametrize("lower_cc,upper_cc,pep8", examples)
def test_pep8_to_camel_case(lower_cc, upper_cc, pep8):
    y1 = util.pep8_to_camel_case(pep8, True)
    y2 = util.pep8_to_camel_case(pep8, False)
    assert upper_cc == y1
    assert lower_cc == y2


bytes_test_values = [
    (b'00', b'01'),
    (b'01', b'02'),
    (b'fe', b'ff'),
    (b'1234', b'1235'),
    (b'12fe', b'12ff'),
    (b'12ff', b'13'),
    (b'424242ff', b'424243'),
    (b'4242ffff', b'4243'),
]


@pytest.mark.parametrize("s_hex,expected", bytes_test_values)
def test_bytes_increment(s_hex, expected):
    s = decode(s_hex, 'hex')
    v = util.bytes_increment(s)
    v_hex = encode(v, 'hex')
    assert expected == v_hex
    assert s < v


def test_bytes_increment_none():
    assert util.bytes_increment(b'\xff\xff\xff') is None


def test_ensure_bytes_exception():
    with pytest.raises(TypeError):
        util.ensure_bytes(123)
