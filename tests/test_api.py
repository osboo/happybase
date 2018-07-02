"""
HappyBase tests.
"""

import collections
import os
import random
import threading

import pytest
import six
from six.moves import range

from happybase import Connection, ConnectionPool, NoConnectionsAvailable

HAPPYBASE_HOST = os.environ.get('HAPPYBASE_HOST')
HAPPYBASE_PORT = os.environ.get('HAPPYBASE_PORT')
HAPPYBASE_COMPAT = os.environ.get('HAPPYBASE_COMPAT', '0.98')
HAPPYBASE_TRANSPORT = os.environ.get('HAPPYBASE_TRANSPORT', 'buffered')
HAPPYBASE_USE_KERBEROS = os.environ.get('HAPPYBASE_USE_KERBEROS', 'false') == 'true'
HAPPYBASE_SASL_SERVICE_NAME = os.environ.get('HAPPYBASE_SASL_SERVICE_NAME', 'hbase')

TABLE_PREFIX = b'happybase_tests_tmp'
TEST_TABLE_NAME = b'test1'

connection_kwargs = dict(
    host=HAPPYBASE_HOST,
    port=HAPPYBASE_PORT,
    table_prefix=TABLE_PREFIX,
    compat=HAPPYBASE_COMPAT,
    transport=HAPPYBASE_TRANSPORT,
    use_kerberos=HAPPYBASE_USE_KERBEROS,
    sasl_service_name=HAPPYBASE_SASL_SERVICE_NAME
)

connection = table = None


def setup_module(module):
    module.connection = Connection(**connection_kwargs)
    assert module.connection is not None

    cfs = {
        'cf1': {},
        'cf2': None,
        'cf3': {'max_versions': 1},
    }

    if TEST_TABLE_NAME in module.connection.tables():
        module.connection.disable_table(TEST_TABLE_NAME)
        module.connection.delete_table(TEST_TABLE_NAME)

    module.connection.create_table(TEST_TABLE_NAME, families=cfs)
    module.table = module.connection.table(TEST_TABLE_NAME)
    assert module.table is not None


def teardown_module(module):
    module.connection.delete_table(TEST_TABLE_NAME, disable=True)
    module.connection.close()


def test_connection_stringify():
    assert connection.__str__() == connection.__repr__()


def test_compat():
    with pytest.raises(ValueError):
        Connection(compat='0.1.invalid.version')


def test_timeout_arg():
    Connection(
        timeout=1,
        autoconnect=False)


def test_table_enabling():
    assert connection.is_table_enabled(TEST_TABLE_NAME)
    connection.disable_table(TEST_TABLE_NAME)
    assert not connection.is_table_enabled(TEST_TABLE_NAME)
    connection.enable_table(TEST_TABLE_NAME)
    assert connection.is_table_enabled(TEST_TABLE_NAME)


@pytest.mark.parametrize("table_name", ['', 'foo'])
def test_table_name_autoconnect_true(table_name):
    assert TABLE_PREFIX + six.b('_' + table_name) == connection._table_name(table_name)


def test_table_name_autoconnect_false():
    c = Connection(autoconnect=False)
    assert b'foo' == c._table_name('foo')


@pytest.mark.parametrize("kwargs,exception", [
    (dict(table_prefix=123), TypeError),
    (dict(table_prefix=2.1), TypeError),
    (dict(transport='invalid'), ValueError),
    (dict(table_prefix_separator=123), TypeError),
    (dict(protocol='invalid'), ValueError)
])
def test_connection_init_exceptions(kwargs, exception):
    with pytest.raises(exception):
        Connection(autoconnect=False, **kwargs)


def test_table_name_use_prefix_true():
    assert connection.table('foobar').name == TABLE_PREFIX + b'_foobar'


def test_table_name_use_prefix_false():
    assert connection.table('foobar', use_prefix=False).name == b'foobar'


def test_compact_table():
    connection.compact_table(TEST_TABLE_NAME)
    connection.compact_table(TEST_TABLE_NAME, major=True)
    assert True


def test_table_listing():
    names = connection.tables()
    assert isinstance(names, list)
    assert TEST_TABLE_NAME in names


@pytest.mark.parametrize("families,error", [
    ({}, ValueError),
    (0, TypeError),
    ([], TypeError)
])
def test_invalid_table_create(families, error):
    with pytest.raises(error):
        connection.create_table('sometable', families=families)


def test_table_stringify():
    assert table.__str__() == table.__repr__()


def test_table_regions():
    regions = table.regions()
    assert isinstance(regions, list)


def test_families():
    families = table.families()
    for name, fdesc in six.iteritems(families):
        assert isinstance(name, bytes)
        assert isinstance(fdesc, dict)
        assert 'name' in fdesc
        assert isinstance(fdesc['name'], six.binary_type)
        assert 'max_versions' in fdesc


@pytest.mark.parametrize("kwargs", [
    dict(columns=123),
    dict(timestamp='invalid')
])
def test_row_exceptions(kwargs):
    with pytest.raises(TypeError):
        table.row(b'row-test', kwargs)


def test_row():
    row_key = b'row-test'
    table.put(row_key, {b'cf1:col1': b'v1old'}, timestamp=1234)
    table.put(row_key, {b'cf1:col1': b'v1new'}, timestamp=3456)
    table.put(row_key, {b'cf1:col2': b'v2',
                        b'cf2:col1': b'v3'})
    table.put(row_key, {b'cf2:col2': b'v4'}, timestamp=1234)

    exp_data = {b'cf1:col1': b'v1new',
                b'cf1:col2': b'v2',
                b'cf2:col1': b'v3',
                b'cf2:col2': b'v4'}

    assert exp_data == table.row(row_key)

    exp_cols_1 = {b'cf1:col1': b'v1new',
                  b'cf1:col2': b'v2'}
    assert exp_cols_1 == table.row(row_key, [b'cf1'])

    exp_cols_2 = {b'cf1:col1': b'v1new',
                  b'cf2:col2': b'v4'}
    assert exp_cols_2 == table.row(row_key, [b'cf1:col1', b'cf2:col2'])

    exp_cols_timestamp_1 = {b'cf1:col1': b'v1old',
                            b'cf2:col2': b'v4'}
    assert exp_cols_timestamp_1 == table.row(row_key, timestamp=2345)
    assert {} == table.row(row_key, timestamp=123)  # empty timestamp check

    res = table.row(row_key, include_timestamp=True)
    assert len(res) == 4
    assert b'v1new' == res[b'cf1:col1'][0]
    assert isinstance(res[b'cf1:col1'][1], int)


def test_atomic_counters():
    row = 'row-with-counter'
    column = 'cf1:counter'

    assert 0 == table.counter_get(row, column)

    assert 10 == table.counter_inc(row, column, 10)
    assert 10 == table.counter_get(row, column)

    table.counter_set(row, column, 0)
    assert 1 == table.counter_inc(row, column)
    assert 4 == table.counter_inc(row, column, 3)
    assert 4 == table.counter_get(row, column)

    table.counter_set(row, column, 3)
    assert 3 == table.counter_get(row, column)
    assert 8 == table.counter_inc(row, column, 5)
    assert 6 == table.counter_inc(row, column, -2)
    assert 5 == table.counter_dec(row, column)
    assert 3 == table.counter_dec(row, column, 2)
    assert 10 == table.counter_dec(row, column, -7)


@pytest.mark.parametrize("kwargs,exception", [
    (dict(batch_size=0), ValueError),
    (dict(transaction=True, batch_size=10), TypeError),
    (dict(timestamp='invalid'), TypeError)
])
def test_batch_exceptions(kwargs, exception):
    with pytest.raises(exception):
        table.batch(**kwargs)


def test_batch():
    b = table.batch()
    b.put(b'row1', {b'cf1:col1': b'value1',
                    b'cf1:col2': b'value2'})
    b.put(b'row2', {b'cf1:col1': b'value1',
                    b'cf1:col2': b'value2',
                    b'cf1:col3': b'value3'})
    b.delete(b'row1', [b'cf1:col4'])
    b.delete(b'another-row')
    b.send()

    row1_expected = {b'cf1:col1': b'value1',
                     b'cf1:col2': b'value2'}
    row2_expected = {b'cf1:col1': b'value1',
                     b'cf1:col2': b'value2',
                     b'cf1:col3': b'value3'}

    assert row1_expected == table.row(b'row1')
    assert row2_expected == table.row(b'row2')
    assert {} == table.row(b'another-row')

    b = table.batch(timestamp=1234567)
    b.put(b'row1', {b'cf1:col5': b'value5'})
    b.send()

    row1_excpected_2 = {b'cf1:col1': b'value1',
                        b'cf1:col2': b'value2',
                        b'cf1:col5': b'value5'}
    assert row1_excpected_2 == table.row(b'row1')


def test_batch_context_managers():
    with table.batch() as b:
        b.put(b'row4', {b'cf1:col3': b'value3'})
        b.put(b'row5', {b'cf1:col4': b'value4'})
        b.put(b'row', {b'cf1:col1': b'value1'})
        b.delete(b'row', [b'cf1:col4'])
        b.put(b'row', {b'cf1:col2': b'value2'})

    with table.batch(timestamp=87654321) as b:
        b.put(b'row', {b'cf1:c3': b'somevalue',
                       b'cf1:c5': b'anothervalue'})
        b.delete(b'row', [b'cf1:c3'])

    with pytest.raises(ValueError):
        with table.batch(transaction=True) as b:
            b.put(b'fooz', {b'cf1:bar': b'baz'})
            raise ValueError
    assert {} == table.row(b'fooz', [b'cf1:bar'])

    with pytest.raises(ValueError):
        with table.batch(transaction=False) as b:
            b.put(b'fooz', {b'cf1:bar': b'baz'})
            raise ValueError
    assert {b'cf1:bar': b'baz'} == table.row(b'fooz', [b'cf1:bar'])

    with table.batch(batch_size=5) as b:
        for i in range(10):
            b.put(('row-batch1-%03d' % i).encode('ascii'),
                  {b'cf1:': str(i).encode('ascii')})

    with table.batch(batch_size=20) as b:
        for i in range(95):
            b.put(('row-batch2-%03d' % i).encode('ascii'),
                  {b'cf1:': str(i).encode('ascii')})
    assert 95 == len(list(table.scan(row_prefix=b'row-batch2-')))

    with table.batch(batch_size=20) as b:
        for i in range(95):
            b.delete(('row-batch2-%03d' % i).encode('ascii'))
    assert 0 == len(list(table.scan(row_prefix=b'row-batch2-')))


@pytest.mark.parametrize("kwargs", [
    dict(columns=object()),
    dict(timestamp='invalid')
])
def test_rows_exceptions(kwargs):
    row_keys = [b'rows-row1', b'rows-row2', b'rows-row3']
    with pytest.raises(TypeError):
        table.rows(row_keys, **kwargs)


def test_rows():
    row_keys = [b'rows-row1', b'rows-row2', b'rows-row3']
    data_old = {b'cf1:col1': b'v1old', b'cf1:col2': b'v2old'}
    data_new = {b'cf1:col1': b'v1new', b'cf1:col2': b'v2new'}

    assert {} == table.rows([])

    for row_key in row_keys:
        table.put(row_key, data_old, timestamp=4000)

    rows = dict(table.rows(row_keys))
    for row_key in row_keys:
        assert row_key in rows
        assert data_old == rows[row_key]

    for row_key in row_keys:
        table.put(row_key, data_new)

    rows = dict(table.rows(row_keys))
    for row_key in row_keys:
        assert row_key in rows
        assert data_new == rows[row_key]

    rows = dict(table.rows(row_keys, timestamp=5000))
    for row_key in row_keys:
        assert row_key in rows
        assert data_old == rows[row_key]


@pytest.mark.parametrize('kwargs,exception', [
    (dict(versions='invalid'), TypeError),
    (dict(versions=3, timestamp='invalid'), TypeError),
    (dict(versions=0), ValueError)
])
def test_cells_exceptions(kwargs, exception):
    row_key = b'cell-test'
    col = b'cf1:col1'

    with pytest.raises(exception):
        table.cells(row_key, col, **kwargs)


def test_cells():
    row_key = b'cell-test'
    col = b'cf1:col1'

    table.put(row_key, {col: b'old'}, timestamp=1234)
    table.put(row_key, {col: b'new'})

    results = table.cells(row_key, col, versions=1)
    assert len(results) == 1
    assert b'new' == results[0]

    results = table.cells(row_key, col)
    assert len(results) == 2
    assert b'new' == results[0]
    assert b'old' == results[1]

    results = table.cells(row_key, col, timestamp=2345, include_timestamp=True)
    assert len(results) == 1
    assert b'old' == results[0][0]
    assert 1234 == results[0][1]


def test_scan():
    with pytest.raises(TypeError):
        list(table.scan(row_prefix='foobar', row_start='xyz'))

    if connection.compat == '0.90':
        with pytest.raises(NotImplementedError):
            list(table.scan(filter='foo'))

    with pytest.raises(ValueError):
        list(table.scan(limit=0))

    with table.batch() as b:
        for i in range(2000):
            b.put(('row-scan-a%05d' % i).encode('ascii'),
                  {b'cf1:col1': b'v1',
                   b'cf1:col2': b'v2',
                   b'cf2:col1': b'v1',
                   b'cf2:col2': b'v2'})
            b.put(('row-scan-b%05d' % i).encode('ascii'),
                  {b'cf1:col1': b'v1',
                   b'cf1:col2': b'v2'})

    def calc_len(scanner):
        d = collections.deque(maxlen=1)
        d.extend(enumerate(scanner, 1))
        if d:
            return d[0][0]
        return 0

    scanner = table.scan(row_start=b'row-scan-a00012',
                         row_stop=b'row-scan-a00022')
    assert 10 == calc_len(scanner)

    scanner = table.scan(row_start=b'xyz')
    assert 0 == calc_len(scanner)

    scanner = table.scan(row_start=b'xyz', row_stop=b'zyx')
    assert 0 == calc_len(scanner)

    scanner = table.scan(row_start=b'row-scan-', row_stop=b'row-scan-a999',
                         columns=[b'cf1:col1', b'cf2:col2'])
    row_key, row = next(scanner)
    assert row_key == b'row-scan-a00000'
    assert row == {b'cf1:col1': b'v1',
                   b'cf2:col2': b'v2'}
    assert 2000 - 1 == calc_len(scanner)

    scanner = table.scan(row_prefix=b'row-scan-a', batch_size=499, limit=1000)
    assert 1000 == calc_len(scanner)

    scanner = table.scan(row_prefix=b'row-scan-b', batch_size=1, limit=10)
    assert 10 == calc_len(scanner)

    scanner = table.scan(row_prefix=b'row-scan-b', batch_size=5, limit=10)
    assert 10 == calc_len(scanner)

    scanner = table.scan(timestamp=123)
    assert 0 == calc_len(scanner)

    scanner = table.scan(row_prefix=b'row', timestamp=123)
    assert 0 == calc_len(scanner)

    scanner = table.scan(batch_size=20)
    next(scanner)
    next(scanner)
    scanner.close()
    with pytest.raises(StopIteration):
        next(scanner)


def test_scan_sorting():
    if connection.compat < '0.96':
        return  # not supported

    input_row = {}
    for i in range(100):
        input_row[('cf1:col-%03d' % i).encode('ascii')] = b''
    input_key = b'row-scan-sorted'
    table.put(input_key, input_row)

    scan = table.scan(row_start=input_key, sorted_columns=True)
    key, row = next(scan)
    assert key == input_key
    assert sorted(input_row.items()) == list(row.items())


def test_scan_reverse():
    if connection.compat < '0.98':
        with pytest.raises(NotImplementedError):
            list(table.scan(reverse=True))
        return

    with table.batch() as b:
        for i in range(2000):
            b.put(('row-scan-reverse-%04d' % i).encode('ascii'),
                  {b'cf1:col1': b'v1',
                   b'cf1:col2': b'v2'})

    scan = table.scan(row_prefix=b'row-scan-reverse', reverse=True)
    assert 2000 == len(list(scan))

    scan = table.scan(limit=10, reverse=True)
    assert 10 == len(list(scan))

    scan = table.scan(row_start=b'row-scan-reverse-1999',
                      row_stop=b'row-scan-reverse-0000', reverse=True)
    key, data = next(scan)
    assert b'row-scan-reverse-1999' == key

    key, data = list(scan)[-1]
    assert b'row-scan-reverse-0001' == key


def test_scan_filter_and_batch_size():
    # See issue #54 and #56
    filter = b"SingleColumnValueFilter ('cf1', 'qual1', =, 'binary:val1')"
    for k, v in table.scan(filter=filter):
        pass
        # print(v)


def test_delete():
    row_key = b'row-test-delete'
    data = {b'cf1:col1': b'v1',
            b'cf1:col2': b'v2',
            b'cf1:col3': b'v3'}
    table.put(row_key, {b'cf1:col2': b'v2old'}, timestamp=1234)
    table.put(row_key, data)

    table.delete(row_key, [b'cf1:col2'], timestamp=2345)
    assert 1 == len(table.cells(row_key, b'cf1:col2', versions=2))
    assert data == table.row(row_key)

    table.delete(row_key, [b'cf1:col1'])
    res = table.row(row_key)
    assert b'cf1:col1' not in res
    assert b'cf1:col2' in res
    assert b'cf1:col3' in res

    table.delete(row_key, timestamp=12345)
    res = table.row(row_key)
    assert b'cf1:col2' in res
    assert b'cf1:col3' in res

    table.delete(row_key)
    assert {} == table.row(row_key)


@pytest.mark.parametrize("kwargs,exception", [
    (dict(size='abc'), TypeError),
    (dict(size=0), ValueError)
])
def test_connection_pool_construction_exceptions(kwargs, exception):
    with pytest.raises(exception):
        ConnectionPool(**kwargs)


def test_connection_pool():
    from thrift.transport.TTransport import TTransportException

    def run():
        name = threading.current_thread().name
        print("Thread %s starting" % name)

        def inner_function():
            # Nested connection requests must return the same connection
            with pool.connection() as another_connection:
                assert connection is another_connection

                # Fake an exception once in a while
                if random.random() < .25:
                    connection.transport.close()
                    raise TTransportException("Fake transport exception")

        for i in range(50):
            with pool.connection() as connection:
                connection.tables()

                try:
                    inner_function()
                except TTransportException:
                    # This error should have been picked up by the
                    # connection pool, and the connection should have
                    # been replaced by a fresh one
                    pass

                connection.tables()

        print("Thread %s done" % name)

    N_THREADS = 10

    pool = ConnectionPool(size=3, **connection_kwargs)
    threads = [threading.Thread(target=run) for i in range(N_THREADS)]

    for t in threads:
        t.start()

    while threads:
        for t in threads:
            t.join(timeout=.1)

        # filter out finished threads
        threads = [t for t in threads if t.is_alive()]
        print("%d threads still alive" % len(threads))


def test_pool_exhaustion():
    pool = ConnectionPool(size=1, **connection_kwargs)

    def run():
        with pytest.raises(NoConnectionsAvailable):
            with pool.connection(timeout=1) as connection:
                connection.tables()

    with pool.connection():
        # At this point the only connection is assigned to this thread,
        # so another thread cannot obtain a connection at this point.

        t = threading.Thread(target=run)
        t.start()
        t.join()
