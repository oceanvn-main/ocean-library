"""
cass_bridge.py — Thin Python bridge for MATLAB → Apache Cassandra.

MATLAB calls these module-level functions via py.cass_bridge.*
All functions accept/return only Python primitives (str, int, float, list, dict)
so MATLAB can pass them across the py.* boundary without type issues.
"""

import sys, os

# --------------------------------------------------------------------------
# Bootstrap: ensure the user site-packages directory where pip installed
# cassandra-driver is on the path.  This is needed when MATLAB's embedded
# Python does not inherit the Windows user PATH / PYTHONPATH correctly.
# --------------------------------------------------------------------------
def _bootstrap_site_packages():
    import glob, pathlib
    # Microsoft Store Python puts user packages here:
    appdata = os.environ.get('LOCALAPPDATA', '')
    pattern = os.path.join(
        appdata, 'Packages', 'PythonSoftwareFo*',
        'LocalCache', 'local-packages', 'Python*', 'site-packages')
    for p in glob.glob(pattern):
        if p not in sys.path:
            sys.path.insert(0, p)
    # Standard pip --user location on Windows:
    user_base = pathlib.Path(os.path.expanduser('~')) / 'AppData' / 'Roaming' / 'Python'
    for p in user_base.glob('Python*/site-packages'):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)

_bootstrap_site_packages()

# These imports must be AFTER bootstrap so the correct site-packages is on path
import json
from cassandra.cluster import Cluster
from cassandra.policies import DCAwareRoundRobinPolicy

# --------------------------------------------------------------------------

_session = None   # module-level session (one per MATLAB process)
_cluster  = None


def connect(host: str, port: int, keyspace: str) -> str:
    """Open a connection. Returns 'OK' or raises."""
    global _session, _cluster
    _cluster = Cluster(
        contact_points=[host],
        port=int(port),
        load_balancing_policy=DCAwareRoundRobinPolicy(local_dc='datacenter1'),
        protocol_version=4,
    )
    _session = _cluster.connect()
    # Ensure keyspace exists
    _session.execute(
        f"CREATE KEYSPACE IF NOT EXISTS {keyspace} "
        f"WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}"
    )
    _session.set_keyspace(keyspace)
    return 'OK'


def disconnect() -> str:
    """Close the connection."""
    global _session, _cluster
    if _session:
        _session.shutdown()
        _session = None
    if _cluster:
        _cluster.shutdown()
        _cluster = None
    return 'OK'


def execute(cql: str) -> str:
    """Run a CQL DDL or DML statement."""
    if _session is None:
        raise RuntimeError('Not connected. Call connect() first.')
    _session.execute(cql)
    return 'OK'


def fetch(cql: str) -> str:
    """Run a SELECT and return rows as a JSON string (list of dicts)."""
    if _session is None:
        raise RuntimeError('Not connected. Call connect() first.')
    rows = _session.execute(cql)
    result = []
    for row in rows:
        d = {}
        for col in row._fields:
            v = getattr(row, col)
            if hasattr(v, 'hex'):          # UUID
                d[col] = str(v)
            else:
                d[col] = v
        result.append(d)
    return json.dumps(result)


def insert_batch(keyspace: str, table: str, rows_json: str) -> str:
    """Bulk insert rows provided as a JSON string (list of dicts)."""
    if _session is None:
        raise RuntimeError('Not connected. Call connect() first.')
    rows = json.loads(rows_json)
    if not rows:
        return 'OK (0 rows)'
    # Build prepared statement from first row's keys
    cols   = list(rows[0].keys())
    ph     = ', '.join(['?'] * len(cols))
    col_s  = ', '.join(cols)
    full   = f'{keyspace}.{table}'
    stmt   = _session.prepare(
        f'INSERT INTO {full} (row_id, {col_s}) VALUES (uuid(), {ph})'
    )
    for row in rows:
        vals = [row[c] for c in cols]
        _session.execute(stmt, vals)
    return f'OK ({len(rows)} rows)'


def append_batch_ts(keyspace: str, table: str, series_id: str, ts_json: str, val_json: str, meta_json: str) -> str:
    """Insert batched time-series data using concurrent execution.
       Lists must match in length.
    """
    from cassandra.concurrent import execute_concurrent_with_args
    if _session is None:
        raise RuntimeError('Not connected. Call connect() first.')
    ts_list   = json.loads(ts_json)
    val_list  = json.loads(val_json)
    meta_list = json.loads(meta_json)
    if not ts_list:
        return 'OK (0 rows)'
    full = f'{keyspace}.{table}'
    stmt = _session.prepare(
        f'INSERT INTO {full} (series_id, ts, value, meta) VALUES (?, ?, ?, ?)'
    )
    # create sequence of argument tuples
    args = [(series_id, t, v, m) for t, v, m in zip(ts_list, val_list, meta_list)]
    
    # execute_concurrent_with_args handles the async parallel batching
    execute_concurrent_with_args(_session, stmt, args, concurrency=100)
    return f'OK ({len(args)} rows)'


def query_range(keyspace: str, table: str, series_id: str, t0_iso: str, t1_iso: str) -> str:
    """Query time-series data for a specific series within a time range."""
    if _session is None:
        raise RuntimeError('Not connected. Call connect() first.')
    full = f'{keyspace}.{table}'
    cql = f"SELECT * FROM {full} WHERE series_id = %s AND ts >= %s AND ts <= %s"
    rows = _session.execute(cql, (series_id, t0_iso, t1_iso))
    result = []
    for row in rows:
        d = {}
        for col in row._fields:
            v = getattr(row, col)
            if hasattr(v, 'isoformat'):
                d[col] = v.isoformat()
            elif hasattr(v, 'hex'):
                d[col] = str(v)
            else:
                d[col] = v
        result.append(d)
    return json.dumps(result)
