"""
IDP aggregator: transparency-log ingestion for aggregator-mode deployments.

Monitors the tlog-tiles transparency logs of all active iscc-hubs of one
network (testnet or mainnet) and indexes their declaration log entries into a
single index (idptest/idp). Split into pure Sans-IO cores (tlog, hublist,
entry, poller.plan_bundles) and a thin async I/O shell (poller).
"""

# Single source of truth for network-keyed knowledge, consumed by both options.py
# (index name, allowed networks) and entry.py (realm check). ``realm`` is the ISCC
# realm id encoded in ISCC-ID header subtypes; ``index`` is the aggregator index name.
NETWORKS = {
    "testnet": {"realm": 0, "index": "idptest"},
    "mainnet": {"realm": 1, "index": "idp"},
}
