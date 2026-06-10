"""
IDP aggregator: transparency-log ingestion for aggregator-mode deployments.

Monitors the tlog-tiles transparency logs of all active iscc-hubs of one
network (testnet or mainnet) and indexes their declaration log entries into a
single index (idptest/idp). Split into pure Sans-IO cores (tlog, hublist,
entry, poller.plan_bundles) and a thin async I/O shell (poller).
"""
