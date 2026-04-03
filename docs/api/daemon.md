# Daemon

A persistent background process that keeps a `CarverSession` alive for low-latency calls from Grasshopper or other clients. Communicates via Python's `multiprocessing.connection` over localhost with a random authkey.

The daemon is started and stopped via the CLI (`urbansolarcarver daemon start/stop`) or from within Grasshopper using the `USC_Session` component. It binds to `127.0.0.1` only and is not network-accessible.

::: urbansolarcarver.daemon
