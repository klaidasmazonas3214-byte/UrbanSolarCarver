# Session

GPU context and Warp module lifecycle management. A `CarverSession` keeps the CUDA context alive across multiple pipeline calls, avoiding the ~2 s cold-start penalty on each invocation.

The Grasshopper plugin uses a long-lived session via the daemon; the Python API and CLI create sessions on demand. The `session_cache` decorator memoises expensive tensors (e.g., Tregenza patch directions) in the active session so they are computed once and reused across stages.

::: urbansolarcarver.session
