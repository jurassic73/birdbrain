"""Looking at the SQL a piece of code runs."""


def plans(conn, run) -> dict[str, list[str]]:
    """EXPLAIN QUERY PLAN of every SELECT that run(conn) executes, by statement."""
    seen: list[str] = []
    conn.set_trace_callback(seen.append)
    try:
        run(conn)
    finally:
        conn.set_trace_callback(None)
    return {sql: [r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + sql)]
            for sql in seen if sql.lstrip().upper().startswith(("SELECT", "WITH"))}
