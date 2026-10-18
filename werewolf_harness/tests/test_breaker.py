from ..app.llm.breaker import Breaker


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def test_opens_on_failure_rate_then_probes_then_closes():
    c = Clock()
    b = Breaker("p", window_s=10, min_calls=4, threshold=0.5, cooldown_s=5, probes=2, clock=c)
    for ok in (True, False, True, False):
        assert b.allow()
        b.record(ok)
    assert b.state == "open" and not b.allow()          # refused at once while open
    c.t = 6
    assert b.allow() and b.allow() and not b.allow()    # half-open: exactly `probes` calls through
    b.record(True)
    b.record(True)
    assert b.state == "closed"


def test_a_failed_probe_reopens():
    c = Clock()
    b = Breaker("p", window_s=10, min_calls=2, threshold=0.5, cooldown_s=5, probes=1, clock=c)
    b.record(False)
    b.record(False)
    assert b.state == "open"
    c.t = 6
    assert b.allow()
    b.record(False)
    assert b.state == "open" and not b.allow()


def test_old_failures_leave_the_window():
    c = Clock()
    b = Breaker("p", window_s=10, min_calls=4, threshold=0.5, clock=c)
    b.record(False)
    b.record(False)
    c.t = 20
    for _ in range(4):
        b.record(True)
    assert b.state == "closed"
