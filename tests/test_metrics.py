from gateway.metrics import Metrics, percentile


def test_percentile_math():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 99) == 5
    assert percentile([], 99) == 0.0


def test_snapshot_reports_slo_attainment():
    m = Metrics()
    for _ in range(10):
        m.observe("ttft_ms", 100.0)
        m.observe("e2e_ms", 500.0)
    snap = m.snapshot(slo_ttft_p99_ms=4000.0, slo_e2e_p99_ms=30000.0)
    assert snap["slos"]["ttft_p99_ms"]["met"] is True
    m.observe("ttft_ms", 99999.0)
    snap = m.snapshot(slo_ttft_p99_ms=4000.0, slo_e2e_p99_ms=30000.0)
    assert snap["slos"]["ttft_p99_ms"]["met"] is False


def test_prometheus_exposition_format():
    m = Metrics()
    m.observe("ttft_ms", 120.0)
    m.inc("requests_completed")
    m.gauges["queue_depth"] = 3
    out = m.prometheus()
    assert "gw_ttft_ms_p99" in out
    assert "gw_requests_completed 1" in out
    assert "gw_queue_depth 3" in out
