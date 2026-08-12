from src.search_providers.content_extractor import ContentExtractor

def test_breakers_are_per_domain():
    ce = ContentExtractor()
    b1 = ce._get_breaker("https://siteA.com/x")
    b2 = ce._get_breaker("https://siteB.com/y")

    for _ in range(5):
        b1.record_failure()

    assert b1.state.value == "open"
    assert b2.can_execute() is True
    assert b2.state.value == "closed"