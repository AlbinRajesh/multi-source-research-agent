from src.graph import check_retry
from src.state import ResearchState, VerificationVerdict

def make_state(confidences, retry_count=0):
    verdicts = [VerificationVerdict(claim_id=str(i), is_grounded=False, confidence=c) for i, c in enumerate(confidences)]
    return ResearchState(research_topic="t", verified_claims=verdicts, retry_count=retry_count, max_retries=2)

def test_retries_bounded():
    s = make_state(["unconfirmed"]*5)
    out1 = check_retry(s)
    assert out1["route_decision"] == "search"
    assert out1["retry_count"] == 1

    s.retry_count = 2  # simulate max reached
    out2 = check_retry(s)
    assert out2["route_decision"] == "synthesize"

def test_conflicting_triggers_retry():
    s = make_state(["conflicting"]*5)
    assert check_retry(s)["route_decision"] == "search"