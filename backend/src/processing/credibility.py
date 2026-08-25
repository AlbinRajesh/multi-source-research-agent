"""Source credibility pre-filter — cheap heuristic scoring (no LLM call).

Runs BEFORE claim extraction / verification to cut load on the expensive
LLM stages. This is a coarse filter, not a substitute for claim-level
groundedness verification (see agents/verifier.py for that).
"""
import re
from typing import List, Dict, Any
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


class CredibilityScorer:
    TRUSTED_DOMAINS = {
        # Academic / government
        '.edu', '.ac.uk', '.ac.in', '.edu.in', '.edu.au', '.ac.jp',
        '.gov', '.gov.uk', '.gov.au', '.gov.ca', '.gov.in', '.europa.eu',

        # Wire services / major news
        'bbc.com', 'reuters.com', 'ap.org', 'npr.org', 'theguardian.com',
        'nytimes.com', 'washingtonpost.com', 'wsj.com', 'ft.com',
        'economist.com', 'bloomberg.com',

        # Reference / encyclopedic
        'wikipedia.org', 'britannica.com',

        # Academic / scientific publishing
        'arxiv.org', 'scholar.google.com', 'pubmed.ncbi.nlm.nih.gov',
        'nih.gov', 'nature.com', 'sciencedirect.com', 'ieee.org', 'jstor.org',
        'springer.com', 'acm.org',

        # International orgs
        'who.int', 'un.org', 'worldbank.org',

        # Business / general-interest press
        'forbes.com', 'businessinsider.com', 'finance.yahoo.com',
        'yahoo.com', 'cnbc.com', 'techcrunch.com', 'theverge.com',
        'wired.com', 'time.com', 'usatoday.com', 'apnews.com',
        'axios.com', 'fortune.com',

        # Company/official sources
         # Company/official sources
        '.google', 'googleblog.com', 'newsroom.',

        # Official framework/language docs
        'react.dev', 'vuejs.org', 'angular.dev', 'developer.mozilla.org',
        'docs.python.org', 'kubernetes.io', 'docker.com', 'nodejs.org',
    }
    SUSPICIOUS_PATTERNS = [
        r'\.(xyz|tk|ml|ga|cf|gq)$',
        r'bit\.ly|tinyurl|t\.co',
        r'blogspot|wordpress\.com',
    ]

    def score_url(self, url: str) -> Dict[str, Any]:
        if not url:
            return {'score': 0, 'factors': ['No URL'], 'level': 'low'}

        score = 40 # Base score
        factors = []

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Graduated TLD signal
            if domain.endswith(('.gov', '.edu', '.int')):
                score += 15
                factors.append('Institutional TLD')

            # Known trusted list signal (extra boost on top of TLD/base)
            for trusted in self.TRUSTED_DOMAINS:
                if trusted in domain:
                    score += 25
                    factors.append(f'Trusted domain: {trusted}')
                    break

            for pattern in self.SUSPICIOUS_PATTERNS:
                if re.search(pattern, domain):
                    score -= 25
                    factors.append(f'Suspicious pattern: {pattern}')
                    break

            if parsed.scheme == 'https':
                score += 5
            else:
                score -= 15

            score = max(0, min(100, score))
            level = 'high' if score >= 70 else 'medium' if score >= 40 else 'low'

            return {'score': score, 'factors': factors or ['Standard domain'], 'level': level, 'domain': domain}
        except Exception as e:
            logger.warning(f"Error scoring URL {url}: {e}")
            return {'score': 20, 'factors': ['Scoring error'], 'level': 'low'}

    def filter_results(self, results: List, min_score: int = 40) -> List:
        kept = []
        scores = []
        for r in results:
            cred = self.score_url(r.url)
            scores.append(cred['score'])
            if cred['score'] >= min_score:
                kept.append(r)
            else:
                logger.info(f"Filtered out low credibility source ({cred['score']}): {r.url}")

        logger.info(f"Credibility pre-filter: {len(results)} -> {len(kept)} (min_score={min_score})")
        if scores:
            logger.info(
                f"[credibility_dist] min={min(scores)} max={max(scores)} "
                f"avg={sum(scores) / len(scores):.1f} n={len(scores)}"
            )
        return kept