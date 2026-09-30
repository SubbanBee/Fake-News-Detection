# ============================================================
# WEB VERIFICATION MODULE
# Fake News Detection System
#
# Purpose:
#   1. Search current news sources
#   2. Fetch the ACTUAL publisher article
#   3. Extract article text
#   4. Find claim-relevant evidence sentences
#   5. Check whether evidence supports or contradicts claim
#   6. Do NOT use simple keyword overlap as proof
#
# Final statuses:
#   REAL / VERIFIED
#   FAKE / CONTRADICTED
#   UNVERIFIED
# ============================================================

import re
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

from html import unescape
from html.parser import HTMLParser

from nltk.corpus import stopwords


# ============================================================
# TRUSTED SOURCES
# ============================================================

TRUSTED_DOMAINS = {
    "reuters.com": "Reuters",
    "apnews.com": "Associated Press",
    "bbc.com": "BBC",
    "bbc.co.uk": "BBC",
    "thehindu.com": "The Hindu",
    "indianexpress.com": "Indian Express",
    "timesofindia.indiatimes.com": "Times of India",
    "ndtv.com": "NDTV",
    "hindustantimes.com": "Hindustan Times",
    "npr.org": "NPR",
    "pib.gov.in": "Press Information Bureau",
    "gov.in": "Government of India",
    "who.int": "World Health Organization",
    "un.org": "United Nations",
    "google.com": "Google",
    "blog.google": "Google",
    "alphabet.com": "Alphabet",
    "tesla.com": "Tesla",
    "sec.gov": "U.S. SEC",
    "cia.gov": "CIA",
    "go.kr": "Government of South Korea",
    "opm.go.kr": "Office for Government Policy Coordination",
}


TRUSTED_PUBLISHERS = {
    "reuters",
    "associated press",
    "ap news",
    "bbc",
    "bbc news",
    "the hindu",
    "hindu",
    "indian express",
    "indianexpress",
    "times of india",
    "toi",
    "ndtv",
    "hindustan times",
    "npr",
    "press information bureau",
    "pib",
    "google",
    "alphabet",
    "tesla",
}


# ============================================================
# HTTP SETTINGS
# ============================================================

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/154.0 Safari/537.36"
)

ARTICLE_TIMEOUT = 10


# ============================================================
# ROLE / CLAIM PATTERNS
# ============================================================

ROLE_PATTERNS = [

    r"\b(.+?)\s+is\s+(?:the\s+)?"
    r"(president|prime minister|pm|ceo|chief executive officer|"
    r"governor|mayor|chancellor)\s+of\s+(.+)",

    r"\b(.+?)\s+is\s+(?:the\s+)?"
    r"(president|prime minister|pm|ceo|chief executive officer|"
    r"governor|mayor|chancellor)\s+(.+)",
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def _normalize_text(text):

    if not text:
        return ""

    text = unescape(text)

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# STOPWORDS
# ============================================================

def _get_stopwords():

    try:

        return set(
            stopwords.words("english")
        )

    except Exception:

        return {
            "the",
            "is",
            "a",
            "an",
            "of",
            "to",
            "in",
            "on",
            "for",
            "and",
            "or",
            "was",
            "were",
            "are",
            "be",
            "as",
            "at",
            "by",
            "with",
            "has",
            "have",
            "had",
            "this",
            "that",
            "from",
        }


# ============================================================
# CLEAN SEARCH QUERY
# ============================================================

def _clean_query(
    text,
    max_words=12
):

    stop_words = _get_stopwords()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        "",
        text
    )

    words = re.findall(
        r"[A-Za-z]{2,}",
        text.lower()
    )

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    unique_words = []

    for word in words:

        if word not in unique_words:

            unique_words.append(
                word
            )

    return " ".join(
        unique_words[:max_words]
    )


# ============================================================
# EXTRACT SIMPLE ROLE CLAIM
#
# Example:
#
# Narendra Modi is the PM of Korea
#
# subject = Narendra Modi
# role    = PM
# target  = Korea
# ============================================================

def _extract_role_claim(text):

    clean = re.sub(
        r"\s+",
        " ",
        text.strip()
    )

    for pattern in ROLE_PATTERNS:

        match = re.search(
            pattern,
            clean,
            flags=re.IGNORECASE
        )

        if match:

            subject = (
                match.group(1)
                .strip()
                .strip(".")
            )

            role = (
                match.group(2)
                .strip()
            )

            target = (
                match.group(3)
                .strip()
                .strip(".")
            )

            return {
                "type": "role_claim",
                "subject": subject,
                "role": role,
                "target": target
            }

    return None


# ============================================================
# ROLE NORMALIZATION
# ============================================================

def _normalize_role(role):

    role = (
        role
        .lower()
        .strip()
    )

    mapping = {

        "pm":
            "prime minister",

        "prime minister":
            "prime minister",

        "president":
            "president",

        "ceo":
            "ceo",

        "chief executive officer":
            "ceo",

        "governor":
            "governor",

        "mayor":
            "mayor",

        "chancellor":
            "chancellor",
    }

    return mapping.get(
        role,
        role
    )


# ============================================================
# SOURCE NAME FROM URL
# ============================================================

def _source_name(url):

    try:

        parsed = urllib.parse.urlparse(
            url
        )

        hostname = (
            parsed.netloc
            .lower()
        )

        hostname = hostname.replace(
            "www.",
            ""
        )

        for domain, name in TRUSTED_DOMAINS.items():

            if (
                hostname == domain
                or hostname.endswith(
                    "." + domain
                )
            ):

                return name, True

        return hostname, False

    except Exception:

        return (
            "Unknown Source",
            False
        )


# ============================================================
# EXTRACT PUBLISHER FROM GOOGLE NEWS TITLE
# ============================================================

def _extract_publisher(title):

    if not title:
        return None

    parts = title.rsplit(
        " - ",
        1
    )

    if len(parts) == 2:

        publisher = (
            parts[1]
            .strip()
        )

        if publisher:

            return publisher

    return None


# ============================================================
# TRUSTED PUBLISHER CHECK
# ============================================================

def _is_trusted_publisher(
    publisher
):

    if not publisher:

        return False

    publisher_lower = (
        publisher
        .strip()
        .lower()
    )

    publisher_lower = re.sub(
        r"[^a-z0-9\s]",
        " ",
        publisher_lower
    )

    publisher_lower = re.sub(
        r"\s+",
        " ",
        publisher_lower
    ).strip()

    for trusted_name in TRUSTED_PUBLISHERS:

        if trusted_name in publisher_lower:

            return True

    return False


# ============================================================
# CLEAN PUBLISHER NAME
# ============================================================

def _clean_publisher_name(
    publisher
):

    if not publisher:

        return "Unknown Source"

    publisher = (
        publisher
        .strip()
    )

    publisher = re.sub(
        r"\s+",
        " ",
        publisher
    )

    return publisher


# ============================================================
# HTML TEXT EXTRACTOR
# ============================================================

class _ArticleTextParser(
    HTMLParser
):

    def __init__(self):

        super().__init__()

        self.parts = []

        self.skip_depth = 0

        self.skip_tags = {
            "script",
            "style",
            "noscript",
            "svg",
            "iframe",
            "nav",
            "footer",
            "header",
            "form",
        }

    def handle_starttag(
        self,
        tag,
        attrs
    ):

        tag = tag.lower()

        if tag in self.skip_tags:

            self.skip_depth += 1

    def handle_endtag(
        self,
        tag
    ):

        tag = tag.lower()

        if (
            tag in self.skip_tags
            and self.skip_depth > 0
        ):

            self.skip_depth -= 1

    def handle_data(
        self,
        data
    ):

        if self.skip_depth > 0:

            return

        text = data.strip()

        if text:

            self.parts.append(
                text
            )

    def get_text(self):

        return " ".join(
            self.parts
        )


# ============================================================
# FETCH ACTUAL ARTICLE PAGE
#
# IMPORTANT:
# Google News RSS is used only to DISCOVER articles.
# The actual publisher page is fetched separately.
# ============================================================

def _fetch_article(
    url,
    max_chars=120000
):

    if not url:

        return {
            "success": False,
            "url": url,
            "final_url": url,
            "title": "",
            "description": "",
            "text": "",
            "error": "Empty URL"
        }

    try:

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept":
                    "text/html,"
                    "application/xhtml+xml"
            }
        )

        with urllib.request.urlopen(
            request,
            timeout=ARTICLE_TIMEOUT
        ) as response:

            final_url = response.geturl()

            content_type = (
                response.headers
                .get(
                    "Content-Type",
                    ""
                )
                .lower()
            )

            raw = response.read(
                2_000_000
            )

        if (
            "html" not in content_type
            and not raw.lstrip().startswith(
                b"<"
            )
        ):

            return {
                "success": False,
                "url": url,
                "final_url": final_url,
                "title": "",
                "description": "",
                "text": "",
                "error":
                    "Publisher did not return HTML."
            }

        encoding = "utf-8"

        try:

            html = raw.decode(
                encoding,
                errors="ignore"
            )

        except Exception:

            html = str(raw)

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        title_match = re.search(
            r"<title[^>]*>(.*?)</title>",
            html,
            flags=re.IGNORECASE | re.DOTALL
        )

        title = ""

        if title_match:

            title = re.sub(
                r"\s+",
                " ",
                unescape(
                    title_match.group(1)
                )
            ).strip()

        # ----------------------------------------------------
        # META DESCRIPTION
        # ----------------------------------------------------

        description = ""

        meta_patterns = [

            r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',

            r'<meta[^>]+content=["\'](.*?)["\'][^>]+name=["\']description["\']',

            r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\'](.*?)["\']',

            r'<meta[^>]+content=["\'](.*?)["\'][^>]+property=["\']og:description["\']',
        ]

        for pattern in meta_patterns:

            match = re.search(
                pattern,
                html,
                flags=re.IGNORECASE | re.DOTALL
            )

            if match:

                description = re.sub(
                    r"\s+",
                    " ",
                    unescape(
                        match.group(1)
                    )
                ).strip()

                if description:

                    break

        # ----------------------------------------------------
        # ARTICLE BODY
        # ----------------------------------------------------

        parser = _ArticleTextParser()

        parser.feed(
            html
        )

        article_text = parser.get_text()

        article_text = re.sub(
            r"\s+",
            " ",
            article_text
        ).strip()

        if len(article_text) > max_chars:

            article_text = (
                article_text[:max_chars]
            )

        return {

            "success": bool(
                article_text
            ),

            "url": url,

            "final_url": final_url,

            "title": title,

            "description": description,

            "text": article_text,

            "error": ""
        }

    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        ValueError,
        Exception
    ) as e:

        return {

            "success": False,

            "url": url,

            "final_url": url,

            "title": "",

            "description": "",

            "text": "",

            "error": str(e)
        }


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def _split_sentences(
    text
):

    if not text:

        return []

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    cleaned = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence) < 20:

            continue

        cleaned.append(
            sentence
        )

    return cleaned


# ============================================================
# SOURCE TEXT
# ============================================================

def _source_text(
    result
):

    article_text = result.get(
        "article_text",
        ""
    )

    if article_text:

        return _normalize_text(
            article_text
        )

    return _normalize_text(
        (
            result.get(
                "title",
                ""
            )
            + " "
            + result.get(
                "description",
                ""
            )
        )
    )


# ============================================================
# ENTITY MATCH
# ============================================================

def _entity_in_source(
    entity,
    source_text
):

    entity_normalized = (
        _normalize_text(
            entity
        )
    )

    if not entity_normalized:

        return False

    # Exact phrase first.
    if entity_normalized in source_text:

        return True

    entity_words = [
        word
        for word in entity_normalized.split()
        if len(word) >= 2
    ]

    if not entity_words:

        return False

    return all(
        word in source_text
        for word in entity_words
    )


# ============================================================
# ROLE MATCH
# ============================================================

def _role_in_source(
    role,
    source_text
):

    role = _normalize_role(
        role
    )

    padded = (
        " "
        + source_text
        + " "
    )

    if role == "prime minister":

        return (
            " prime minister "
            in padded
            or " pm "
            in padded
        )

    if role == "ceo":

        return (
            " ceo "
            in padded
            or
            " chief executive officer "
            in padded
        )

    return (
        " "
        + role
        + " "
    ) in padded


# ============================================================
# TARGET MATCH
# ============================================================

def _target_in_source(
    target,
    source_text
):

    target_normalized = (
        _normalize_text(
            target
        )
    )

    if not target_normalized:

        return False

    if target_normalized in source_text:

        return True

    target_words = [
        word
        for word in target_normalized.split()
        if len(word) >= 3
    ]

    if not target_words:

        return False

    return all(
        word in source_text
        for word in target_words
    )


# ============================================================
# CLAIM-RELEVANT SENTENCES
# ============================================================

def _find_relevant_sentences(
    claim,
    article_text,
    max_sentences=5
):

    if not article_text:

        return []

    sentences = _split_sentences(
        article_text
    )

    relevant = []

    subject = _normalize_text(
        claim["subject"]
    )

    role = _normalize_role(
        claim["role"]
    )

    target = _normalize_text(
        claim["target"]
    )

    for sentence in sentences:

        normalized = _normalize_text(
            sentence
        )

        subject_match = (
            subject in normalized
        )

        role_match = _role_in_source(
            role,
            normalized
        )

        target_match = (
            target in normalized
            or _target_in_source(
                target,
                normalized
            )
        )

        score = 0

        if subject_match:
            score += 3

        if role_match:
            score += 2

        if target_match:
            score += 2

        if (
            subject_match
            and role_match
        ):

            score += 2

        if (
            role_match
            and target_match
        ):

            score += 2

        if score >= 5:

            relevant.append(
                (
                    score,
                    sentence
                )
            )

    relevant.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        sentence
        for score, sentence
        in relevant[:max_sentences]
    ]


# ============================================================
# EXPLICIT SUPPORT CHECK
#
# This checks the ACTUAL ARTICLE TEXT.
# ============================================================

def _strong_role_support(
    claim,
    result
):

    article_text = result.get(
        "article_text",
        ""
    )

    if not article_text:

        return False

    normalized = _normalize_text(
        article_text
    )

    subject = _normalize_text(
        claim["subject"]
    )

    target = _normalize_text(
        claim["target"]
    )

    role = _normalize_role(
        claim["role"]
    )

    if not subject:
        return False

    if not target:
        return False

    if not _entity_in_source(
        subject,
        normalized
    ):

        return False

    if not _role_in_source(
        role,
        normalized
    ):

        return False

    if not _target_in_source(
        target,
        normalized
    ):

        return False

    relevant_sentences = (
        _find_relevant_sentences(
            claim,
            article_text
        )
    )

    if not relevant_sentences:

        return False

    subject_pattern = re.escape(
        subject
    )

    target_pattern = re.escape(
        target
    )

    if role == "prime minister":

        role_pattern = (
            r"(?:prime minister|pm)"
        )

    elif role == "ceo":

        role_pattern = (
            r"(?:ceo|chief executive officer)"
        )

    else:

        role_pattern = re.escape(
            role
        )

    patterns = [

        rf"{subject_pattern}.{{0,250}}"
        rf"{role_pattern}.{{0,250}}"
        rf"{target_pattern}",

        rf"{subject_pattern}.{{0,250}}"
        rf"{target_pattern}.{{0,250}}"
        rf"{role_pattern}",

        rf"{role_pattern}.{{0,250}}"
        rf"{subject_pattern}.{{0,250}}"
        rf"{target_pattern}",

        rf"{target_pattern}.{{0,250}}"
        rf"{role_pattern}.{{0,250}}"
        rf"{subject_pattern}",
    ]

    for sentence in relevant_sentences:

        normalized_sentence = (
            _normalize_text(
                sentence
            )
        )

        for pattern in patterns:

            if re.search(
                pattern,
                normalized_sentence,
                flags=re.IGNORECASE
            ):

                return True

    return False


# ============================================================
# DIFFERENT ROLE HOLDER DETECTION
#
# Example:
#
# Claim:
#   X is PM of Y
#
# Evidence:
#   Z is the Prime Minister of Y
#
# This is stronger contradiction evidence than
# simply saying "X was not found".
# ============================================================

def _extract_role_holders(
    claim,
    article_text
):

    if not article_text:

        return []

    sentences = _split_sentences(
        article_text
    )

    role = _normalize_role(
        claim["role"]
    )

    target = _normalize_text(
        claim["target"]
    )

    holders = []

    if role == "prime minister":

        role_regex = (
            r"([A-Z][A-Za-z.\-']"
            r"(?:\s+[A-Z][A-Za-z.\-']+){0,5})"
            r"\s+(?:is|was|serves as|served as|"
            r"became|becomes)\s+"
            r"(?:the\s+)?"
            r"(?:Prime Minister|PM)"
            r"\s+(?:of|for)\s+"
            r"([A-Za-z][A-Za-z .\-']+)"
        )

    elif role == "ceo":

        role_regex = (
            r"([A-Z][A-Za-z.\-']"
            r"(?:\s+[A-Z][A-Za-z.\-']+){0,5})"
            r"\s+(?:is|was|serves as|served as|"
            r"became|becomes)\s+"
            r"(?:the\s+)?"
            r"(?:CEO|Chief Executive Officer)"
            r"\s+(?:of|for)\s+"
            r"([A-Za-z][A-Za-z .\-']+)"
        )

    else:

        role_regex = (
            r"([A-Z][A-Za-z.\-']"
            r"(?:\s+[A-Z][A-Za-z.\-']+){0,5})"
            r"\s+(?:is|was|serves as|served as|"
            r"became|becomes)\s+"
            r"(?:the\s+)?"
            + re.escape(role)
            + r"\s+(?:of|for)\s+"
            r"([A-Za-z][A-Za-z .\-']+)"
        )

    for sentence in sentences:

        match = re.search(
            role_regex,
            sentence,
            flags=re.IGNORECASE
        )

        if not match:

            continue

        person = (
            match.group(1)
            .strip()
        )

        matched_target = (
            match.group(2)
            .strip()
        )

        normalized_target = (
            _normalize_text(
                matched_target
            )
        )

        if (
            target in normalized_target
            or normalized_target in target
        ):

            holders.append(
                {
                    "person": person,
                    "sentence": sentence
                }
            )

    return holders


# ============================================================
# FIND SUPPORTING SOURCES
# ============================================================

def _find_supporting_sources(
    claim,
    results
):

    supporting = []

    for result in results:

        if not result.get(
            "trusted",
            False
        ):

            continue

        if not result.get(
            "article_fetched",
            False
        ):

            continue

        if _strong_role_support(
            claim,
            result
        ):

            supporting.append(
                result
            )

    return supporting


# ============================================================
# FIND CONTRADICTORY SOURCES
# ============================================================

def _find_contradictory_sources(
    claim,
    target_results
):

    contradictions = []

    claimed_subject = _normalize_text(
        claim["subject"]
    )

    for result in target_results:

        if not result.get(
            "trusted",
            False
        ):

            continue

        if not result.get(
            "article_fetched",
            False
        ):

            continue

        article_text = result.get(
            "article_text",
            ""
        )

        if not article_text:

            continue

        holders = _extract_role_holders(
            claim,
            article_text
        )

        different_holders = []

        for holder in holders:

            holder_name = _normalize_text(
                holder["person"]
            )

            if not holder_name:

                continue

            if (
                holder_name
                != claimed_subject
                and claimed_subject
                not in holder_name
            ):

                different_holders.append(
                    holder
                )

        if different_holders:

            result_copy = dict(
                result
            )

            result_copy[
                "identified_role_holders"
            ] = different_holders

            contradictions.append(
                result_copy
            )

    return contradictions


# ============================================================
# FETCH ARTICLE EVIDENCE
# ============================================================

def _attach_article_evidence(
    results
):

    enriched = []

    for result in results:

        item = dict(
            result
        )

        article = _fetch_article(
            result.get(
                "link",
                ""
            )
        )

        item[
            "article_fetched"
        ] = article[
            "success"
        ]

        item[
            "article_url"
        ] = article[
            "final_url"
        ]

        item[
            "article_title"
        ] = article[
            "title"
        ]

        item[
            "article_description"
        ] = article[
            "description"
        ]

        item[
            "article_text"
        ] = article[
            "text"
        ]

        item[
            "article_error"
        ] = article[
            "error"
        ]

        enriched.append(
            item
        )

    return enriched


# ============================================================
# SEARCH GOOGLE NEWS RSS
# ============================================================

def _search_news(
    query,
    max_results=8
):

    encoded_query = (
        urllib.parse.quote_plus(
            query
        )
    )

    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={encoded_query}"
        "&hl=en-IN"
        "&gl=IN"
        "&ceid=IN:en"
    )

    request = urllib.request.Request(
        rss_url,
        headers={
            "User-Agent": USER_AGENT
        }
    )

    with urllib.request.urlopen(
        request,
        timeout=10
    ) as response:

        data = response.read()

    root = ET.fromstring(
        data
    )

    results = []

    for item in root.findall(
        ".//item"
    )[:max_results]:

        title_element = item.find(
            "title"
        )

        title = (
            title_element.text.strip()
            if (
                title_element is not None
                and title_element.text
            )
            else "Untitled"
        )

        link_element = item.find(
            "link"
        )

        link = (
            link_element.text.strip()
            if (
                link_element is not None
                and link_element.text
            )
            else ""
        )

        description_element = (
            item.find(
                "description"
            )
        )

        description = (
            description_element.text.strip()
            if (
                description_element is not None
                and description_element.text
            )
            else ""
        )

        pub_date_element = (
            item.find(
                "pubDate"
            )
        )

        published = (
            pub_date_element.text.strip()
            if (
                pub_date_element is not None
                and pub_date_element.text
            )
            else ""
        )

        publisher = _extract_publisher(
            title
        )

        if publisher:

            source_name = (
                _clean_publisher_name(
                    publisher
                )
            )

            trusted = (
                _is_trusted_publisher(
                    source_name
                )
            )

        else:

            source_name, trusted = (
                _source_name(
                    link
                )
            )

        results.append({

            "title": title,

            "description": description,

            "link": link,

            "source": source_name,

            "trusted": trusted,

            "published": published,

            "article_fetched": False,

            "article_url": "",

            "article_title": "",

            "article_description": "",

            "article_text": "",

            "article_error": "",

        })

    return results


# ============================================================
# FINAL CLAIM ASSESSMENT
# ============================================================

def _assess_claim(
    claim,
    exact_results,
    target_results
):

    if not claim:

        return {

            "status":
                "UNVERIFIED",

            "label":
                "🟡 UNVERIFIED",

            "reason":
                "The submitted text does not contain "
                "a supported role-based claim structure. "
                "The current verifier therefore cannot "
                "make a reliable factual determination.",

            "supporting_sources": [],

            "contradicting_sources": []
        }


    supporting = (
        _find_supporting_sources(
            claim,
            exact_results
        )
    )

    contradictions = (
        _find_contradictory_sources(
            claim,
            target_results
        )
    )


    # ========================================================
    # REAL / VERIFIED
    # ========================================================

    if supporting:

        return {

            "status":
                "REAL",

            "label":
                "🟢 REAL / VERIFIED",

            "reason":
                "A trusted publisher article was "
                "retrieved and its article text contains "
                "claim-relevant evidence connecting the "
                "claimed subject with the stated role "
                "and target.",

            "supporting_sources":
                supporting,

            "contradicting_sources":
                contradictions
        }


    # ========================================================
    # FAKE / CONTRADICTED
    # ========================================================

    if contradictions:

        return {

            "status":
                "FAKE",

            "label":
                "🔴 FAKE / CONTRADICTED",

            "reason":
                "A trusted publisher article was "
                "retrieved and explicitly identifies "
                "a different person as the stated role "
                "holder for the claimed target.",

            "supporting_sources":
                supporting,

            "contradicting_sources":
                contradictions
        }


    # ========================================================
    # UNVERIFIED
    # ========================================================

    return {

        "status":
            "UNVERIFIED",

        "label":
            "🟡 UNVERIFIED",

        "reason":
            "Current sources were found, but the "
            "retrieved trusted article text did not "
            "provide sufficiently explicit evidence "
            "to support or contradict the submitted "
            "claim.",

        "supporting_sources":
            supporting,

        "contradicting_sources":
            contradictions
    }


# ============================================================
# MAIN VERIFICATION FUNCTION
# ============================================================

def verify_news(
    text,
    max_results=8
):

    """
    Performs real-time source verification.

    IMPORTANT:

    ML prediction is NOT used to determine factual truth.

    Google News RSS is used for source discovery.

    Trusted publisher pages are then fetched directly.

    The actual article text is examined before assigning
    REAL or FAKE.

    If evidence is insufficient:
        UNVERIFIED
    """

    if not text or not text.strip():

        return {

            "status":
                "not_found",

            "query":
                "",

            "results":
                [],

            "trusted_count":
                0,

            "claim":
                None,

            "assessment": {

                "status":
                    "UNVERIFIED",

                "label":
                    "🟡 UNVERIFIED",

                "reason":
                    "No text was provided."
            },

            "message":
                "Please enter a news claim."
        }


    # ========================================================
    # EXTRACT CLAIM
    # ========================================================

    claim = _extract_role_claim(
        text
    )


    # ========================================================
    # BUILD SEARCH QUERIES
    # ========================================================

    general_query = _clean_query(
        text
    )

    if claim:

        subject = claim[
            "subject"
        ]

        role = claim[
            "role"
        ]

        target = claim[
            "target"
        ]

        exact_query = (
            f'"{subject}" '
            f'"{role}" '
            f'"{target}"'
        )

        target_query = (
            f'"{target}" '
            f'"{role}"'
        )

        search_queries = [
            exact_query,
            target_query
        ]

    else:

        search_queries = [
            general_query
        ]


    # ========================================================
    # SEARCH CURRENT SOURCES
    # ========================================================

    all_results = []

    exact_results = []

    target_results = []

    try:

        if claim:

            exact_results = _search_news(
                search_queries[0],
                max_results
            )

            target_results = _search_news(
                search_queries[1],
                max_results
            )

            all_results = (
                exact_results
                + target_results
            )

        else:

            exact_results = _search_news(
                general_query,
                max_results
            )

            all_results = (
                exact_results
            )


    except Exception as e:

        return {

            "status":
                "error",

            "query":
                general_query,

            "results":
                [],

            "trusted_count":
                0,

            "claim":
                claim,

            "assessment": {

                "status":
                    "UNVERIFIED",

                "label":
                    "🟡 UNVERIFIED",

                "reason":
                    "Real-time verification could not "
                    "be completed: "
                    + str(e)
            },

            "message":
                "Real-time source verification "
                "is currently unavailable."
        }


    # ========================================================
    # REMOVE DUPLICATE LINKS
    # ========================================================

    unique_results = []

    seen_links = set()

    for result in all_results:

        link = result.get(
            "link",
            ""
        )

        if link in seen_links:

            continue

        seen_links.add(
            link
        )

        unique_results.append(
            result
        )


    all_results = unique_results


    # ========================================================
    # FETCH ACTUAL ARTICLE PAGES
    # ========================================================

    # We fetch trusted sources first.
    # This reduces unnecessary requests.

    trusted_results = [

        result
        for result in all_results
        if result.get(
            "trusted",
            False
        )
    ]

    trusted_results = (
        trusted_results[:max_results]
    )

    enriched_trusted = (
        _attach_article_evidence(
            trusted_results
        )
    )


    # ========================================================
    # MERGE ARTICLE DATA BACK
    # ========================================================

    enriched_by_link = {

        item.get(
            "link",
            ""
        ): item

        for item in enriched_trusted
    }


    final_results = []

    for result in all_results:

        link = result.get(
            "link",
            ""
        )

        if link in enriched_by_link:

            final_results.append(
                enriched_by_link[
                    link
                ]
            )

        else:

            final_results.append(
                result
            )


    all_results = final_results


    # ========================================================
    # REBUILD EXACT / TARGET RESULT LISTS
    # ========================================================

    result_by_link = {

        item.get(
            "link",
            ""
        ): item

        for item in all_results
    }


    exact_results = [

        result_by_link.get(
            item.get(
                "link",
                ""
            ),
            item
        )

        for item in exact_results
    ]


    target_results = [

        result_by_link.get(
            item.get(
                "link",
                ""
            ),
            item
        )

        for item in target_results
    ]


    # ========================================================
    # TRUSTED SOURCE COUNT
    # ========================================================

    trusted_count = sum(

        1

        for result in all_results

        if result.get(
            "trusted",
            False
        )
    )


    # ========================================================
    # ARTICLE FETCH COUNT
    # ========================================================

    article_fetched_count = sum(

        1

        for result in all_results

        if result.get(
            "article_fetched",
            False
        )
    )


    # ========================================================
    # CLAIM ASSESSMENT
    # ========================================================

    assessment = _assess_claim(
        claim,
        exact_results,
        target_results
    )


    # ========================================================
    # NO RESULTS
    # ========================================================

    if not all_results:

        return {

            "status":
                "not_found",

            "query":
                general_query,

            "results":
                [],

            "trusted_count":
                0,

            "article_fetched_count":
                0,

            "claim":
                claim,

            "assessment": {

                "status":
                    "UNVERIFIED",

                "label":
                    "🟡 UNVERIFIED",

                "reason":
                    "No current related sources "
                    "were found."
            },

            "message":
                "No related current news reports "
                "were found."
        }


    # ========================================================
    # RETURN FINAL RESULT
    # ========================================================

    return {

        "status":
            "found",

        "query":
            general_query,

        "results":
            all_results,

        "trusted_count":
            trusted_count,

        "article_fetched_count":
            article_fetched_count,

        "claim":
            claim,

        "assessment":
            assessment,

        "message":
            "Current sources were retrieved. "
            "Trusted publisher pages were fetched "
            "where available, and claim-level evidence "
            "was evaluated."
    }