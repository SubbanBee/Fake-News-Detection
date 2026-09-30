# ============================================================
# WEB VERIFICATION MODULE
# Fake News Detection System
#
# Purpose:
#   1. Search current news sources
#   2. Extract evidence from trusted sources
#   3. Check whether the evidence actually supports
#      or contradicts the submitted claim
#   4. Never treat simple keyword overlap as proof
#
# Final statuses:
#   REAL / VERIFIED
#   FAKE / CONTRADICTED
#   UNVERIFIED
# ============================================================

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html import unescape

from nltk.corpus import stopwords


# ============================================================
# TRUSTED SOURCES
# ============================================================

TRUSTED_DOMAINS = {
    "reuters.com": "Reuters",
    "apnews.com": "Associated Press",
    "bbc.com": "BBC",
    "thehindu.com": "The Hindu",
    "indianexpress.com": "Indian Express",
    "timesofindia.indiatimes.com": "Times of India",
    "ndtv.com": "NDTV",
    "hindustantimes.com": "Hindustan Times",
    "npr.org": "NPR",
    "gov.in": "Government of India",
    "pib.gov.in": "Press Information Bureau",
    "who.int": "World Health Organization",
    "un.org": "United Nations",
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
}


# ============================================================
# ROLE / CLAIM PATTERNS
# ============================================================

ROLE_PATTERNS = [
    r"\b(.+?)\s+is\s+(?:the\s+)?(president|prime minister|pm|ceo|chief executive officer|governor|mayor|chancellor)\s+of\s+(.+)",
    r"\b(.+?)\s+is\s+(?:the\s+)?(president|prime minister|pm|ceo|chief executive officer|governor|mayor|chancellor)\s+(.+)",
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
        }


# ============================================================
# CLEAN SEARCH QUERY
# ============================================================

def _clean_query(text, max_words=12):

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
            unique_words.append(word)

    return " ".join(
        unique_words[:max_words]
    )


# ============================================================
# EXTRACT SIMPLE CLAIM
#
# Example:
#
# Narendra Modi is the PM of Korea
#
# subject = Narendra Modi
# role    = PM
# object  = Korea
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

            subject = match.group(1).strip()
            role = match.group(2).strip()
            target = match.group(3).strip()

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

    role = role.lower().strip()

    mapping = {
        "pm": "prime minister",
        "prime minister": "prime minister",
        "president": "president",
        "ceo": "ceo",
        "chief executive officer": "ceo",
        "governor": "governor",
        "mayor": "mayor",
        "chancellor": "chancellor",
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

        hostname = parsed.netloc.lower()

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

        return "Unknown Source", False


# ============================================================
# EXTRACT PUBLISHER
# ============================================================

def _extract_publisher(title):

    if not title:
        return None

    parts = title.rsplit(
        " - ",
        1
    )

    if len(parts) == 2:

        publisher = parts[1].strip()

        if publisher:
            return publisher

    return None


# ============================================================
# TRUSTED PUBLISHER CHECK
# ============================================================

def _is_trusted_publisher(publisher):

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
# CLEAN PUBLISHER
# ============================================================

def _clean_publisher_name(publisher):

    if not publisher:
        return "Unknown Source"

    publisher = publisher.strip()

    publisher = re.sub(
        r"\s+",
        " ",
        publisher
    )

    return publisher


# ============================================================
# FETCH GOOGLE NEWS RSS
# ============================================================

def _search_news(query, max_results=8):

    encoded_query = urllib.parse.quote_plus(
        query
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
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urllib.request.urlopen(
        request,
        timeout=10
    ) as response:

        data = response.read()

    root = ET.fromstring(data)

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

        description_element = item.find(
            "description"
        )

        description = (
            description_element.text.strip()
            if (
                description_element is not None
                and description_element.text
            )
            else ""
        )

        pub_date_element = item.find(
            "pubDate"
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
                _source_name(link)
            )

        results.append({

            "title": title,

            "description": description,

            "link": link,

            "source": source_name,

            "trusted": trusted,

            "published": published
        })

    return results


# ============================================================
# NORMALIZED SOURCE TEXT
# ============================================================

def _source_text(result):

    return _normalize_text(
        (
            result.get("title", "")
            + " "
            + result.get("description", "")
        )
    )


# ============================================================
# ENTITY MATCH
# ============================================================

def _entity_in_source(entity, source_text):

    entity_words = [
        word
        for word in _normalize_text(
            entity
        ).split()
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

def _role_in_source(role, source_text):

    role = _normalize_role(role)

    if role == "prime minister":

        return (
            "prime minister" in source_text
            or " pm " in f" {source_text} "
        )

    if role == "ceo":

        return (
            "ceo" in source_text
            or "chief executive officer"
            in source_text
        )

    return role in source_text


# ============================================================
# TARGET MATCH
# ============================================================

def _target_in_source(target, source_text):

    target_normalized = _normalize_text(
        target
    )

    if not target_normalized:
        return False

    target_words = [
        word
        for word in target_normalized.split()
        if len(word) >= 3
    ]

    return all(
        word in source_text
        for word in target_words
    )


# ============================================================
# STRONG CLAIM SUPPORT CHECK
#
# IMPORTANT:
# Mere word overlap is NOT support.
# ============================================================

def _strong_role_support(
    claim,
    result
):

    source_text = _source_text(
        result
    )

    subject_match = _entity_in_source(
        claim["subject"],
        source_text
    )

    role_match = _role_in_source(
        claim["role"],
        source_text
    )

    target_match = _target_in_source(
        claim["target"],
        source_text
    )

    if not (
        subject_match
        and role_match
        and target_match
    ):
        return False

    # Strong patterns where subject is explicitly
    # connected with the role.
    subject = re.escape(
        _normalize_text(
            claim["subject"]
        )
    )

    target = re.escape(
        _normalize_text(
            claim["target"]
        )
    )

    role = _normalize_role(
        claim["role"]
    )

    patterns = [

        rf"{subject}.*{role}.*{target}",

        rf"{subject}.*{target}.*{role}",

        rf"{role}.*{subject}.*{target}",

        rf"{target}.*{role}.*{subject}",
    ]

    for pattern in patterns:

        if re.search(
            pattern,
            source_text
        ):

            return True

    return False


# ============================================================
# FIND SOURCES THAT SUPPORT CLAIM
# ============================================================

def _find_supporting_sources(
    claim,
    results
):

    supporting = []

    for result in results:

        if not result["trusted"]:
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
#
# For a role claim:
#
# "X is PM of Y"
#
# We search:
#
# "Y prime minister"
#
# If trusted sources clearly identify another
# person as the role holder, the submitted
# claim can be treated as contradicted.
# ============================================================

def _find_contradictory_sources(
    claim,
    target_results
):

    contradictions = []

    subject_normalized = _normalize_text(
        claim["subject"]
    )

    role_normalized = _normalize_role(
        claim["role"]
    )

    for result in target_results:

        if not result["trusted"]:
            continue

        text = _source_text(
            result
        )

        if not _role_in_source(
            role_normalized,
            text
        ):
            continue

        # If the source discusses the target + role
        # but does NOT contain the claimed subject,
        # it may identify the actual office holder.
        #
        # We require explicit role wording and target
        # presence to avoid simple keyword mistakes.

        target_present = _target_in_source(
            claim["target"],
            text
        )

        subject_present = (
            subject_normalized
            and subject_normalized in text
        )

        if (
            target_present
            and not subject_present
        ):

            contradictions.append(
                result
            )

    return contradictions


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
            "status": "UNVERIFIED",

            "label": "🟡 UNVERIFIED",

            "reason":
                "The submitted text does not contain "
                "a claim structure that can be "
                "reliably checked with the current "
                "rule-based verifier.",

            "supporting_sources": [],

            "contradicting_sources": []
        }


    supporting = _find_supporting_sources(
        claim,
        exact_results
    )

    contradictions = (
        _find_contradictory_sources(
            claim,
            target_results
        )
    )


    # --------------------------------------------------------
    # STRONG SUPPORT
    # --------------------------------------------------------

    if supporting:

        return {

            "status": "REAL",

            "label": "🟢 REAL / VERIFIED",

            "reason":
                "A trusted current source explicitly "
                "connects the claimed person/entity "
                "with the stated role and target. "
                "The claim is supported by evidence.",

            "supporting_sources":
                supporting,

            "contradicting_sources":
                contradictions
        }


    # --------------------------------------------------------
    # STRONG CONTRADICTION
    # --------------------------------------------------------

    if contradictions:

        return {

            "status": "FAKE",

            "label":
                "🔴 FAKE / CONTRADICTED",

            "reason":
                "Trusted current sources discuss "
                "the stated role and target but do "
                "not identify the claimed subject as "
                "the role holder. This provides "
                "contradictory evidence.",

            "supporting_sources":
                supporting,

            "contradicting_sources":
                contradictions
        }


    # --------------------------------------------------------
    # NO STRONG EVIDENCE
    # --------------------------------------------------------

    return {

        "status": "UNVERIFIED",

        "label": "🟡 UNVERIFIED",

        "reason":
            "Current sources were found, but they "
            "do not provide strong claim-level "
            "evidence supporting or contradicting "
            "the submitted statement.",

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
        ML prediction is NOT used here to determine
        factual truth.

    The verifier searches current sources and only
    marks REAL/FAKE when there is claim-level evidence.

    Otherwise:
        UNVERIFIED
    """

    if not text or not text.strip():

        return {

            "status": "not_found",

            "query": "",

            "results": [],

            "trusted_count": 0,

            "claim": None,

            "assessment": {

                "status": "UNVERIFIED",

                "label": "🟡 UNVERIFIED",

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
    # BUILD SEARCHES
    # ========================================================

    general_query = _clean_query(
        text
    )

    if claim:

        subject = claim["subject"]
        role = claim["role"]
        target = claim["target"]

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
    # SEARCH
    # ========================================================

    all_results = []

    target_results = []

    try:

        # Exact claim search
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

            all_results = exact_results


    except Exception as e:

        return {

            "status": "error",

            "query": general_query,

            "results": [],

            "trusted_count": 0,

            "claim": claim,

            "assessment": {

                "status": "UNVERIFIED",

                "label": "🟡 UNVERIFIED",

                "reason":
                    "Real-time verification could not "
                    f"be completed: {str(e)}"
            },

            "message":
                "Real-time source verification "
                "is currently unavailable."
        }


    # ========================================================
    # REMOVE DUPLICATES
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

        seen_links.add(link)

        unique_results.append(
            result
        )


    all_results = unique_results


    # ========================================================
    # TRUSTED SOURCE COUNT
    # ========================================================

    trusted_count = sum(
        1
        for result in all_results
        if result["trusted"]
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

            "status": "not_found",

            "query": general_query,

            "results": [],

            "trusted_count": 0,

            "claim": claim,

            "assessment": {

                "status": "UNVERIFIED",

                "label": "🟡 UNVERIFIED",

                "reason":
                    "No current related sources were found."
            },

            "message":
                "No related current news reports "
                "were found."
        }


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "status": "found",

        "query": general_query,

        "results": all_results,

        "trusted_count": trusted_count,

        "claim": claim,

        "assessment": assessment,

        "message":
            "Current sources were retrieved and "
            "evaluated using claim-level evidence."
    }