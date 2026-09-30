# ============================================================
# WEB VERIFICATION MODULE
# Fake News Detection System
#
# Pipeline:
#
#   User Claim
#       ↓
#   Google News RSS - Real-Time Discovery
#       ↓
#   Trusted Source Filtering
#       ↓
#   Publisher Page Fetching
#       ↓
#   HTML Web Scraping
#       ↓
#   Claim-Level Evidence Extraction
#       ↓
#   Entity / Role / Target Matching
#       ↓
#   REAL / FAKE / UNVERIFIED
#
# IMPORTANT:
#   ML prediction is NOT used as final factual truth.
#   Web evidence is evaluated independently.
#
# ============================================================

import re
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

from html import unescape
from html.parser import HTMLParser
from difflib import SequenceMatcher


# ============================================================
# OPTIONAL NLTK
# ============================================================

try:
    from nltk.corpus import stopwords
except Exception:
    stopwords = None


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
    "about.google": "Google",

    "alphabet.com": "Alphabet",

    "tesla.com": "Tesla",

    "sec.gov": "U.S. SEC",

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

    "government of south korea",
}


# ============================================================
# REQUEST SETTINGS
# ============================================================

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/154.0 Safari/537.36"
)

ARTICLE_TIMEOUT = 12
RSS_TIMEOUT = 12


# ============================================================
# ROLE DEFINITIONS
# ============================================================

ROLE_NAMES = (
    r"president|"
    r"prime minister|"
    r"pm|"
    r"ceo|"
    r"chief executive officer|"
    r"governor|"
    r"mayor|"
    r"chancellor"
)


# ============================================================
# BASIC TEXT HELPERS
# ============================================================

def _normalize_text(text):
    """
    Normalize text for comparison.

    Example:

        "Sundar Pichai, CEO of Google!"
            ->
        "sundar pichai ceo of google"
    """

    if not text:
        return ""

    text = unescape(str(text))

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


def _display_text(text):
    """
    Preserve readable text while removing HTML.
    """

    if not text:
        return ""

    text = unescape(str(text))

    text = re.sub(
        r"<[^>]+>",
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

    if stopwords is not None:

        try:
            return set(
                stopwords.words("english")
            )

        except Exception:
            pass

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
        "it",
        "its",
        "their",
        "his",
        "her",
        "who",
        "been",
        "being",
    }


# ============================================================
# QUERY CLEANING
# ============================================================

def _clean_query(text, max_words=12):

    stop_words = _get_stopwords()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        "",
        text or ""
    )

    words = re.findall(
        r"[A-Za-z]{2,}",
        text.lower()
    )

    result = []

    for word in words:

        if word in stop_words:
            continue

        if word not in result:
            result.append(word)

    return " ".join(
        result[:max_words]
    )


# ============================================================
# ROLE NORMALIZATION
# ============================================================

def _normalize_role(role):

    role = (
        role or ""
    ).lower().strip()

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
# FUZZY ENTITY MATCHING
# ============================================================

def _entity_tokens(text):

    normalized = _normalize_text(
        text
    )

    return [
        token
        for token in normalized.split()
        if len(token) >= 2
    ]


def _entity_similarity(first, second):

    first_norm = _normalize_text(
        first
    )

    second_norm = _normalize_text(
        second
    )

    if not first_norm or not second_norm:
        return 0.0

    if first_norm == second_norm:
        return 1.0

    # Direct substring is strong evidence for longer names.
    if (
        first_norm in second_norm
        or second_norm in first_norm
    ):
        return 0.92

    first_tokens = _entity_tokens(
        first
    )

    second_tokens = _entity_tokens(
        second
    )

    if not first_tokens or not second_tokens:
        return 0.0

    # Compare each token to the best token in the other name.
    scores = []

    for token in first_tokens:

        best = max(
            SequenceMatcher(
                None,
                token,
                other
            ).ratio()
            for other in second_tokens
        )

        scores.append(best)

    average_score = sum(scores) / len(scores)

    # For names such as:
    #
    # Sundar pichi
    # Sundar Pichai
    #
    # both tokens match strongly.

    return average_score


def _same_entity(first, second):

    first_norm = _normalize_text(
        first
    )

    second_norm = _normalize_text(
        second
    )

    if not first_norm or not second_norm:
        return False

    if first_norm == second_norm:
        return True

    similarity = _entity_similarity(
        first,
        second
    )

    tokens_a = _entity_tokens(first)
    tokens_b = _entity_tokens(second)

    # Strong fuzzy match for person/entity names.
    if len(tokens_a) >= 2 and len(tokens_b) >= 2:

        if similarity >= 0.84:
            return True

    # Single-word names need a stricter threshold.
    if len(tokens_a) == 1 and len(tokens_b) == 1:

        if similarity >= 0.93:
            return True

    return False


# ============================================================
# SOURCE NAME
# ============================================================

def _source_name(url):

    try:

        host = urllib.parse.urlparse(
            url or ""
        ).netloc.lower()

        host = host.replace(
            "www.",
            ""
        )

        for domain, name in TRUSTED_DOMAINS.items():

            if (
                host == domain
                or host.endswith(
                    "." + domain
                )
            ):
                return name, True

        return (
            host or "Unknown source",
            False
        )

    except Exception:

        return (
            "Unknown source",
            False
        )


def _clean_publisher_name(name):

    name = _display_text(
        name
    )

    return re.sub(
        r"\s*-\s*Google News.*$",
        "",
        name,
        flags=re.I
    ).strip()


def _is_trusted_publisher(name):

    normalized = _normalize_text(
        name
    )

    for publisher in TRUSTED_PUBLISHERS:

        if (
            _normalize_text(publisher)
            in normalized
        ):
            return True

    return False


def _publisher_from_title(title):

    if not title:
        return ""

    parts = re.split(
        r"\s+-\s+",
        title
    )

    if len(parts) >= 2:

        return _clean_publisher_name(
            parts[-1]
        )

    return ""


# ============================================================
# CLAIM EXTRACTION
# ============================================================

def _extract_role_claim(text):

    clean = re.sub(
        r"\s+",
        " ",
        (text or "").strip()
    )

    patterns = [

        # Sundar Pichai is the CEO of Google
        rf"^(.+?)\s+is\s+(?:the\s+)?"
        rf"({ROLE_NAMES})\s+of\s+(.+?)$",

        # Sundar Pichai is CEO Google
        rf"^(.+?)\s+is\s+(?:the\s+)?"
        rf"({ROLE_NAMES})\s+(.+?)$",

        # Sundar Pichai, CEO of Google
        rf"^(.+?),\s*(?:the\s+)?"
        rf"({ROLE_NAMES})\s+of\s+(.+?)$",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            clean,
            flags=re.I
        )

        if not match:
            continue

        subject = (
            match.group(1)
            .strip(" .,:;")
        )

        role = (
            match.group(2)
            .strip()
        )

        target = (
            match.group(3)
            .strip(" .,:;")
        )

        # Remove trailing sentence content.
        target = re.split(
            r"[.!?]\s+",
            target
        )[0].strip()

        if (
            len(subject) < 2
            or len(target) < 2
        ):
            continue

        return {
            "type": "role_claim",
            "subject": subject,
            "role": role,
            "target": target,
        }

    return None


# ============================================================
# HTML ARTICLE SCRAPER
# ============================================================

class _ArticleTextParser(HTMLParser):

    BLOCK_TAGS = {
        "p",
        "article",
        "main",
        "section",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "li",
    }

    SKIP_TAGS = {
        "script",
        "style",
        "noscript",
        "svg",
        "canvas",
        "nav",
        "footer",
        "form",
        "aside",
    }

    def __init__(self):

        super().__init__(
            convert_charrefs=True
        )

        self.title_parts = []

        self.description = ""

        self.text_parts = []

        self.current_tag = None

        self.skip_depth = 0

        self.in_title = False

        self.meta = {}

    def handle_starttag(
        self,
        tag,
        attrs
    ):

        tag = tag.lower()

        attrs_dict = dict(attrs)

        if tag in self.SKIP_TAGS:

            self.skip_depth += 1

            return

        if self.skip_depth:
            return

        if tag == "title":

            self.in_title = True

        if tag == "meta":

            name = (
                attrs_dict.get("name")
                or ""
            ).lower()

            prop = (
                attrs_dict.get("property")
                or ""
            ).lower()

            content = (
                attrs_dict.get("content")
                or ""
            )

            key = name or prop

            if key and content:

                self.meta[key] = (
                    _display_text(
                        content
                    )
                )

        self.current_tag = tag

    def handle_endtag(self, tag):

        tag = tag.lower()

        if tag in self.SKIP_TAGS:

            if self.skip_depth:
                self.skip_depth -= 1

            return

        if self.skip_depth:
            return

        if tag == "title":

            self.in_title = False

        self.current_tag = None

    def handle_data(self, data):

        if self.skip_depth:
            return

        text = _display_text(
            data
        )

        if not text:
            return

        if self.in_title:

            self.title_parts.append(
                text
            )

            return

        if self.current_tag in self.BLOCK_TAGS:

            self.text_parts.append(
                text
            )


# ============================================================
# REAL WEB PAGE SCRAPING
# ============================================================

def _scrape_html(
    url,
    max_chars=150000
):

    if not url:

        return {
            "success": False,
            "url": "",
            "final_url": "",
            "title": "",
            "description": "",
            "text": "",
            "error": "Empty URL",
        }

    try:

        request = urllib.request.Request(

            url,

            headers={
                "User-Agent":
                    USER_AGENT,

                "Accept":
                    (
                        "text/html,"
                        "application/xhtml+xml,"
                        "application/xml;q=0.9,"
                        "*/*;q=0.8"
                    ),

                "Accept-Language":
                    "en-IN,en;q=0.9",
            },
        )

        with urllib.request.urlopen(
            request,
            timeout=ARTICLE_TIMEOUT
        ) as response:

            final_url = response.geturl()

            content_type = (
                response.headers.get(
                    "Content-Type",
                    ""
                ).lower()
            )

            raw = response.read(
                3_000_000
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
                    "Response is not HTML",
            }

        encoding = "utf-8"

        charset_match = re.search(
            r"charset=([A-Za-z0-9._-]+)",
            content_type,
            flags=re.I
        )

        if charset_match:

            encoding = (
                charset_match.group(1)
            )

        html = raw.decode(
            encoding,
            errors="ignore"
        )

        parser = _ArticleTextParser()

        parser.feed(html)

        title = " ".join(
            parser.title_parts
        ).strip()

        description = (
            parser.meta.get(
                "description"
            )
            or parser.meta.get(
                "og:description"
            )
            or parser.meta.get(
                "twitter:description"
            )
            or ""
        )

        chunks = []

        seen = set()

        for part in parser.text_parts:

            clean = re.sub(
                r"\s+",
                " ",
                part
            ).strip()

            if len(clean) < 20:
                continue

            key = clean.lower()

            if key in seen:
                continue

            seen.add(key)

            chunks.append(clean)

        article_text = " ".join(
            chunks
        )

        if len(article_text) > max_chars:

            article_text = (
                article_text[:max_chars]
            )

        return {
            "success":
                bool(article_text),

            "url":
                url,

            "final_url":
                final_url,

            "title":
                title,

            "description":
                description,

            "text":
                article_text,

            "error":
                "",
        }

    except Exception as exc:

        return {
            "success": False,
            "url": url,
            "final_url": url,
            "title": "",
            "description": "",
            "text": "",
            "error": str(exc),
        }


# ============================================================
# SENTENCE HELPERS
# ============================================================

def _split_sentences(text):

    if not text:
        return []

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    parts = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        part.strip()
        for part in parts
        if len(part.strip()) >= 20
    ]


def _source_text(result):

    article_text = result.get(
        "article_text",
        ""
    )

    if article_text:
        return article_text

    return " ".join(
        [
            result.get(
                "article_title",
                ""
            ),
            result.get(
                "article_description",
                ""
            ),
            result.get(
                "title",
                ""
            ),
            result.get(
                "description",
                ""
            ),
        ]
    )


# ============================================================
# ROLE / TARGET HELPERS
# ============================================================

def _role_in_source(
    role,
    source_text
):

    role = _normalize_role(
        role
    )

    source = _normalize_text(
        source_text
    )

    if role == "prime minister":

        return (
            "prime minister"
            in source
            or re.search(
                r"\bpm\b",
                source
            )
            is not None
        )

    if role == "ceo":

        return (
            "ceo" in source
            or
            "chief executive officer"
            in source
        )

    return role in source


def _target_in_source(
    target,
    source_text
):

    target_norm = _normalize_text(
        target
    )

    source = _normalize_text(
        source_text
    )

    if (
        not target_norm
        or not source
    ):
        return False

    if target_norm in source:
        return True

    words = [
        word
        for word in target_norm.split()
        if len(word) >= 3
    ]

    return (
        bool(words)
        and all(
            word in source
            for word in words
        )
    )


# ============================================================
# EXTRACT ROLE HOLDERS
# ============================================================

def _extract_role_holders(
    claim,
    text
):

    """
    Extract explicit role-holder relationships.

    Examples:

        Sundar Pichai is the CEO of Google

        Google CEO Sundar Pichai

        Sundar Pichai, Google CEO

        Tim Cook is the CEO of Google

    IMPORTANT:

    The returned names are candidates.
    The caller must compare them with the
    claimed subject.
    """

    if (
        not text
        or not claim
    ):
        return []

    role = _normalize_role(
        claim.get("role")
    )

    target = _normalize_text(
        claim.get("target")
    )

    if not target:
        return []

    original_sentences = (
        _split_sentences(text)
        if text
        else []
    )

    if not original_sentences:

        original_sentences = [
            text
        ]

    candidates = []

    if role == "ceo":

        role_pattern = (
            r"(?:ceo|"
            r"chief executive officer)"
        )

    elif role == "prime minister":

        role_pattern = (
            r"(?:prime minister|pm)"
        )

    else:

        role_pattern = re.escape(
            role
        )

    target_pattern = re.escape(
        target
    )

    for sentence in original_sentences:

        clean_sentence = (
            sentence.strip()
        )

        normalized = _normalize_text(
            clean_sentence
        )

        # ----------------------------------------------------
        # Pattern 1:
        #
        # Person is the CEO of Google
        # ----------------------------------------------------

        pattern_1 = re.compile(
            rf"\b"
            rf"([a-z][a-z .'\-]{{1,60}})"
            rf"\s+is\s+"
            rf"(?:the\s+)?"
            rf"{role_pattern}"
            rf"\s+(?:of\s+)?"
            rf"{target_pattern}"
            rf"\b",
            re.I
        )

        for match in pattern_1.finditer(
            normalized
        ):

            person = (
                match.group(1)
                .strip(
                    " ,.-"
                )
            )

            if person:
                candidates.append(
                    person
                )

        # ----------------------------------------------------
        # Pattern 2:
        #
        # Google CEO Sundar Pichai
        #
        # This is the important case that fixes:
        #
        # Sundar pichi
        #      vs
        # Sundar Pichai
        # ----------------------------------------------------

        pattern_2 = re.compile(
            rf"\b"
            rf"{target_pattern}"
            rf"(?:'s)?\s+"
            rf"{role_pattern}"
            rf"\s+"
            rf"([a-z][a-z .'\-]{{1,60}})"
            rf"\b",
            re.I
        )

        for match in pattern_2.finditer(
            normalized
        ):

            person = (
                match.group(1)
                .strip(
                    " ,.-"
                )
            )

            # Remove common trailing words.
            person = re.split(
                r"\b(?:said|says|and|who|has|"
                r"will|was|is|after|during)\b",
                person,
                flags=re.I
            )[0].strip()

            if person:
                candidates.append(
                    person
                )

        # ----------------------------------------------------
        # Pattern 3:
        #
        # Sundar Pichai, Google CEO
        # ----------------------------------------------------

        pattern_3 = re.compile(
            rf"\b"
            rf"([a-z][a-z .'\-]{{1,60}})"
            rf",?\s+"
            rf"(?:the\s+)?"
            rf"{target_pattern}"
            rf"(?:'s)?\s+"
            rf"{role_pattern}"
            rf"\b",
            re.I
        )

        for match in pattern_3.finditer(
            normalized
        ):

            person = (
                match.group(1)
                .strip(
                    " ,.-"
                )
            )

            if person:
                candidates.append(
                    person
                )

        # ----------------------------------------------------
        # Pattern 4:
        #
        # CEO Sundar Pichai of Google
        # ----------------------------------------------------

        pattern_4 = re.compile(
            rf"\b"
            rf"{role_pattern}"
            rf"\s+"
            rf"([a-z][a-z .'\-]{{1,60}})"
            rf"\s+of\s+"
            rf"{target_pattern}"
            rf"\b",
            re.I
        )

        for match in pattern_4.finditer(
            normalized
        ):

            person = (
                match.group(1)
                .strip(
                    " ,.-"
                )
            )

            if person:
                candidates.append(
                    person
                )

    # --------------------------------------------------------
    # Clean candidates
    # --------------------------------------------------------

    cleaned = []

    for candidate in candidates:

        candidate = re.sub(
            r"\s+",
            " ",
            candidate
        ).strip(
            " ,.-"
        )

        if not candidate:
            continue

        if len(
            candidate.split()
        ) > 7:
            continue

        if candidate not in cleaned:

            cleaned.append(
                candidate
            )

    return cleaned


# ============================================================
# RELEVANT SENTENCES
# ============================================================

def _find_relevant_sentences(
    claim,
    article_text,
    max_sentences=6
):

    if (
        not article_text
        or not claim
    ):
        return []

    sentences = _split_sentences(
        article_text
    )

    subject = _normalize_text(
        claim.get("subject")
    )

    role = _normalize_role(
        claim.get("role")
    )

    target = _normalize_text(
        claim.get("target")
    )

    scored = []

    for sentence in sentences:

        normalized = _normalize_text(
            sentence
        )

        score = 0

        if subject and subject in normalized:

            score += 5

        else:

            subject_tokens = [
                token
                for token in subject.split()
                if len(token) >= 3
            ]

            score += sum(
                1
                for token in subject_tokens
                if token in normalized
            )

        if role == "ceo":

            if (
                "ceo" in normalized
                or
                "chief executive officer"
                in normalized
            ):
                score += 4

        elif role == "prime minister":

            if (
                "prime minister"
                in normalized
                or re.search(
                    r"\bpm\b",
                    normalized
                )
            ):
                score += 4

        elif role in normalized:

            score += 3

        if target and target in normalized:

            score += 4

        else:

            target_tokens = [
                token
                for token in target.split()
                if len(token) >= 3
            ]

            score += sum(
                1
                for token in target_tokens
                if token in normalized
            )

        if score >= 4:

            scored.append(
                (
                    score,
                    sentence
                )
            )

    scored.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        sentence
        for _, sentence
        in scored[:max_sentences]
    ]


# ============================================================
# STRONG SUPPORT DETECTION
# ============================================================

def _strong_role_support(
    claim,
    result
):

    """
    Determine whether the source explicitly supports:

        SUBJECT + ROLE + TARGET

    We do NOT simply check whether all keywords
    appear somewhere in the article.

    Instead we:

        1. find explicit role holders
        2. compare them with claimed subject
        3. require target + role evidence
    """

    source = _source_text(
        result
    )

    if not source:
        return []

    role = _normalize_role(
        claim.get("role")
    )

    target = _normalize_text(
        claim.get("target")
    )

    subject = claim.get(
        "subject",
        ""
    )

    if not _target_in_source(
        target,
        source
    ):
        return []

    if not _role_in_source(
        role,
        source
    ):
        return []

    # --------------------------------------------------------
    # Find explicit role holders.
    # --------------------------------------------------------

    holders = _extract_role_holders(
        claim,
        source
    )

    # --------------------------------------------------------
    # If any extracted role holder is the
    # same person as the claim subject,
    # the source supports the claim.
    # --------------------------------------------------------

    matching_holders = []

    for holder in holders:

        if _same_entity(
            holder,
            subject
        ):

            matching_holders.append(
                holder
            )

    if not matching_holders:
        return []

    # --------------------------------------------------------
    # Return the actual sentences as evidence.
    # --------------------------------------------------------

    relevant = _find_relevant_sentences(
        claim,
        source
    )

    if relevant:
        return relevant

    # Fallback:
    # source itself contains explicit holder.
    return [
        f"Explicit role-holder evidence: "
        f"{matching_holders[0]}"
    ]


# ============================================================
# CONTRADICTION DETECTION
# ============================================================

def _find_contradictory_sources(
    claim,
    results
):

    contradictions = []

    claimed_subject = claim.get(
        "subject",
        ""
    )

    for result in results:

        if not result.get(
            "trusted",
            False
        ):
            continue

        source_text = (
            result.get(
                "article_text",
                ""
            )
            or result.get(
                "article_description",
                ""
            )
            or result.get(
                "article_title",
                ""
            )
            or result.get(
                "description",
                ""
            )
            or result.get(
                "title",
                ""
            )
        )

        if not source_text:
            continue

        try:

            holders = _extract_role_holders(
                claim,
                source_text
            )

        except Exception:

            holders = []

        if not holders:
            continue

        # ----------------------------------------------------
        # CRITICAL FIX:
        #
        # If the extracted role holder is actually
        # the same person as the claimed subject,
        # it is SUPPORTING evidence.
        #
        # NEVER count it as contradiction.
        # ----------------------------------------------------

        alternate_holders = []

        for holder in holders:

            if _same_entity(
                holder,
                claimed_subject
            ):
                continue

            alternate_holders.append(
                holder
            )

        if not alternate_holders:
            continue

        relevant = _find_relevant_sentences(
            claim,
            source_text
        )

        item = dict(result)

        item[
            "alternate_role_holders"
        ] = alternate_holders

        item[
            "evidence_sentences"
        ] = relevant

        item[
            "evidence_type"
        ] = (
            "publisher_article"
            if result.get(
                "article_text"
            )
            else
            "publisher_title_description"
        )

        contradictions.append(
            item
        )

    return contradictions


# ============================================================
# SUPPORTING SOURCES
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

        evidence = _strong_role_support(
            claim,
            result
        )

        if not evidence:
            continue

        item = dict(result)

        item[
            "evidence_sentences"
        ] = evidence

        item[
            "evidence_type"
        ] = (
            "publisher_article"
            if result.get(
                "article_text"
            )
            else
            "publisher_title_description"
        )

        supporting.append(
            item
        )

    return supporting


# ============================================================
# GOOGLE NEWS RSS SEARCH
# ============================================================

def _search_news(
    query,
    max_results=8
):

    encoded = urllib.parse.quote_plus(
        query
    )

    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={encoded}"
        "&hl=en-IN"
        "&gl=IN"
        "&ceid=IN:en"
    )

    request = urllib.request.Request(

        rss_url,

        headers={
            "User-Agent":
                USER_AGENT
        }
    )

    with urllib.request.urlopen(
        request,
        timeout=RSS_TIMEOUT
    ) as response:

        data = response.read()

    root = ET.fromstring(
        data
    )

    results = []

    for item in root.findall(
        ".//item"
    )[:max_results]:

        title_el = item.find(
            "title"
        )

        link_el = item.find(
            "link"
        )

        desc_el = item.find(
            "description"
        )

        date_el = item.find(
            "pubDate"
        )

        source_el = item.find(
            "source"
        )

        title = (

            _display_text(
                title_el.text
            )

            if (
                title_el is not None
                and title_el.text
            )

            else ""
        )

        link = (

            link_el.text.strip()

            if (
                link_el is not None
                and link_el.text
            )

            else ""
        )

        description = (

            _display_text(
                desc_el.text
            )

            if (
                desc_el is not None
                and desc_el.text
            )

            else ""
        )

        published = (

            date_el.text.strip()

            if (
                date_el is not None
                and date_el.text
            )

            else ""
        )

        rss_source = (

            _display_text(
                source_el.text
            )

            if (
                source_el is not None
                and source_el.text
            )

            else ""
        )

        source_url = ""

        if source_el is not None:

            source_url = (
                source_el.attrib.get(
                    "url",
                    ""
                )
            )

        publisher = (
            rss_source
            or
            _publisher_from_title(
                title
            )
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

            if source_url:

                source_name_from_url, trusted_from_url = (
                    _source_name(
                        source_url
                    )
                )

                if trusted_from_url:

                    source_name = (
                        source_name_from_url
                    )

                    trusted = True

        else:

            source_name, trusted = (
                _source_name(
                    link
                )
            )

        results.append({

            "title":
                title,

            "description":
                description,

            "link":
                link,

            "source":
                source_name,

            "trusted":
                trusted,

            "published":
                published,

            "rss_source_url":
                source_url,

            "article_fetched":
                False,

            "article_url":
                "",

            "article_title":
                "",

            "article_description":
                "",

            "article_text":
                "",

            "article_error":
                "",

            "evidence_type":
                "",

            "evidence_sentences":
                [],
        })

    return results


# ============================================================
# FETCH + SCRAPE PUBLISHER PAGE
# ============================================================

def _attach_article_evidence(
    results
):

    enriched = []

    for result in results:

        item = dict(
            result
        )

        article = _scrape_html(
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

        # ----------------------------------------------------
        # If actual publisher page cannot be fetched,
        # preserve RSS title/description as secondary evidence.
        # ----------------------------------------------------

        if not article[
            "success"
        ]:

            item[
                "article_title"
            ] = item.get(
                "title",
                ""
            )

            item[
                "article_description"
            ] = item.get(
                "description",
                ""
            )

            item[
                "article_text"
            ] = ""

            item[
                "evidence_type"
            ] = (
                "google_news_rss_title_description"
            )

        else:

            item[
                "evidence_type"
            ] = (
                "publisher_article"
            )

        enriched.append(
            item
        )

    return enriched


# ============================================================
# CLAIM ASSESSMENT
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
                (
                    "The submitted text does not "
                    "contain a supported role-based "
                    "claim structure."
                ),

            "supporting_sources":
                [],

            "contradicting_sources":
                [],
        }

    # --------------------------------------------------------
    # Support is evaluated from exact/person search results.
    # --------------------------------------------------------

    supporting = (
        _find_supporting_sources(
            claim,
            exact_results
        )
    )

    # --------------------------------------------------------
    # Contradiction is evaluated from target/role search results.
    # --------------------------------------------------------

    contradictions = (
        _find_contradictory_sources(
            claim,
            target_results
        )
    )

    # --------------------------------------------------------
    # Separate actual publisher scraping from RSS fallback.
    # --------------------------------------------------------

    article_support = [
        item
        for item in supporting
        if item.get(
            "evidence_type"
        ) == "publisher_article"
    ]

    article_contradictions = [
        item
        for item in contradictions
        if item.get(
            "evidence_type"
        ) == "publisher_article"
    ]

    rss_support = [
        item
        for item in supporting
        if item.get(
            "evidence_type"
        ) == "publisher_title_description"
    ]

    rss_contradictions = [
        item
        for item in contradictions
        if item.get(
            "evidence_type"
        ) == "publisher_title_description"
    ]

    # ========================================================
    # DECISION PRIORITY
    # ========================================================

    # --------------------------------------------------------
    # 1. Strong actual article support
    # --------------------------------------------------------

    if article_support:

        return {

            "status":
                "REAL",

            "label":
                "🟢 REAL / VERIFIED",

            "reason":
                (
                    "A trusted publisher page was "
                    "fetched and scraped. The "
                    "retrieved content explicitly "
                    "connects the claimed subject, "
                    "role and target."
                ),

            "supporting_sources":
                article_support,

            "contradicting_sources":
                [],
        }

    # --------------------------------------------------------
    # 2. Strong actual article contradiction
    #
    # Only reaches here if another explicit role holder
    # is identified AND that person is not the claimed person.
    # --------------------------------------------------------

    if article_contradictions:

        return {

            "status":
                "FAKE",

            "label":
                "🔴 FAKE / CONTRADICTED",

            "reason":
                (
                    "A trusted publisher page was "
                    "fetched and scraped. Its content "
                    "explicitly identifies a different "
                    "person as the role holder for "
                    "the stated target."
                ),

            "supporting_sources":
                [],

            "contradicting_sources":
                article_contradictions,
        }

    # --------------------------------------------------------
    # 3. RSS/title support
    # --------------------------------------------------------

    if rss_support:

        return {

            "status":
                "REAL",

            "label":
                "🟢 REAL / VERIFIED",

            "reason":
                (
                    "A trusted current source explicitly "
                    "states the claimed relationship in "
                    "its title or description. The "
                    "publisher article could not be "
                    "fully fetched, so this is treated "
                    "as secondary evidence."
                ),

            "supporting_sources":
                rss_support,

            "contradicting_sources":
                [],
        }

    # --------------------------------------------------------
    # 4. RSS contradiction
    # --------------------------------------------------------

    if rss_contradictions:

        return {

            "status":
                "FAKE",

            "label":
                "🔴 FAKE / CONTRADICTED",

            "reason":
                (
                    "A trusted current source explicitly "
                    "identifies a different person as "
                    "the role holder for the target."
                ),

            "supporting_sources":
                [],

            "contradicting_sources":
                rss_contradictions,
        }

    # --------------------------------------------------------
    # 5. No strong evidence
    # --------------------------------------------------------

    return {

        "status":
            "UNVERIFIED",

        "label":
            "🟡 UNVERIFIED",

        "reason":
            (
                "Current trusted sources were found, "
                "but the retrieved content does not "
                "contain sufficiently explicit evidence "
                "to support or contradict the claim."
            ),

        "supporting_sources":
            [],

        "contradicting_sources":
            [],
    }


# ============================================================
# MAIN VERIFICATION FUNCTION
# ============================================================

def verify_news(
    text,
    max_results=8
):

    """
    Real-time claim verification.

    ML prediction is deliberately NOT used as
    the final verdict.

    Pipeline:

        1. Extract claim
        2. Search current news
        3. Filter trusted sources
        4. Scrape publisher pages
        5. Extract claim-level evidence
        6. Compare entities
        7. Check support / contradiction
        8. Return REAL / FAKE / UNVERIFIED
    """

    # ========================================================
    # EMPTY INPUT
    # ========================================================

    if (
        not text
        or not text.strip()
    ):

        return {

            "status":
                "not_found",

            "query":
                "",

            "results":
                [],

            "trusted_count":
                0,

            "article_fetched_count":
                0,

            "claim":
                None,

            "assessment": {

                "status":
                    "UNVERIFIED",

                "label":
                    "🟡 UNVERIFIED",

                "reason":
                    "No text was provided.",
            },

            "message":
                "Please enter a news claim.",
        }

    # ========================================================
    # CLAIM EXTRACTION
    # ========================================================

    claim = _extract_role_claim(
        text
    )

    general_query = _clean_query(
        text
    )

    # ========================================================
    # SEARCH QUERIES
    # ========================================================

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

        # ----------------------------------------------------
        # Exact claim query
        #
        # Example:
        #
        # "Sundar pichi" "ceo" "google"
        # ----------------------------------------------------

        exact_query = (
            f'"{subject}" '
            f'"{role}" '
            f'"{target}"'
        )

        # ----------------------------------------------------
        # Role + target query
        #
        # This is very important for typo cases.
        #
        # Even if:
        #
        # Sundar pichi
        #
        # does not appear in search results,
        # this query can find:
        #
        # Google CEO Sundar Pichai
        # ----------------------------------------------------

        target_query = (
            f'"{target}" '
            f'"{role}"'
        )

        # ----------------------------------------------------
        # Role target without quotes.
        #
        # Useful for natural web results.
        # ----------------------------------------------------

        broad_query = (
            f"{target} "
            f"{role}"
        )

        queries = [
            exact_query,
            target_query,
            broad_query,
        ]

    else:

        queries = [
            general_query
        ]

    # ========================================================
    # REAL-TIME SEARCH
    # ========================================================

    try:

        exact_results = (
            _search_news(
                queries[0],
                max_results
            )
        )

        target_results = []

        if claim:

            target_results = (
                _search_news(
                    queries[1],
                    max_results
                )
            )

            broad_results = (
                _search_news(
                    queries[2],
                    max_results
                )
            )

        else:

            broad_results = []

    except Exception as exc:

        return {

            "status":
                "error",

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
                    (
                        "Real-time verification "
                        "could not be completed: "
                        + str(exc)
                    ),

                "supporting_sources":
                    [],

                "contradicting_sources":
                    [],
            },

            "message":
                (
                    "Real-time source verification "
                    "is currently unavailable."
                ),
        }

    # ========================================================
    # MERGE RESULTS
    # ========================================================

    combined = (
        exact_results
        + target_results
        + broad_results
    )

    unique = []

    seen = set()

    for result in combined:

        key = (
            result.get(
                "link",
                ""
            )
            or
            result.get(
                "title",
                ""
            )
        )

        if key in seen:
            continue

        seen.add(key)

        unique.append(
            result
        )

    all_results = unique

    # ========================================================
    # FETCH ONLY TRUSTED SOURCES
    # ========================================================

    trusted_results = [

        result

        for result in all_results

        if result.get(
            "trusted",
            False
        )

    ][:max_results]

    # ========================================================
    # REAL WEB SCRAPING
    # ========================================================

    enriched = (
        _attach_article_evidence(
            trusted_results
        )
    )

    by_link = {

        result.get(
            "link",
            ""
        ):
        result

        for result in enriched
    }

    final_results = []

    for result in all_results:

        link = result.get(
            "link",
            ""
        )

        if link in by_link:

            final_results.append(
                by_link[link]
            )

        else:

            final_results.append(
                result
            )

    all_results = final_results

    # ========================================================
    # REBUILD SEARCH RESULT LISTS
    # ========================================================

    result_by_link = {

        result.get(
            "link",
            ""
        ):
        result

        for result in all_results
    }

    exact_results = [

        result_by_link.get(
            result.get(
                "link",
                ""
            ),
            result
        )

        for result in exact_results
    ]

    target_results = [

        result_by_link.get(
            result.get(
                "link",
                ""
            ),
            result
        )

        for result in target_results
    ]

    broad_results = [

        result_by_link.get(
            result.get(
                "link",
                ""
            ),
            result
        )

        for result in broad_results
    ]

    # ========================================================
    # COUNTS
    # ========================================================

    trusted_count = sum(

        1

        for result in all_results

        if result.get(
            "trusted",
            False
        )
    )

    article_fetched_count = sum(

        1

        for result in all_results

        if result.get(
            "article_fetched",
            False
        )
    )

    # ========================================================
    # IMPORTANT:
    #
    # For SUPPORT:
    #
    #   exact results + broad results
    #
    # For CONTRADICTION:
    #
    #   target/broad results
    #
    # ========================================================

    support_results = []

    seen_support = set()

    for result in (
        exact_results
        + broad_results
        + target_results
    ):

        key = result.get(
            "link",
            ""
        )

        if key in seen_support:
            continue

        seen_support.add(
            key
        )

        support_results.append(
            result
        )

    contradiction_results = []

    seen_contradiction = set()

    for result in (
        target_results
        + broad_results
    ):

        key = result.get(
            "link",
            ""
        )

        if key in seen_contradiction:
            continue

        seen_contradiction.add(
            key
        )

        contradiction_results.append(
            result
        )

    # ========================================================
    # FINAL EVIDENCE ASSESSMENT
    # ========================================================

    assessment = _assess_claim(

        claim,

        support_results,

        contradiction_results
    )

    # ========================================================
    # NO RESULTS
    # ========================================================

    if not all_results:

        assessment = {

            "status":
                "UNVERIFIED",

            "label":
                "🟡 UNVERIFIED",

            "reason":
                (
                    "No current related sources "
                    "were found."
                ),

            "supporting_sources":
                [],

            "contradicting_sources":
                [],
        }

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "status":
            (
                "found"
                if all_results
                else
                "not_found"
            ),

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
            (
                "Verification completed using "
                "current source discovery, "
                "real-time publisher-page "
                "scraping and claim-level "
                "evidence matching."
                if all_results

                else

                "No related current news "
                "reports were found."
            ),
    }