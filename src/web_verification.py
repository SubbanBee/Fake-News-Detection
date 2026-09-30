
import re
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from html import unescape
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple, Set


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
    "cia.gov": "CIA",
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


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/154.0 Safari/537.36"
)

RSS_TIMEOUT = 12
ARTICLE_TIMEOUT = 12
MAX_ARTICLE_BYTES = 3_000_000
MAX_ARTICLE_CHARS = 150_000


# ============================================================
# TEMPORAL STATE INDICATORS
# ============================================================

TEMPORAL_CURRENT = {
    "is", "remains", "currently", "serves as", "serving as",
    "continues to", "still", "now", "present"
}

TEMPORAL_FORMER = {
    "was", "former", "previous", "ex-", "used to be",
    "formerly", "previously", "past"
}

TEMPORAL_PLANNED = {
    "will", "scheduled", "planned", "expected", "set to",
    "upcoming", "future", "will be", "going to"
}

TEMPORAL_COMPLETED = {
    "completed", "finished", "ended", "concluded", "done",
    "has", "have", "did", "successfully"
}

TEMPORAL_CANCELLED = {
    "cancelled", "postponed", "delayed", "called off",
    "scrapped", "abandoned"
}


# ============================================================
# ROLE DEFINITIONS
# ============================================================

ROLE_ALIASES = {
    "pm": "prime minister",
    "prime minister": "prime minister",
    "president": "president",
    "ceo": "ceo",
    "chief executive officer": "ceo",
    "governor": "governor",
    "mayor": "mayor",
    "chancellor": "chancellor",
}


ROLE_WORDS = {
    "prime minister": ["prime minister", "pm"],
    "president": ["president"],
    "ceo": ["ceo", "chief executive officer"],
    "governor": ["governor"],
    "mayor": ["mayor"],
    "chancellor": ["chancellor"],
}


# ============================================================
# BASIC TEXT HELPERS
# ============================================================

def _display_text(text):
    if not text:
        return ""
    text = unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_text(text):
    if not text:
        return ""
    text = _display_text(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_role(role):
    role = _normalize_text(role)
    return ROLE_ALIASES.get(role, role)


def _words(text):
    return re.findall(r"[a-z0-9]+", _normalize_text(text))


def _clean_query(text, max_words=16):
    stop_words = {
        "the", "is", "a", "an", "of", "to", "in", "on", "for",
        "and", "or", "was", "were", "are", "be", "as", "at", "by",
        "with", "has", "have", "had", "this", "that", "from", "it",
        "its", "their", "his", "her", "who",
    }
    text = re.sub(r"https?://\S+|www\.\S+", "", text or "")
    words = re.findall(r"[A-Za-z0-9]{2,}", text.lower())
    result = []
    for word in words:
        if word in stop_words:
            continue
        if word not in result:
            result.append(word)
    return " ".join(result[:max_words])


# ============================================================
# TEMPORAL STATE DETECTION
# ============================================================

def _detect_temporal_state(text):
    """
    Detects the temporal state of a claim or evidence:
    CURRENT, FORMER, PLANNED, COMPLETED, CANCELLED, or UNKNOWN
    """
    normalized = _normalize_text(text)
    words = set(normalized.split())
    
    # Check for planned indicators
    if any(word in TEMPORAL_PLANNED for word in words):
        return "PLANNED"
    
    # Check for completed indicators
    if any(word in TEMPORAL_COMPLETED for word in words):
        return "COMPLETED"
    
    # Check for cancelled indicators
    if any(word in TEMPORAL_CANCELLED for word in words):
        return "CANCELLED"
    
    # Check for former indicators
    if any(word in TEMPORAL_FORMER for word in words):
        return "FORMER"
    
    # Check for current indicators
    if any(word in TEMPORAL_CURRENT for word in words):
        return "CURRENT"
    
    return "UNKNOWN"


# ============================================================
# FUZZY ENTITY MATCHING
# ============================================================

def _similarity(a, b):
    a = _normalize_text(a)
    b = _normalize_text(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _entity_match(claimed, text, threshold=0.92):
    """
    Determines whether a claimed entity appears in source text.
    
    STRICTER RULES:
    - For single-word names (e.g., "Pallavi"), require exact match or very high similarity
    - For multi-word names, allow fuzzy matching
    - Context must support the entity match
    """
    claimed = _normalize_text(claimed)
    source = _normalize_text(text)
    if not claimed or not source:
        return False
    
    claimed_words = claimed.split()
    if not claimed_words:
        return False
    
    source_words = source.split()
    n = len(claimed_words)
    
    # Single-word entity (e.g., "Pallavi") - STRICTER matching
    if n == 1:
        # First try exact match
        if claimed in source:
            return True
        
        # For short names, require very high similarity and minimum length
        for word in source_words:
            if len(word) < 4:
                continue
            # If it's a common first name, require near-exact match
            if len(claimed_words[0]) <= 7:  # Short name like "Pallavi"
                if _similarity(claimed_words[0], word) >= 0.98:
                    return True
            else:
                if _similarity(claimed_words[0], word) >= threshold:
                    return True
        return False
    
    # Multi-word entity (e.g., "Sundar Pichai") - allow fuzzy matching
    if claimed in source:
        return True
    
    for i in range(len(source_words) - n + 1):
        candidate = " ".join(source_words[i:i + n])
        score = _similarity(claimed, candidate)
        if score >= threshold:
            return True
    
    return False


def _find_entity_in_sentence(entity, sentence):
    """
    Returns the best matching occurrence of an entity inside a sentence.
    """
    entity_norm = _normalize_text(entity)
    sentence_norm = _normalize_text(sentence)
    if not entity_norm or not sentence_norm:
        return None
    if entity_norm in sentence_norm:
        return entity_norm
    entity_words = entity_norm.split()
    sentence_words = sentence_norm.split()
    n = len(entity_words)
    if n == 0:
        return None
    best = None
    best_score = 0.0
    if n == 1:
        for word in sentence_words:
            score = _similarity(entity_words[0], word)
            if score > best_score:
                best_score = score
                best = word
    else:
        for i in range(len(sentence_words) - n + 1):
            candidate = " ".join(sentence_words[i:i + n])
            score = _similarity(entity_norm, candidate)
            if score > best_score:
                best_score = score
                best = candidate
    if best_score >= 0.92:
        return best
    return None


def _check_contextual_coherence(claim, sentence):
    """
    Checks if the sentence contains the full context needed to support the claim.
    
    For a claim like "Pallavi is CEO of Google", the sentence must contain:
    - The entity (Pallavi)
    - The role (CEO)
    - The target (Google)
    
    All three must appear in the SAME sentence/context to be considered evidence.
    """
    if not claim or not sentence:
        return False
    
    sentence_norm = _normalize_text(sentence)
    
    # Check for role claims
    if claim["type"] == "role_claim":
        subject = _normalize_text(claim["subject"])
        role = _normalize_role(claim["role"])
        target = _normalize_text(claim["target"])
        
        # All three must be present
        if not _entity_match(subject, sentence_norm, threshold=0.95):
            return False
        if not _role_present(role, sentence_norm):
            return False
        if not _target_present(target, sentence_norm):
            return False
        
        return True
    
    # Check for event claims
    elif claim["type"] == "event_claim":
        event = _normalize_text(claim["event"])
        context = claim.get("context", "")
        
        if not _entity_match(event, sentence_norm, threshold=0.95):
            return False
        if context and not _entity_match(context, sentence_norm, threshold=0.90):
            return False
        
        return True
    
    # Check for numeric claims
    elif claim["type"] == "numeric_claim":
        value = claim.get("value", "")
        context = claim.get("context", "")
        
        if value and value not in sentence_norm:
            return False
        if context and not _entity_match(context, sentence_norm, threshold=0.90):
            return False
        
        return True
    
    return False


def _is_same_entity(claimed_entity, source_entity, context=None):
    """
    Generic entity disambiguation.
    
    Determines if two entity references refer to the SAME entity based on:
    - Full name match
    - Contextual clues (organization, role, location)
    - Similarity thresholds
    
    Returns True if the entities are likely the same, False otherwise.
    """
    claimed_norm = _normalize_text(claimed_entity)
    source_norm = _normalize_text(source_entity)
    
    if not claimed_norm or not source_norm:
        return False
    
    # Exact match
    if claimed_norm == source_norm:
        return True
    
    # Very high similarity (near-exact)
    if _similarity(claimed_norm, source_norm) >= 0.98:
        return True
    
    # For short names (single word), be very strict
    claimed_words = claimed_norm.split()
    source_words = source_norm.split()
    
    if len(claimed_words) == 1 and len(source_words) == 1:
        # Short common names require near-exact match
        if len(claimed_words[0]) <= 8:
            return _similarity(claimed_norm, source_norm) >= 0.98
        else:
            return _similarity(claimed_norm, source_norm) >= 0.95
    
    # For multi-word names, allow some flexibility
    if len(claimed_words) > 1 and len(source_words) > 1:
        return _similarity(claimed_norm, source_norm) >= 0.92
    
    # If one is multi-word and one is single-word, they're likely different
    # unless context strongly suggests otherwise
    if context:
        # If context is provided, check if it helps disambiguate
        context_norm = _normalize_text(context)
        if claimed_norm in context_norm and source_norm in context_norm:
            return True
    
    return False


# ============================================================
# SOURCE IDENTIFICATION
# ============================================================

def _source_name(url):
    try:
        host = urllib.parse.urlparse(url or "").netloc.lower()
        host = host.replace("www.", "")
        for domain, name in TRUSTED_DOMAINS.items():
            if host == domain or host.endswith("." + domain):
                return name, True
        return host or "Unknown source", False
    except Exception:
        return "Unknown source", False


def _clean_publisher_name(name):
    name = _display_text(name)
    return re.sub(r"\s*-\s*Google News.*$", "", name, flags=re.I).strip()


def _is_trusted_publisher(name):
    normalized = _normalize_text(name)
    if not normalized:
        return False
    for publisher in TRUSTED_PUBLISHERS:
        if _normalize_text(publisher) in normalized:
            return True
    return False


def _publisher_from_title(title):
    if not title:
        return ""
    parts = re.split(r"\s+-\s+", title)
    if len(parts) >= 2:
        return _clean_publisher_name(parts[-1])
    return ""


# ============================================================
# CLAIM EXTRACTION - MULTIPLE TYPES
# ============================================================

def _extract_role_claim(text):
    """
    Extracts role claims: Person is ROLE of TARGET
    """
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if not clean:
        return None
    
    patterns = [
        r"^\s*(.+?)\s+is\s+(?:the\s+)?(prime minister|pm|president|ceo|chief executive officer|governor|mayor|chancellor)\s+of\s+(.+?)\s*[.!?]?\s*$",
        r"^\s*(.+?)\s+is\s+(?:the\s+)?(prime minister|pm|president|ceo|chief executive officer|governor|mayor|chancellor)\s+(.+?)\s*[.!?]?\s*$",
        r"^\s*(.+?),\s*(?:the\s+)?(prime minister|pm|president|ceo|chief executive officer|governor|mayor|chancellor)\s+of\s+(.+?)\s*[.!?]?\s*$",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, clean, flags=re.I)
        if not match:
            continue
        subject = match.group(1).strip(" .,:;")
        role = match.group(2).strip()
        target = match.group(3).strip(" .,:;")
        target = re.split(r"[.!?]", target)[0].strip()
        if len(subject) < 2 or len(target) < 2:
            continue
        return {
            "type": "role_claim",
            "subject": subject,
            "role": role,
            "target": target,
            "temporal_state": _detect_temporal_state(clean),
        }
    return None


def _extract_event_claim(text):
    """
    Extracts event claims: Event happened/will happen
    """
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if not clean:
        return None
    
    # Simple pattern: [Event] [verb] [location/context]
    patterns = [
        r"^\s*(.+?)\s+(happened|occurred|took place|will happen|will occur|is scheduled|is planned)\s+(?:in|at|on|during)?\s*(.+?)\s*[.!?]?\s*$",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, clean, flags=re.I)
        if not match:
            continue
        event = match.group(1).strip(" .,:;")
        action = match.group(2).strip()
        context = match.group(3).strip(" .,:;")
        if len(event) < 2:
            continue
        return {
            "type": "event_claim",
            "event": event,
            "action": action,
            "context": context,
            "temporal_state": _detect_temporal_state(clean),
        }
    return None


def _extract_numeric_claim(text):
    """
    Extracts numeric/statistical claims
    """
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if not clean:
        return None
    
    # Pattern: Number + unit + context
    patterns = [
        r"^\s*(\d+(?:\.\d+)?)\s*%?\s*(.+?)\s*[.!?]?\s*$",
        r"^\s*(.+?)\s+(increased|decreased|rose|fell|grew|dropped)\s+by\s*(\d+(?:\.\d+)?)\s*%?\s*[.!?]?\s*$",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, clean, flags=re.I)
        if not match:
            continue
        if pattern.startswith(r"^\s*(\d+"):
            value = match.group(1)
            context = match.group(2).strip(" .,:;")
            return {
                "type": "numeric_claim",
                "value": value,
                "context": context,
                "temporal_state": _detect_temporal_state(clean),
            }
        else:
            context = match.group(1).strip(" .,:;")
            action = match.group(2).strip()
            value = match.group(3)
            return {
                "type": "numeric_claim",
                "context": context,
                "action": action,
                "value": value,
                "temporal_state": _detect_temporal_state(clean),
            }
    return None


def _extract_claim(text):
    """
    Generic claim extraction - tries multiple claim types
    """
    if not text or not text.strip():
        return None
    
    # Try role claim first (most common)
    claim = _extract_role_claim(text)
    if claim:
        return claim
    
    # Try event claim
    claim = _extract_event_claim(text)
    if claim:
        return claim
    
    # Try numeric claim
    claim = _extract_numeric_claim(text)
    if claim:
        return claim
    
    return None


# ============================================================
# HTML ARTICLE SCRAPER
# ============================================================

class _ArticleTextParser(HTMLParser):
    BLOCK_TAGS = {"p", "article", "main", "section", "div", "h1", "h2", "h3", "h4", "li"}
    SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "nav", "footer", "form", "aside"}
    
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title_parts = []
        self.text_parts = []
        self.meta = {}
        self.skip_depth = 0
        self.in_title = False
        self.current_tag = None
    
    def handle_starttag(self, tag, attrs):
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
            name = (attrs_dict.get("name") or "").lower()
            prop = (attrs_dict.get("property") or "").lower()
            content = attrs_dict.get("content") or ""
            key = name or prop
            if key and content:
                self.meta[key] = _display_text(content)
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
        text = _display_text(data)
        if not text:
            return
        if self.in_title:
            self.title_parts.append(text)
            return
        if self.current_tag in self.BLOCK_TAGS:
            self.text_parts.append(text)


def _scrape_html(url, max_chars=MAX_ARTICLE_CHARS):
    if not url:
        return {
            "success": False, "url": "", "final_url": "",
            "title": "", "description": "", "text": "",
            "error": "Empty URL",
        }
    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-IN,en;q=0.9",
            },
        )
        with urllib.request.urlopen(request, timeout=ARTICLE_TIMEOUT) as response:
            final_url = response.geturl()
            content_type = response.headers.get("Content-Type", "").lower()
            raw = response.read(MAX_ARTICLE_BYTES)
        
        if "html" not in content_type and not raw.lstrip().startswith(b"<"):
            return {
                "success": False, "url": url, "final_url": final_url,
                "title": "", "description": "", "text": "",
                "error": "Response is not HTML",
            }
        
        encoding = "utf-8"
        charset = re.search(r"charset=([A-Za-z0-9._-]+)", content_type, flags=re.I)
        if charset:
            encoding = charset.group(1)
        
        html = raw.decode(encoding, errors="ignore")
        parser = _ArticleTextParser()
        parser.feed(html)
        
        title = " ".join(parser.title_parts).strip()
        description = (
            parser.meta.get("description")
            or parser.meta.get("og:description")
            or parser.meta.get("twitter:description")
            or ""
        )
        
        chunks = []
        seen = set()
        for part in parser.text_parts:
            clean = re.sub(r"\s+", " ", part).strip()
            if len(clean) < 20:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            chunks.append(clean)
        
        article_text = " ".join(chunks)
        if len(article_text) > max_chars:
            article_text = article_text[:max_chars]
        
        return {
            "success": bool(article_text),
            "url": url,
            "final_url": final_url,
            "title": title,
            "description": description,
            "text": article_text,
            "error": "",
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
# SENTENCE PROCESSING
# ============================================================

def _split_sentences(text):
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if len(part.strip()) >= 15]


def _role_present(role, text):
    normalized = _normalize_text(text)
    role = _normalize_role(role)
    aliases = ROLE_WORDS.get(role, [role])
    for alias in aliases:
        alias_norm = _normalize_text(alias)
        if alias_norm and alias_norm in normalized:
            return True
    return False


def _target_present(target, text):
    return _entity_match(target, text, threshold=0.92)


# ============================================================
# EVIDENCE CLASSIFICATION
# ============================================================

def _classify_evidence_role_claim(claim, sentence):
    """
    Classifies a sentence as SUPPORT, CONTRADICTION, UNRELATED, or INSUFFICIENT
    for a role claim.
    
    STRICT CRITERIA:
    - SUPPORT: Explicit subject+role+target with CURRENT temporal state
    - CONTRADICTION: Explicit DIFFERENT person+role+target with CURRENT state
    - UNRELATED: Related but doesn't establish the claim
    - INSUFFICIENT: Weak, ambiguous, or partial evidence
    
    CRITICAL: All three (subject, role, target) must appear in the SAME sentence/context.
    """
    if not claim or not sentence:
        return "INSUFFICIENT"
    
    subject = _normalize_text(claim["subject"])
    role = _normalize_role(claim["role"])
    target = _normalize_text(claim["target"])
    claim_temporal = claim.get("temporal_state", "UNKNOWN")
    
    sentence_norm = _normalize_text(sentence)
    sentence_temporal = _detect_temporal_state(sentence)
    
    # CRITICAL: Check contextual coherence first
    # All three elements must be present in the same sentence
    if not _check_contextual_coherence(claim, sentence):
        return "UNRELATED"
    
    # Check if subject is present with stricter matching
    subject_found = _find_entity_in_sentence(subject, sentence_norm)
    if not subject_found:
        return "UNRELATED"
    
    # Verify it's the SAME entity (not just a similar name)
    # For short names, require near-exact match
    if len(subject.split()) == 1 and len(subject) <= 8:
        # Short name like "Pallavi" - check if it's exact in the sentence
        if subject not in sentence_norm:
            return "UNRELATED"
    
    # Check if role is present
    if not _role_present(role, sentence_norm):
        return "UNRELATED"
    
    # Check if target is present
    if not _target_present(target, sentence_norm):
        return "UNRELATED"
    
    # Extract any person mentioned with the role in this sentence
    alternate_holders = _extract_role_holders(claim, sentence)
    
    # If no alternate holders found, might be support
    if not alternate_holders:
        # Check if the sentence explicitly states the claimed relationship
        if _explicit_support_pattern(claim, sentence):
            # Verify temporal state matches
            if claim_temporal in ["CURRENT", "UNKNOWN"] and sentence_temporal in ["CURRENT", "UNKNOWN"]:
                return "SUPPORT"
            elif claim_temporal == "FORMER" and sentence_temporal == "FORMER":
                return "SUPPORT"
            else:
                # Temporal mismatch - could be current vs former
                return "INSUFFICIENT"
        else:
            # Contains entities but doesn't explicitly state the relationship
            return "INSUFFICIENT"
    
    # If alternate holders found, check if they are different from claimed subject
    for holder in alternate_holders:
        holder_norm = _normalize_text(holder)
        # Use stricter entity matching for disambiguation
        if not _is_same_entity(subject, holder):
            # If the alternate holder is NOT the claimed subject, it's a contradiction
            # Verify temporal state
            if claim_temporal in ["CURRENT", "UNKNOWN"] and sentence_temporal in ["CURRENT", "UNKNOWN"]:
                return "CONTRADICTION"
            elif claim_temporal == "FORMER" and sentence_temporal == "FORMER":
                return "CONTRADICTION"
            else:
                # Temporal mismatch
                return "INSUFFICIENT"
    
    # If we got here, the sentence contains the claimed person and role
    if _explicit_support_pattern(claim, sentence):
        if claim_temporal in ["CURRENT", "UNKNOWN"] and sentence_temporal in ["CURRENT", "UNKNOWN"]:
            return "SUPPORT"
    
    return "INSUFFICIENT"


def _explicit_support_pattern(claim, sentence):
    """
    Detects explicit subject-role-target relationships.
    STRICTER than before - requires explicit connecting verbs.
    """
    subject = _normalize_text(claim["subject"])
    role = _normalize_role(claim["role"])
    target = _normalize_text(claim["target"])
    sentence_norm = _normalize_text(sentence)
    
    if not sentence_norm:
        return False
    
    subject_found = _find_entity_in_sentence(subject, sentence_norm)
    if not subject_found:
        return False
    
    if not _role_present(role, sentence_norm):
        return False
    
    if not _target_present(target, sentence_norm):
        return False
    
    role_regex = {
        "ceo": r"(?:ceo|chief executive officer)",
        "prime minister": r"(?:prime minister|pm)",
        "president": r"president",
        "governor": r"governor",
        "mayor": r"mayor",
        "chancellor": r"chancellor",
    }
    
    role_pattern = role_regex.get(role, re.escape(role))
    subject_pattern = re.escape(subject_found)
    target_pattern = re.escape(target)
    
    patterns = [
        rf"\b{subject_pattern}\b.{{0,50}}?\b(?:is|was|serves as|serving as|remains|became|appointed as|named as)?\s*(?:the\s+)?{role_pattern}\b.{{0,70}}?\b{target_pattern}\b",
        rf"\b{target_pattern}\b.{{0,60}}?(?:'s|is|has)\s*(?:the\s+)?{role_pattern}\b.{{0,70}}?\b{subject_pattern}\b",
        rf"\b{role_pattern}\b.{{0,50}}?\b{subject_pattern}\b.{{0,70}}?\b(?:of|at|for)\s+{target_pattern}\b",
        rf"\b{target_pattern}\b.{{0,30}}?\b{role_pattern}\b.{{0,50}}?\b{subject_pattern}\b",
    ]
    
    for pattern in patterns:
        if re.search(pattern, sentence_norm, flags=re.I):
            return True
    
    return False


def _extract_role_holders(claim, text):
    """
    Finds explicit different persons holding the claimed role for the claimed target.
    STRICT: Only returns persons who are NOT the claimed subject.
    """
    if not text or not claim:
        return []
    
    source = _normalize_text(text)
    target = _normalize_text(claim["target"])
    role = _normalize_role(claim["role"])
    claimed_subject = _normalize_text(claim["subject"])
    
    role_regex = {
        "ceo": r"(?:ceo|chief executive officer)",
        "prime minister": r"(?:prime minister|pm)",
        "president": r"president",
        "governor": r"governor",
        "mayor": r"mayor",
        "chancellor": r"chancellor",
    }
    
    role_pattern = role_regex.get(role, re.escape(role))
    target_pattern = re.escape(target)
    candidates = []
    
    patterns = [
        rf"\b([a-z][a-z .'-]{{2,60}})\s+is\s+(?:the\s+)?{role_pattern}\s+(?:of\s+)?{target_pattern}\b",
        rf"\b([a-z][a-z .'-]{{2,60}}),?\s+(?:the\s+)?{target_pattern}(?:'s)?\s+{role_pattern}\b",
        rf"\b{target_pattern}\b(?:'s)?\s+{role_pattern}\s+([a-z][a-z .'-]{{2,60}})",
        rf"\b{role_pattern}\s+([a-z][a-z .'-]{{2,60}})\s+(?:of|at|for)\s+{target_pattern}\b",
    ]
    
    for pattern in patterns:
        try:
            matches = re.finditer(pattern, source, flags=re.I)
        except re.error:
            continue
        for match in matches:
            raw_person = match.group(1) or ""
            person = re.sub(r"\s+", " ", raw_person).strip(" ,.-")
            if not person:
                continue
            if len(person.split()) > 8:
                continue
            # STRICT: Exclude the claimed subject
            if _similarity(person, claimed_subject) >= 0.92:
                continue
            if _entity_match(claimed_subject, person, threshold=0.92):
                continue
            candidates.append(person)
    
    # Remove duplicates
    unique = []
    for candidate in candidates:
        candidate_norm = _normalize_text(candidate)
        already = False
        for existing in unique:
            if _similarity(candidate_norm, existing) >= 0.95:
                already = True
                break
        if not already:
            unique.append(candidate)
    
    return unique


def _relevant_sentences(claim, article_text, max_sentences=8):
    if not article_text:
        return []
    sentences = _split_sentences(article_text)
    scored = []
    
    for sentence in sentences:
        score = 0
        
        # CRITICAL: First check contextual coherence
        # Only consider sentences that contain all relevant elements
        if not _check_contextual_coherence(claim, sentence):
            continue
        
        if claim["type"] == "role_claim":
            # Use stricter threshold for short names
            subject_threshold = 0.98 if len(claim["subject"].split()) == 1 and len(claim["subject"]) <= 8 else 0.92
            if _entity_match(claim["subject"], sentence, threshold=subject_threshold):
                score += 5
            if _role_present(claim["role"], sentence):
                score += 4
            if _target_present(claim["target"], sentence):
                score += 4
            if _explicit_support_pattern(claim, sentence):
                score += 8
        elif claim["type"] == "event_claim":
            event_threshold = 0.98 if len(claim["event"].split()) == 1 and len(claim["event"]) <= 8 else 0.92
            if _entity_match(claim["event"], sentence, threshold=event_threshold):
                score += 5
            if claim.get("context") and _entity_match(claim["context"], sentence, threshold=0.90):
                score += 4
        elif claim["type"] == "numeric_claim":
            if claim.get("value") and claim["value"] in sentence:
                score += 5
            if claim.get("context") and _entity_match(claim["context"], sentence, threshold=0.90):
                score += 4
        
        # Higher threshold to reduce false positives
        if score >= 7:
            scored.append((score, sentence))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [sentence for _, sentence in scored[:max_sentences]]


# ============================================================
# SOURCE EVIDENCE ANALYSIS
# ============================================================

def _analyze_source_evidence(claim, result):
    """
    Analyzes a single source and classifies its relationship to the claim.
    Returns: SUPPORT, CONTRADICTION, UNRELATED, or INSUFFICIENT
    """
    if not claim:
        return "INSUFFICIENT"
    
    text = (
        result.get("article_text", "")
        or result.get("article_description", "")
        or result.get("article_title", "")
        or result.get("description", "")
        or result.get("title", "")
    )
    
    if not text:
        return "INSUFFICIENT"
    
    # Get relevant sentences
    relevant = _relevant_sentences(claim, text, max_sentences=10)
    if not relevant:
        return "UNRELATED"
    
    # Classify each relevant sentence
    classifications = []
    for sentence in relevant:
        if claim["type"] == "role_claim":
            classification = _classify_evidence_role_claim(claim, sentence)
        else:
            # For other claim types, use simpler classification
            classification = _classify_evidence_generic(claim, sentence)
        classifications.append(classification)
    
    # Aggregate classifications
    if not classifications:
        return "INSUFFICIENT"
    
    # If any sentence is SUPPORT, and none are CONTRADICTION
    if "SUPPORT" in classifications and "CONTRADICTION" not in classifications:
        return "SUPPORT"
    
    # If any sentence is CONTRADICTION, and none are SUPPORT
    if "CONTRADICTION" in classifications and "SUPPORT" not in classifications:
        return "CONTRADICTION"
    
    # If both SUPPORT and CONTRADICTION present (conflicting evidence in same source)
    if "SUPPORT" in classifications and "CONTRADICTION" in classifications:
        return "INSUFFICIENT"
    
    # If majority are UNRELATED
    if classifications.count("UNRELATED") > len(classifications) / 2:
        return "UNRELATED"
    
    # Default to INSUFFICIENT
    return "INSUFFICIENT"


def _classify_evidence_generic(claim, sentence):
    """
    Generic evidence classification for non-role claims.
    """
    sentence_norm = _normalize_text(sentence)
    
    if claim["type"] == "event_claim":
        event = _normalize_text(claim["event"])
        if _entity_match(event, sentence_norm, threshold=0.92):
            # Check if the action matches
            action = claim.get("action", "")
            if action and action in sentence_norm:
                return "SUPPORT"
            return "INSUFFICIENT"
        return "UNRELATED"
    
    elif claim["type"] == "numeric_claim":
        value = claim.get("value", "")
        if value and value in sentence_norm:
            return "SUPPORT"
        return "UNRELATED"
    
    return "INSUFFICIENT"


# ============================================================
# GOOGLE NEWS RSS SEARCH
# ============================================================

def _search_news(query, max_results=8):
    if not query:
        return []
    
    encoded = urllib.parse.quote_plus(query)
    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={encoded}"
        "&hl=en-IN"
        "&gl=IN"
        "&ceid=IN:en"
    )
    
    request = urllib.request.Request(
        rss_url,
        headers={"User-Agent": USER_AGENT},
    )
    
    with urllib.request.urlopen(request, timeout=RSS_TIMEOUT) as response:
        data = response.read()
    
    root = ET.fromstring(data)
    results = []
    
    for item in root.findall(".//item")[:max_results]:
        title_el = item.find("title")
        link_el = item.find("link")
        desc_el = item.find("description")
        date_el = item.find("pubDate")
        source_el = item.find("source")
        
        title = _display_text(title_el.text) if (title_el is not None and title_el.text) else ""
        link = link_el.text.strip() if (link_el is not None and link_el.text) else ""
        description = _display_text(desc_el.text) if (desc_el is not None and desc_el.text) else ""
        published = date_el.text.strip() if (date_el is not None and date_el.text) else ""
        rss_source = _display_text(source_el.text) if (source_el is not None and source_el.text) else ""
        source_url = source_el.attrib.get("url", "") if source_el is not None else ""
        
        publisher = rss_source or _publisher_from_title(title)
        if publisher:
            source_name = _clean_publisher_name(publisher)
            trusted = _is_trusted_publisher(source_name)
            if source_url:
                domain_name, domain_trusted = _source_name(source_url)
                if domain_trusted:
                    source_name = domain_name
                    trusted = True
        else:
            source_name, trusted = _source_name(link)
        
        results.append({
            "title": title,
            "description": description,
            "link": link,
            "source": source_name,
            "trusted": trusted,
            "published": published,
            "rss_source_url": source_url,
            "article_fetched": False,
            "article_url": "",
            "article_title": "",
            "article_description": "",
            "article_text": "",
            "article_error": "",
            "evidence_type": "",
            "evidence_class": "",
            "evidence_sentences": [],
        })
    
    return results


# ============================================================
# FETCH PUBLISHER PAGES
# ============================================================

def _attach_article_evidence(results):
    enriched = []
    for result in results:
        item = dict(result)
        article = _scrape_html(result.get("link", ""))
        item["article_fetched"] = article["success"]
        item["article_url"] = article["final_url"]
        item["article_title"] = article["title"]
        item["article_description"] = article["description"]
        item["article_text"] = article["text"]
        item["article_error"] = article["error"]
        
        if article["success"]:
            item["evidence_type"] = "publisher_article"
        else:
            item["article_title"] = item.get("title", "")
            item["article_description"] = item.get("description", "")
            item["article_text"] = ""
            item["evidence_type"] = "google_news_rss_title_description"
        
        enriched.append(item)
    return enriched


# ============================================================
# SEARCH QUERY GENERATION
# ============================================================

def _build_queries(text, claim):
    general = _clean_query(text)
    if not claim:
        return [general]
    
    if claim["type"] == "role_claim":
        subject = claim["subject"]
        role = claim["role"]
        target = claim["target"]
        
        # For short names, prioritize exact match queries
        if len(subject.split()) == 1 and len(subject) <= 8:
            # Short name like "Pallavi" - use exact match only
            exact = f'"{subject}" "{role}" "{target}"'
            target_role = f'"{target}" "{role}"'
            return [exact, target_role]
        else:
            # Multi-word name - use multiple query strategies
            exact = f'"{subject}" "{role}" "{target}"'
            normal = f"{subject} {role} {target}"
            target_role = f'"{target}" "{role}"'
            return [exact, normal, target_role]
    
    elif claim["type"] == "event_claim":
        event = claim["event"]
        context = claim.get("context", "")
        if context:
            return [f'"{event}" "{context}"', f"{event} {context}"]
        return [event]
    
    elif claim["type"] == "numeric_claim":
        context = claim.get("context", "")
        value = claim.get("value", "")
        if context and value:
            return [f'"{context}" "{value}"', f"{context} {value}"]
        if context:
            return [context]
        return [value]
    
    return [general]


# ============================================================
# RESULT MERGING
# ============================================================

def _merge_results(result_lists):
    unique = []
    seen = set()
    for results in result_lists:
        for result in results:
            key = (result.get("link", "") or result.get("title", "")).strip()
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            unique.append(result)
    return unique


# ============================================================
# EVIDENCE AGGREGATION
# ============================================================

def _extract_current_role_holders(claim, text):
    """
    Extract explicit current role holders for ROLE + TARGET evidence.

    Unlike claim-level evidence, this intentionally does NOT require the
    claimed subject to appear in the same sentence.  A source such as
    "Google CEO Sundar Pichai" is useful evidence for determining who
    currently holds Google's CEO role.
    """
    if not claim or claim.get("type") != "role_claim" or not text:
        return []

    target = _normalize_text(claim.get("target", ""))
    role = _normalize_role(claim.get("role", ""))
    if not target or not role:
        return []

    role_pattern = {
        "ceo": r"(?:ceo|chief executive officer)",
        "prime minister": r"(?:prime minister|pm)",
        "president": r"president",
        "governor": r"governor",
        "mayor": r"mayor",
        "chancellor": r"chancellor",
    }.get(role, re.escape(role))

    target_pattern = re.escape(target)
    clean = _display_text(text)
    candidates = []

    # Keep the patterns deliberately narrow so article sentences do not
    # produce arbitrary nearby people as supposed role holders.
    patterns = [
        rf"\b([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){{0,4}})\s+(?:is|remains|serves as|serving as)\s+(?:the\s+)?{role_pattern}\s+(?:of\s+)?{target_pattern}\b",
        rf"\b{target_pattern}(?:'s)?\s+{role_pattern}\s+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){{0,4}})\b",
        rf"\b([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){{0,4}}),?\s+(?:the\s+)?{role_pattern}\s+(?:of\s+)?{target_pattern}\b",
        rf"\b{role_pattern}\s+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){{0,4}})\s+(?:of|at|for)\s+{target_pattern}\b",
    ]

    # Also inspect normalized text for lowercase RSS titles, e.g.
    # "Google CEO Sundar Pichai..." where capitalization may be lost.
    normalized = _normalize_text(clean)
    normalized_patterns = [
        rf"\b{target_pattern}\b(?:'s)?\s+{role_pattern}\s+([a-z][a-z.'-]+(?:\s+[a-z][a-z.'-]+){{0,4}})\b",
        rf"\b([a-z][a-z.'-]+(?:\s+[a-z][a-z.'-]+){{0,4}})\s+(?:is|remains|serves as|serving as)\s+(?:the\s+)?{role_pattern}\s+(?:of\s+)?{target_pattern}\b",
    ]

    def add_candidate(raw):
        person = re.sub(r"\s+", " ", raw or "").strip(" ,.-")
        if not person:
            return
        # Remove common leading articles/pronouns accidentally captured.
        person = re.sub(r"^(the|a|an|current|former|new)\s+", "", person, flags=re.I)
        words = person.split()
        if not (1 <= len(words) <= 5):
            return
        # A one-word generic role token is not a person name.
        if len(words) == 1 and _normalize_text(person) in {
            "google", "microsoft", "india", "america", "company", "organization"
        }:
            return
        if person.lower() not in {x.lower() for x in candidates}:
            candidates.append(person)

    for pattern in patterns:
        for match in re.finditer(pattern, clean, flags=re.I):
            add_candidate(match.group(1))

    for pattern in normalized_patterns:
        for match in re.finditer(pattern, normalized, flags=re.I):
            add_candidate(match.group(1))

    return candidates


def _holder_matches_subject(subject, holder):
    """Return True when a role-holder reference is the claimed person."""
    if _is_same_entity(subject, holder):
        return True

    subject_norm = _normalize_text(subject)
    holder_norm = _normalize_text(holder)
    subject_words = subject_norm.split()
    holder_words = holder_norm.split()

    # News headlines often use a person's surname alone, e.g. "PM Modi".
    if len(subject_words) >= 2 and len(holder_words) == 1:
        return holder_norm == subject_words[-1]

    # Likewise, a source may use the full name while the claim uses a
    # single distinctive surname.
    if len(subject_words) == 1 and len(holder_words) >= 2:
        return holder_norm.split()[-1] == subject_norm

    return False


def _assess_role_holder_claim(claim, results):
    """
    Decisive verification for PERSON + ROLE + ORGANIZATION claims.

    The current role holder is established from trusted sources that
    explicitly connect a person with the requested role and target.
    This is separate from article-level claim matching because the
    alternate holder does not need to mention the submitted person's name.
    """
    if not claim or claim.get("type") != "role_claim":
        return None

    subject = claim.get("subject", "")
    role = _normalize_role(claim.get("role", ""))
    target = claim.get("target", "")

    holder_sources = []
    support_sources = []
    seen = set()

    for result in results:
        if not result.get("trusted", False):
            continue

        text_parts = [
            result.get("title", ""),
            result.get("description", ""),
            result.get("article_title", ""),
            result.get("article_description", ""),
            result.get("article_text", ""),
        ]
        combined = " ".join(part for part in text_parts if part)
        if not combined:
            continue

        holders = _extract_current_role_holders(claim, combined)
        for holder in holders:
            key = (_normalize_text(holder), result.get("link", ""))
            if key in seen:
                continue
            seen.add(key)

            item = dict(result)
            item["role_holder"] = holder
            item["role"] = role
            item["target"] = target
            item["evidence_class"] = "ROLE_HOLDER"
            holder_sources.append(item)

            if _holder_matches_subject(subject, holder):
                support_sources.append(item)

    # Prefer direct support for the claimed subject if available.
    if support_sources:
        return {
            "status": "REAL",
            "label": "🟢 REAL / VERIFIED",
            "reason": f"Trusted current-source evidence identifies {subject} as the {role} of {target}.",
            "supporting_sources": support_sources,
            "contradicting_sources": [],
            "unrelated_sources": [],
            "insufficient_sources": [],
        }

    # If trusted sources consistently identify a different person, the
    # submitted role claim is contradicted.
    alternate_sources = [
        item for item in holder_sources
        if not _holder_matches_subject(subject, item.get("role_holder", ""))
    ]

    if alternate_sources:
        # Deduplicate alternate-holder evidence by source and holder.
        unique = []
        seen_alt = set()
        for item in alternate_sources:
            key = (
                _normalize_text(item.get("role_holder", "")),
                item.get("link", ""),
            )
            if key not in seen_alt:
                seen_alt.add(key)
                unique.append(item)

        holders_text = []
        for item in unique:
            holder = item.get("role_holder", "")
            if holder and holder not in holders_text:
                holders_text.append(holder)

        holder_text = ", ".join(holders_text[:3])
        return {
            "status": "FAKE",
            "label": "🔴 FAKE / CONTRADICTED",
            "reason": (
                f"Trusted current-source evidence identifies {holder_text} "
                f"as the {role} of {target}, not {subject}."
            ),
            "supporting_sources": [],
            "contradicting_sources": unique,
            "unrelated_sources": [],
            "insufficient_sources": [],
        }

    # No explicit current holder was established.
    return None


def _assess_claim(claim, results):
    if not claim:
        return {
            "status": "UNVERIFIED",
            "label": "🟡 UNVERIFIED",
            "reason": "The submitted text does not contain a supported factual relationship structure for claim-level verification.",
            "supporting_sources": [],
            "contradicting_sources": [],
            "unrelated_sources": [],
            "insufficient_sources": [],
        }

    # --------------------------------------------------------
    # ROLE CLAIMS GET A SEPARATE CURRENT-HOLDER CHECK.
    # Example:
    #   "Pallavi is CEO of Google"
    # can be contradicted by a trusted source saying
    #   "Google CEO Sundar Pichai"
    # even when that source never mentions Pallavi.
    # --------------------------------------------------------
    if claim.get("type") == "role_claim":
        role_assessment = _assess_role_holder_claim(claim, results)
        if role_assessment is not None:
            return role_assessment

    # --------------------------------------------------------
    # GENERAL CLAIM-LEVEL EVIDENCE
    # --------------------------------------------------------
    filtered_results = []
    for result in results:
        combined_text = " ".join([
            result.get("title", ""),
            result.get("description", ""),
            result.get("article_title", ""),
            result.get("article_description", ""),
            result.get("article_text", ""),
        ])

        if claim.get("type") == "role_claim":
            subject_threshold = (
                0.98
                if len(claim["subject"].split()) == 1
                and len(claim["subject"]) <= 8
                else 0.92
            )

            has_subject = _entity_match(
                claim["subject"], combined_text, threshold=subject_threshold
            )
            has_role = _role_present(claim["role"], combined_text)
            has_target = _target_present(claim["target"], combined_text)

            if has_subject and has_role and has_target:
                filtered_results.append(result)
            else:
                item = dict(result)
                item["evidence_class"] = "UNRELATED"
                filtered_results.append(item)
        else:
            filtered_results.append(result)

    supporting = []
    contradicting = []
    unrelated = []
    insufficient = []
    used_links = set()

    for result in filtered_results:
        if not result.get("trusted", False):
            continue

        link = result.get("link", "")
        if link in used_links:
            continue

        if result.get("evidence_class") == "UNRELATED":
            unrelated.append(result)
            used_links.add(link)
            continue

        classification = _analyze_source_evidence(claim, result)
        item = dict(result)
        item["evidence_class"] = classification

        if classification == "SUPPORT":
            supporting.append(item)
            used_links.add(link)
        elif classification == "CONTRADICTION":
            contradicting.append(item)
            used_links.add(link)
        elif classification == "UNRELATED":
            unrelated.append(item)
        else:
            insufficient.append(item)

    article_support = [
        s for s in supporting
        if s.get("evidence_type") == "publisher_article"
    ]
    rss_support = [
        s for s in supporting
        if s.get("evidence_type") != "publisher_article"
    ]
    article_contradictions = [
        c for c in contradicting
        if c.get("evidence_type") == "publisher_article"
    ]
    rss_contradictions = [
        c for c in contradicting
        if c.get("evidence_type") != "publisher_article"
    ]

    if article_support:
        return {
            "status": "REAL",
            "label": "🟢 REAL / VERIFIED",
            "reason": "A trusted publisher page was retrieved and contains explicit claim-level evidence supporting the claim.",
            "supporting_sources": article_support,
            "contradicting_sources": article_contradictions,
            "unrelated_sources": unrelated,
            "insufficient_sources": insufficient,
        }

    if article_contradictions:
        return {
            "status": "FAKE",
            "label": "🔴 FAKE / CONTRADICTED",
            "reason": "A trusted publisher page explicitly contradicts the claim with direct incompatible evidence.",
            "supporting_sources": [],
            "contradicting_sources": article_contradictions,
            "unrelated_sources": unrelated,
            "insufficient_sources": insufficient,
        }

    if rss_support:
        return {
            "status": "REAL",
            "label": "🟢 REAL / VERIFIED",
            "reason": "A trusted current source explicitly states the claimed relationship in its title or description. This is secondary evidence as the publisher page could not be retrieved.",
            "supporting_sources": rss_support,
            "contradicting_sources": rss_contradictions,
            "unrelated_sources": unrelated,
            "insufficient_sources": insufficient,
        }

    if rss_contradictions:
        return {
            "status": "FAKE",
            "label": "🔴 FAKE / CONTRADICTED",
            "reason": "A trusted current source explicitly contradicts the claim in its title or description.",
            "supporting_sources": [],
            "contradicting_sources": rss_contradictions,
            "unrelated_sources": unrelated,
            "insufficient_sources": insufficient,
        }

    if unrelated or insufficient:
        return {
            "status": "UNVERIFIED",
            "label": "🟡 UNVERIFIED",
            "reason": "Current trusted sources were found, but the retrieved evidence does not explicitly establish that the claim is true or false. Related information is not treated as proof.",
            "supporting_sources": [],
            "contradicting_sources": [],
            "unrelated_sources": unrelated,
            "insufficient_sources": insufficient,
        }

    return {
        "status": "UNVERIFIED",
        "label": "🟡 UNVERIFIED",
        "reason": "Current trusted sources were found, but the retrieved evidence is insufficient to establish the claim as true or false.",
        "supporting_sources": [],
        "contradicting_sources": [],
        "unrelated_sources": [],
        "insufficient_sources": insufficient,
    }


# ============================================================
# MAIN PUBLIC FUNCTION
# ============================================================

def verify_news(text, max_results=8):
    """
    Main entry point used by app.py.
    
    Returns:
        status
        query
        results
        trusted_count
        article_fetched_count
        claim
        assessment
        message
    
    ML prediction is intentionally NOT accepted as factual truth here.
    """
    if not text or not text.strip():
        return {
            "status": "not_found",
            "query": "",
            "results": [],
            "trusted_count": 0,
            "article_fetched_count": 0,
            "claim": None,
            "assessment": {
                "status": "UNVERIFIED",
                "label": "🟡 UNVERIFIED",
                "reason": "No text was provided.",
                "supporting_sources": [],
                "contradicting_sources": [],
                "unrelated_sources": [],
                "insufficient_sources": [],
            },
            "message": "Please enter a news claim.",
        }
    
    claim = _extract_claim(text)
    general_query = _clean_query(text)
    queries = _build_queries(text, claim)
    
    # SEARCH
    result_lists = []
    try:
        for query in queries:
            if not query:
                continue
            try:
                results = _search_news(query, max_results)
                result_lists.append(results)
            except Exception:
                continue
    except Exception as exc:
        return {
            "status": "error",
            "query": general_query,
            "results": [],
            "trusted_count": 0,
            "article_fetched_count": 0,
            "claim": claim,
            "assessment": {
                "status": "UNVERIFIED",
                "label": "🟡 UNVERIFIED",
                "reason": f"Real-time source verification could not be completed: {str(exc)}",
                "supporting_sources": [],
                "contradicting_sources": [],
                "unrelated_sources": [],
                "insufficient_sources": [],
            },
            "message": "Real-time source verification is currently unavailable.",
        }
    
    all_results = _merge_results(result_lists)
    
    if not all_results:
        return {
            "status": "not_found",
            "query": general_query,
            "results": [],
            "trusted_count": 0,
            "article_fetched_count": 0,
            "claim": claim,
            "assessment": {
                "status": "UNVERIFIED",
                "label": "🟡 UNVERIFIED",
                "reason": "No current related sources were found. Absence of search results is not treated as proof that the claim is false.",
                "supporting_sources": [],
                "contradicting_sources": [],
                "unrelated_sources": [],
                "insufficient_sources": [],
            },
            "message": "No related current news reports were found.",
        }
    
    # Only trusted sources are fetched
    trusted_results = [result for result in all_results if result.get("trusted", False)]
    trusted_to_fetch = trusted_results[:max_results]
    
    enriched = _attach_article_evidence(trusted_to_fetch)
    enriched_by_link = {item.get("link", ""): item for item in enriched}
    
    final_results = []
    for result in all_results:
        link = result.get("link", "")
        if link in enriched_by_link:
            final_results.append(enriched_by_link[link])
        else:
            final_results.append(result)
    
    all_results = final_results
    
    # Assess evidence
    assessment = _assess_claim(claim, all_results)
    
    trusted_count = sum(1 for result in all_results if result.get("trusted", False))
    article_fetched_count = sum(1 for result in all_results if result.get("article_fetched", False))
    
    return {
        "status": "found",
        "query": general_query,
        "results": all_results,
        "trusted_count": trusted_count,
        "article_fetched_count": article_fetched_count,
        "claim": claim,
        "assessment": assessment,
        "message": "Verification completed using real-time source discovery, trusted-source filtering and publisher-page evidence analysis.",
    }
