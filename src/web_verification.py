# ============================================================
# WEB VERIFICATION MODULE
# Fake News Detection System
# ============================================================

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from nltk.corpus import stopwords


# ============================================================
# TRUSTED NEWS SOURCES
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
}


# ============================================================
# CREATE SEARCH QUERY
# ============================================================

def _clean_query(text, max_words=12):

    try:
        stop_words = set(
            stopwords.words("english")
        )
    except Exception:
        stop_words = set()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        "",
        text
    )

    # Extract words
    words = re.findall(
        r"[A-Za-z]{3,}",
        text.lower()
    )

    # Remove stopwords
    words = [
        word
        for word in words
        if word not in stop_words
    ]

    # Remove duplicate words
    unique_words = []

    for word in words:
        if word not in unique_words:
            unique_words.append(word)

    return " ".join(
        unique_words[:max_words]
    )


# ============================================================
# IDENTIFY SOURCE FROM URL
# ============================================================

def _source_name(url):

    try:

        parsed = urllib.parse.urlparse(url)

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
# EXTRACT PUBLISHER FROM TITLE
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
# CHECK TRUSTED PUBLISHER
# ============================================================

def _is_trusted_publisher(publisher):

    if not publisher:
        return False

    publisher_lower = (
        publisher
        .strip()
        .lower()
    )

    # Normalize punctuation
    publisher_lower = re.sub(
        r"[^a-z0-9\s]",
        " ",
        publisher_lower
    )

    # Normalize spaces
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
# REAL-TIME NEWS VERIFICATION
# ============================================================

def verify_news(text, max_results=5):

    """
    Search current news reports related to the
    submitted article.

    This module provides supporting source evidence.

    It does NOT automatically determine whether
    an article is TRUE or FALSE.
    """

    # --------------------------------------------------------
    # CREATE SEARCH QUERY
    # --------------------------------------------------------

    query = _clean_query(text)

    if not query:

        return {
            "status": "not_found",
            "query": "",
            "results": [],
            "trusted_count": 0,
            "message":
                "Not enough meaningful text "
                "to search."
        }

    # --------------------------------------------------------
    # GOOGLE NEWS RSS URL
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # REQUEST NEWS
    # --------------------------------------------------------

    try:

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

        # ----------------------------------------------------
        # PARSE XML
        # ----------------------------------------------------

        root = ET.fromstring(data)

        results = []

        trusted_count = 0

        # ----------------------------------------------------
        # PROCESS RESULTS
        # ----------------------------------------------------

        for item in root.findall(
            ".//item"
        )[:max_results]:

            # ------------------------------------------------
            # TITLE
            # ------------------------------------------------

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

            # ------------------------------------------------
            # LINK
            # ------------------------------------------------

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

            # ------------------------------------------------
            # PUBLICATION DATE
            # ------------------------------------------------

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

            # ------------------------------------------------
            # DETERMINE PUBLISHER
            # ------------------------------------------------

            publisher = _extract_publisher(
                title
            )

            if publisher:

                source_name = (
                    _clean_publisher_name(
                        publisher
                    )
                )

                is_trusted = (
                    _is_trusted_publisher(
                        source_name
                    )
                )

            else:

                source_name, is_trusted = (
                    _source_name(link)
                )

            # ------------------------------------------------
            # TRUSTED COUNT
            # ------------------------------------------------

            if is_trusted:
                trusted_count += 1

            # ------------------------------------------------
            # STORE RESULT
            # ------------------------------------------------

            results.append({
                "title": title,
                "link": link,
                "source": source_name,
                "trusted": is_trusted,
                "published": published
            })

        # ----------------------------------------------------
        # NO RESULTS
        # ----------------------------------------------------

        if not results:

            return {
                "status": "not_found",
                "query": query,
                "results": [],
                "trusted_count": 0,
                "message":
                    "No related news reports "
                    "were found."
            }

        # ----------------------------------------------------
        # RESULTS FOUND
        # ----------------------------------------------------

        return {
            "status": "found",
            "query": query,
            "results": results,
            "trusted_count": trusted_count,
            "message":
                "Related news reports were "
                "found. Review these sources "
                "alongside the ML prediction."
        }

    # --------------------------------------------------------
    # ERROR HANDLING
    # --------------------------------------------------------

    except Exception as e:

        return {
            "status": "error",
            "query": query,
            "results": [],
            "trusted_count": 0,
            "message":
                f"Real-time verification "
                f"unavailable: {str(e)}"
        }