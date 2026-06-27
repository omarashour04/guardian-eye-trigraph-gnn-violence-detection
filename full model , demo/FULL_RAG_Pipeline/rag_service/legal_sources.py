"""Legal source registry for the Legal Consequences RAG store.

This module only records source metadata. It intentionally does not scrape,
index, retrieve, summarize, or call any model.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


LEGAL_SOURCE_REGISTRY: dict[str, list[dict[str, Any]]] = {
    "UK": [
        {
            "country": "UK",
            "source_language": "en",
            "law_title": "Offences Against the Person Act 1861",
            "source_url": "https://www.legislation.gov.uk/ukpga/Vict/24-25/100/contents",
            "official_source": True,
            "source_type": "html",
            "notes": "UK official legislation portal.",
        },
        {
            "country": "UK",
            "source_language": "en",
            "law_title": "Crime and Disorder Act 1998",
            "source_url": "https://www.legislation.gov.uk/ukpga/1998/37/contents",
            "official_source": True,
            "source_type": "html",
            "notes": "UK official legislation portal.",
        },
        {
            "country": "UK",
            "source_language": "en",
            "law_title": "Public Order Act 1986",
            "source_url": "https://www.legislation.gov.uk/ukpga/1986/64",
            "official_source": True,
            "source_type": "html",
            "notes": "UK official legislation portal.",
        },
        {
            "country": "UK",
            "source_language": "en",
            "law_title": "Criminal Justice Act 2003",
            "source_url": "https://www.legislation.gov.uk/ukpga/2003/44/contents",
            "official_source": True,
            "source_type": "html",
            "notes": "Optional UK official legislation source for sentencing context.",
        },
    ],
    "USA California": [
        {
            "country": "USA California",
            "source_language": "en",
            "law_title": "California Penal Code Section 240",
            "source_url": "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=PEN&sectionNum=240",
            "official_source": True,
            "source_type": "html",
            "notes": "California Legislative Information official code section.",
        }
    ],
    "Canada": [
        {
            "country": "Canada",
            "source_language": "en",
            "law_title": "Criminal Code Section 265",
            "source_url": "https://laws-lois.justice.gc.ca/eng/acts/c-46/section-265.html",
            "official_source": True,
            "source_type": "html",
            "notes": "Justice Laws Website official source.",
        },
        {
            "country": "Canada",
            "source_language": "en",
            "law_title": "Criminal Code Section 267",
            "source_url": "https://laws-lois.justice.gc.ca/eng/acts/c-46/section-267.html",
            "official_source": True,
            "source_type": "html",
            "notes": "Justice Laws Website official source.",
        },
        {
            "country": "Canada",
            "source_language": "en",
            "law_title": "Criminal Code Page 38",
            "source_url": "https://laws-lois.justice.gc.ca/eng/acts/c-46/page-38.html",
            "official_source": True,
            "source_type": "html",
            "notes": "Justice Laws Website official source containing nearby Criminal Code provisions.",
        },
    ],
    "KSA": [
        {
            "country": "KSA",
            "source_language": "ar",
            "law_title": "Protection from Abuse Law",
            "source_url": "https://www.mof.gov.sa/docslibrary/RegulationsInstructions/DocLib/%D9%86%D8%B8%D8%A7%D9%85%20%D8%A7%D9%84%D8%AD%D9%85%D8%A7%D9%8A%D8%A9%20%D9%85%D9%86%20%D8%A7%D9%84%D8%A5%D9%8A%D8%B0%D8%A7%D8%A1.pdf",
            "official_source": True,
            "source_type": "pdf",
            "notes": "Arabic PDF hosted by Saudi Ministry of Finance.",
        },
        {
            "country": "KSA",
            "source_language": "ar",
            "law_title": "Protection from Abuse Law",
            "source_url": "https://www.hrsd.gov.sa/sites/default/files/2015-07/07102022.pdf",
            "official_source": True,
            "source_type": "pdf",
            "notes": "Arabic PDF hosted by Saudi Ministry of Human Resources and Social Development.",
        },
        {
            "country": "KSA",
            "source_language": "ar",
            "law_title": "Executive Regulations of the Protection from Abuse Law",
            "source_url": "https://nshr.org.sa/wp-content/uploads/2013/10/%D8%A7%D9%84%D9%84%D8%A7%D8%A6%D8%AD%D8%A9-%D8%A7%D9%84%D8%AA%D9%86%D9%81%D9%8A%D8%B0%D9%8A%D8%A9-%D9%84%D9%86%D8%B8%D8%A7%D9%85-%D8%A7%D9%84%D8%AD%D9%85%D8%A7%D9%8A%D8%A9-%D9%85%D9%86-%D8%A7%D9%84%D8%A5%D9%8A%D8%B0%D8%A7%D8%A1-1.pdf",
            "official_source": False,
            "source_type": "pdf",
            "notes": "Arabic PDF hosted by the National Society for Human Rights.",
        },
    ],
    "UAE": [
        {
            "country": "UAE",
            "source_language": "ar",
            "law_title": "UAE Crimes and Penalties Law",
            "source_url": "https://uaelegislation.gov.ae/ar/legislations/1529",
            "official_source": True,
            "source_type": "html",
            "notes": "UAE official legislation portal.",
        },
        {
            "country": "UAE",
            "source_language": "ar",
            "law_title": "UAE Crimes and Penalties Law PDF",
            "source_url": "https://www.moj.gov.ae/assets/89ba4caf/%D9%82%D8%A7%D9%86%D9%88%D9%86-%D8%A7%D9%84%D8%AC%D8%B1%D8%A7%D8%A6%D9%85-%D9%88%D8%A7%D9%84%D8%B9%D9%82%D9%88%D8%A8%D8%A7%D8%AA-638336735665323881.aspx",
            "official_source": True,
            "source_type": "pdf",
            "notes": "Arabic source hosted by UAE Ministry of Justice.",
        },
        {
            "country": "UAE",
            "source_language": "ar",
            "law_title": "UAE Public Legal Content Asset",
            "source_url": "https://assets.u.ae/api/public/content/7c977074c2684b66b30bfb5817e3eaf2?v=061ee26e",
            "official_source": True,
            "source_type": "pdf",
            "notes": "Public UAE government content asset.",
        },
    ],
    "Egypt": [
        {
            "country": "Egypt",
            "source_language": "ar",
            "law_title": "Egyptian Penal Code Law No. 58 of 1937",
            "source_url": "https://www.eastlaws.com/legislation-full-text/ar/egypt/law/05-08-1937/no-58?type=1&id=209",
            "official_source": False,
            "source_type": "html",
            "notes": "Arabic legal reference page for Egypt Penal Code.",
        }
    ],
}


def get_supported_countries() -> list[str]:
    """Return countries configured for the Legal Consequences RAG store."""

    return list(LEGAL_SOURCE_REGISTRY.keys())


def get_sources_for_country(country: str) -> list[dict[str, Any]]:
    """Return a defensive copy of configured sources for a country."""

    return deepcopy(LEGAL_SOURCE_REGISTRY.get(country, []))


def get_all_legal_sources() -> dict[str, list[dict[str, Any]]]:
    """Return a defensive copy of the complete legal source registry."""

    return deepcopy(LEGAL_SOURCE_REGISTRY)
