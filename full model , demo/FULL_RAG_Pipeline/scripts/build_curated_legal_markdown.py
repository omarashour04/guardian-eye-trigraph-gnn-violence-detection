"""Generate curated RAG-ready legal consequence documents and chunks.

This deliberately avoids the old scrape-and-dump style. Each violent-act
section is self-contained markdown and is also exported as one LegalChunk dict.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "legal_curated_docs"
CHUNKS_PATH = ROOT / "data" / "legal_curated_chunks.json"


@dataclass(frozen=True)
class Act:
    name: str
    category: str
    keywords: str
    legal_meaning: str
    consequence: str
    law: str
    article: str | None
    source: str
    notes: str
    official: bool = True


@dataclass(frozen=True)
class CountryDoc:
    country: str
    filename: str
    acts: tuple[Act, ...]


COMMON_NOTE = "Guardian Eye uses this only for retrieval context. It is not legal advice."


DOCS: tuple[CountryDoc, ...] = (
    CountryDoc(
        country="UK",
        filename="uk_legal_consequences.md",
        acts=(
            Act(
                "Common assault and battery",
                "assault",
                "UK, assault, battery, fight, physical contact, threat, violence",
                "Common assault and battery cover intentional or reckless unlawful force, or causing another person to apprehend immediate unlawful violence.",
                "A verified incident may lead to investigation or prosecution for common assault/battery. The court outcome depends on evidence, harm, aggravating factors, and sentencing rules.",
                "Criminal Justice Act 1988",
                "Section 39",
                "https://www.legislation.gov.uk/ukpga/1988/33/section/39",
                COMMON_NOTE,
            ),
            Act(
                "Assault occasioning actual bodily harm",
                "actual_bodily_harm",
                "UK, assault, actual bodily harm, ABH, injury, violence",
                "ABH is relevant when an assault or battery occasions actual bodily harm rather than only transient contact.",
                "A verified ABH incident may be charged under the Offences Against the Person Act and can carry more serious consequences than common assault.",
                "Offences Against the Person Act 1861",
                "Section 47",
                "https://www.legislation.gov.uk/ukpga/Vict/24-25/100/section/47",
                COMMON_NOTE,
            ),
            Act(
                "Grievous bodily harm or wounding",
                "grievous_bodily_harm",
                "UK, grievous bodily harm, GBH, wounding, serious injury, violence",
                "GBH or unlawful wounding is relevant where violence causes serious bodily injury or a wound.",
                "A verified GBH or wounding incident may lead to a serious charge. Intent, injury severity, and weapon use can materially affect charging and sentencing.",
                "Offences Against the Person Act 1861",
                "Sections 18 and 20",
                "https://www.legislation.gov.uk/ukpga/Vict/24-25/100/contents",
                COMMON_NOTE,
            ),
            Act(
                "Threats to kill",
                "threats",
                "UK, threat, threats to kill, intimidation, fear, violence",
                "Threats to kill are relevant where a person makes a threat that another would reasonably fear may be carried out.",
                "A verified threat-to-kill incident may lead to prosecution under section 16, separate from any physical assault charge.",
                "Offences Against the Person Act 1861",
                "Section 16",
                "https://www.legislation.gov.uk/ukpga/Vict/24-25/100/section/16",
                COMMON_NOTE,
            ),
            Act(
                "Robbery with force or fear",
                "robbery",
                "UK, robbery, force, fear, theft, violence, weapon",
                "Robbery is relevant where theft is accompanied by force, or by putting or seeking to put a person in fear of force.",
                "A verified robbery-with-violence incident may be treated as a serious theft and violence offence. Weapon use or injury can aggravate the case.",
                "Theft Act 1968",
                "Section 8",
                "https://www.legislation.gov.uk/ukpga/1968/60/section/8",
                COMMON_NOTE,
            ),
            Act(
                "Affray and public fighting",
                "public_fighting",
                "UK, affray, public fighting, public order, violence, threat",
                "Affray is relevant where a person uses or threatens unlawful violence and the conduct would cause a person of reasonable firmness to fear for safety.",
                "A verified public fight may be reviewed under public order law, even if no specific victim is identified in the footage.",
                "Public Order Act 1986",
                "Section 3",
                "https://www.legislation.gov.uk/ukpga/1986/64/section/3",
                COMMON_NOTE,
            ),
            Act(
                "Sexual assault",
                "sexual_assault",
                "UK, sexual assault, unwanted sexual touching, violence, consent",
                "Sexual assault is relevant where intentional sexual touching occurs without consent and without reasonable belief in consent.",
                "A verified sexual assault incident may lead to investigation and prosecution under the Sexual Offences Act. Guardian Eye should not infer consent or identity from video alone.",
                "Sexual Offences Act 2003",
                "Section 3",
                "https://www.legislation.gov.uk/ukpga/2003/42/section/3",
                COMMON_NOTE,
            ),
            Act(
                "Offensive weapon in a public place",
                "weapon",
                "UK, weapon, offensive weapon, public place, dangerous object, assault",
                "Weapon context is relevant where an offensive weapon is possessed in a public place without lawful authority or reasonable excuse.",
                "A weapon flag may make a violent incident legally more serious and can support separate weapon-related review.",
                "Prevention of Crime Act 1953",
                "Section 1",
                "https://www.legislation.gov.uk/ukpga/Eliz2/1-2/14/section/1",
                COMMON_NOTE,
            ),
        ),
    ),
    CountryDoc(
        country="Canada",
        filename="canada_legal_consequences.md",
        acts=(
            Act(
                "Assault",
                "assault",
                "Canada, assault, force, threat, fight, violence, no consent",
                "Assault includes intentional application of force without consent, attempts or threats by act or gesture, and accosting or impeding while openly wearing or carrying a weapon.",
                "A verified assault incident may lead to a Criminal Code assault charge and release or court conditions depending on evidence and risk.",
                "Criminal Code, R.S.C., 1985, c. C-46",
                "Sections 265 and 266",
                "https://laws-lois.justice.gc.ca/eng/acts/C-46/section-265.html",
                COMMON_NOTE,
            ),
            Act(
                "Assault with a weapon or causing bodily harm",
                "weapon_assault",
                "Canada, assault with weapon, bodily harm, choking, injury, violence",
                "This act is relevant where assault involves a weapon, imitation weapon, bodily harm, choking, suffocation, or strangling.",
                "A verified weapon or bodily-harm assault may be treated more seriously than simple assault and can affect bail, charging, and sentencing.",
                "Criminal Code, R.S.C., 1985, c. C-46",
                "Section 267",
                "https://laws-lois.justice.gc.ca/eng/acts/C-46/section-267.html",
                COMMON_NOTE,
            ),
            Act(
                "Aggravated assault",
                "grievous_bodily_harm",
                "Canada, aggravated assault, wounds, maims, disfigures, endangers life",
                "Aggravated assault is relevant where violence wounds, maims, disfigures, or endangers the life of the complainant.",
                "A verified aggravated assault incident is a serious Criminal Code matter and should be escalated for legal review.",
                "Criminal Code, R.S.C., 1985, c. C-46",
                "Section 268",
                "https://laws-lois.justice.gc.ca/eng/acts/C-46/section-268.html",
                COMMON_NOTE,
            ),
            Act(
                "Uttering threats",
                "threats",
                "Canada, threats, death threat, bodily harm, intimidation, violence",
                "Uttering threats is relevant where a person knowingly conveys a threat to cause death or bodily harm, or to damage property or harm animals in the statutory categories.",
                "A verified threat may support a Criminal Code threats charge even without completed physical injury.",
                "Criminal Code, R.S.C., 1985, c. C-46",
                "Section 264.1",
                "https://laws-lois.justice.gc.ca/eng/acts/C-46/section-264.1.html",
                COMMON_NOTE,
            ),
            Act(
                "Robbery with violence",
                "robbery",
                "Canada, robbery, violence, threats, theft, weapon, assault",
                "Robbery is relevant where theft is accompanied by violence, threats of violence, assault, or offensive weapon conduct.",
                "A verified robbery-with-violence incident may trigger robbery provisions and can be more serious than theft alone.",
                "Criminal Code, R.S.C., 1985, c. C-46",
                "Sections 343 and 344",
                "https://laws-lois.justice.gc.ca/eng/acts/C-46/section-343.html",
                COMMON_NOTE,
            ),
            Act(
                "Sexual assault",
                "sexual_assault",
                "Canada, sexual assault, sexual violence, consent, force",
                "Sexual assault applies where the assault is committed in circumstances of a sexual nature; consent rules are central and cannot be inferred from video alone.",
                "A verified sexual assault allegation may lead to Criminal Code sexual assault review and protective conditions.",
                "Criminal Code, R.S.C., 1985, c. C-46",
                "Sections 265 and 271",
                "https://laws-lois.justice.gc.ca/eng/acts/C-46/section-271.html",
                COMMON_NOTE,
            ),
            Act(
                "Homicide and murder",
                "homicide",
                "Canada, homicide, murder, death, lethal violence",
                "Homicide provisions are relevant only where violence causes death and the mental element and statutory category are legally assessed.",
                "Guardian Eye must not classify homicide from ordinary violence footage; it can only flag that lethal harm would require homicide-specific legal review.",
                "Criminal Code, R.S.C., 1985, c. C-46",
                "Sections 222, 229, and 235",
                "https://laws-lois.justice.gc.ca/eng/acts/C-46/section-222.html",
                COMMON_NOTE,
            ),
            Act(
                "Domestic or intimate partner violence aggravating context",
                "domestic_violence",
                "Canada, domestic violence, intimate partner, abuse, assault, threat",
                "Canada generally applies existing offences such as assault, threats, sexual assault, and homicide; intimate-partner abuse can be relevant at sentencing.",
                "A verified domestic violence context may affect release conditions, protection planning, and sentencing aggravation where proven.",
                "Criminal Code, R.S.C., 1985, c. C-46",
                "Section 718.2(a)(ii)",
                "https://laws-lois.justice.gc.ca/eng/acts/C-46/section-718.2.html",
                COMMON_NOTE,
            ),
        ),
    ),
    CountryDoc(
        country="USA California",
        filename="usa_legal_consequences.md",
        acts=(
            Act(
                "Assault",
                "assault",
                "USA California, assault, attempt, violent injury, fight, threat",
                "California assault is an unlawful attempt, coupled with present ability, to commit a violent injury on another person.",
                "A verified assault incident may support a Penal Code assault review even if no completed battery is visible.",
                "California Penal Code",
                "Section 240",
                "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=PEN&sectionNum=240",
                COMMON_NOTE,
            ),
            Act(
                "Battery",
                "battery",
                "USA California, battery, force, violence, physical contact",
                "Battery is the willful and unlawful use of force or violence upon another person.",
                "A verified battery incident may support a Penal Code battery review when physical force is completed.",
                "California Penal Code",
                "Section 242",
                "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=PEN&sectionNum=242",
                COMMON_NOTE,
            ),
            Act(
                "Assault with a deadly weapon or force likely to produce great bodily injury",
                "weapon_assault",
                "USA California, deadly weapon, firearm, great bodily injury, assault",
                "This act is relevant when assault involves a deadly weapon, firearm, or force likely to produce great bodily injury.",
                "A weapon or great-bodily-injury context can make the incident substantially more serious than simple assault.",
                "California Penal Code",
                "Section 245",
                "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=PEN&sectionNum=245",
                COMMON_NOTE,
            ),
            Act(
                "Corporal injury in domestic violence context",
                "domestic_violence",
                "USA California, domestic violence, corporal injury, spouse, cohabitant, dating partner",
                "Section 273.5 is relevant where willful corporal injury results in a traumatic condition for a protected relationship category.",
                "A verified domestic violence injury context may support criminal review and protective orders; relationship status and injury must be legally established.",
                "California Penal Code",
                "Section 273.5",
                "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=PEN&sectionNum=273.5",
                COMMON_NOTE,
            ),
            Act(
                "Criminal threats",
                "threats",
                "USA California, criminal threat, threat, fear, bodily injury, death",
                "Criminal threats are relevant where a willful threat to commit a crime resulting in death or great bodily injury creates sustained fear under statutory conditions.",
                "A verified threat may be reviewed separately from assault or battery if the statutory elements are supported.",
                "California Penal Code",
                "Section 422",
                "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=PEN&sectionNum=422",
                COMMON_NOTE,
            ),
            Act(
                "Robbery by force or fear",
                "robbery",
                "USA California, robbery, force, fear, property, violence",
                "Robbery is the felonious taking of personal property from another person or immediate presence, against the person's will, by force or fear.",
                "A verified taking with force or fear may support robbery review rather than theft-only review.",
                "California Penal Code",
                "Section 211",
                "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=PEN&sectionNum=211",
                COMMON_NOTE,
            ),
            Act(
                "Murder or homicide",
                "homicide",
                "USA California, murder, homicide, death, malice, lethal violence",
                "Murder/homicide provisions are relevant only if death and the required legal mental state are established by authorities.",
                "Guardian Eye should never infer homicide from ordinary violence detection; it can flag that lethal harm would require homicide-specific legal review.",
                "California Penal Code",
                "Section 187",
                "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=PEN&sectionNum=187",
                COMMON_NOTE,
            ),
            Act(
                "Sexual assault or rape",
                "sexual_assault",
                "USA California, sexual assault, rape, consent, force, fear",
                "California rape provisions concern acts of sexual intercourse under listed non-consent, force, fear, incapacity, or statutory circumstances.",
                "A verified sexual violence allegation requires specialist legal review; Guardian Eye must not infer consent, identity, or statutory circumstances from video alone.",
                "California Penal Code",
                "Section 261",
                "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=PEN&sectionNum=261",
                COMMON_NOTE,
            ),
        ),
    ),
    CountryDoc(
        country="UAE",
        filename="uae_legal_consequences.md",
        acts=(
            Act(
                "Assault and physical injury",
                "assault",
                "UAE, assault, battery, physical injury, fight, violence",
                "The UAE Crimes and Penalties Law is the verified federal source for crimes and penalties; accessible source review did not provide a stable article-level English citation for each injury grade.",
                "A verified assault or injury incident may lead to criminal investigation under the UAE Crimes and Penalties Law. Exact article and penalty must be checked in the authoritative Arabic text.",
                "Federal Decree-Law No. 31 of 2021 Promulgating the Crimes and Penalties Law",
                None,
                "https://uaelegislation.gov.ae/en/legislations/1529",
                "Official UAE legislation platform. Article-level assault citation unavailable in the local verified extract; do not invent it.",
            ),
            Act(
                "Use of weapon or dangerous object during violence",
                "weapon_assault",
                "UAE, weapon, dangerous object, assault, injury, violence",
                "Weapon context may be relevant to violence, injury, threats, or public safety review under UAE criminal law.",
                "A verified weapon or dangerous-object flag can increase legal seriousness, but the exact charge must come from the official Arabic legal text and competent authority review.",
                "Federal Decree-Law No. 31 of 2021 Promulgating the Crimes and Penalties Law",
                None,
                "https://uaelegislation.gov.ae/en/legislations/1529",
                "Official source verified, but exact weapon-assault article unavailable in the current local extract.",
            ),
            Act(
                "Threats and intimidation",
                "threats",
                "UAE, threat, intimidation, fear, violence, coercion",
                "Threats may be legally relevant even when no physical injury is completed.",
                "A verified threat incident may be reviewed under UAE crimes and penalties provisions. Exact article-level consequence is unavailable in the current verified local source set.",
                "Federal Decree-Law No. 31 of 2021 Promulgating the Crimes and Penalties Law",
                None,
                "https://uaelegislation.gov.ae/en/legislations/1529",
                "Do not fabricate penalty ranges; verify the Arabic source text.",
            ),
            Act(
                "Robbery or taking property by violence",
                "robbery",
                "UAE, robbery, theft, force, threat, violence, weapon",
                "Robbery-style conduct is relevant where taking property is connected with force, threats, or violence.",
                "A verified violent taking may require review under theft/robbery and violence provisions. The exact article must be verified in official law.",
                "Federal Decree-Law No. 31 of 2021 Promulgating the Crimes and Penalties Law",
                None,
                "https://uaelegislation.gov.ae/en/legislations/1529",
                "Article-level source unavailable in verified local extract.",
            ),
            Act(
                "Sexual assault or sexual violence",
                "sexual_assault",
                "UAE, sexual assault, sexual violence, consent, force, threat",
                "Sexual violence allegations require specialist legal review and should not be inferred solely from generic violence detection.",
                "A verified sexual violence report may trigger serious criminal review. Guardian Eye must not infer consent, identity, or sexual context from normal fight footage.",
                "Federal Decree-Law No. 31 of 2021 Promulgating the Crimes and Penalties Law",
                None,
                "https://uaelegislation.gov.ae/en/legislations/1529",
                "Exact article-level citation unavailable; use official Arabic law text for final review.",
            ),
            Act(
                "Domestic violence or family abuse",
                "domestic_violence",
                "UAE, domestic violence, family abuse, assault, threat, protection",
                "Domestic violence context may involve assault, threats, coercion, or protective measures, but relationship and household status must be legally established.",
                "A verified domestic violence context may support protection-focused review in addition to criminal review.",
                "UAE federal legal framework; exact verified article unavailable in local source set",
                None,
                "unavailable://uae/domestic-violence-official-article",
                "Reliable article-level source was unavailable during this local rebuild; marked unavailable instead of invented.",
                official=False,
            ),
        ),
    ),
    CountryDoc(
        country="KSA",
        filename="ksa_legal_consequences.md",
        acts=(
            Act(
                "Domestic violence and abuse",
                "domestic_violence",
                "KSA, domestic violence, abuse, assault, threat, harm, protection",
                "The Saudi Protection from Abuse Law is relevant where violence or mistreatment occurs in family, dependency, guardianship, or similar protection contexts.",
                "A verified abuse context may lead to reporting, protection measures, and penalties under the Protection from Abuse Law as determined by competent authorities.",
                "Protection from Abuse Law",
                None,
                "https://www.mof.gov.sa/docslibrary/RegulationsInstructions/DocLib/%D9%86%D8%B8%D8%A7%D9%85%20%D8%A7%D9%84%D8%AD%D9%85%D8%A7%D9%8A%D8%A9%20%D9%85%D9%86%20%D8%A7%D9%84%D8%A5%D9%8A%D8%B0%D8%A7%D8%A1.pdf",
                "Official PDF source. Exact article text should be checked in Arabic for final legal review.",
            ),
            Act(
                "Physical assault outside abuse context",
                "assault",
                "KSA, assault, battery, fight, physical harm, violence",
                "Saudi criminal consequences for physical assault may involve rights of the victim, public prosecution, and court-determined outcomes.",
                "Reliable article-level official source for general assault was unavailable in the local verified source set, so no penalty or article is asserted here.",
                "Unavailable verified official article for general assault",
                None,
                "unavailable://ksa/general-assault-official-article",
                "Marked unavailable instead of reusing old demo fixture text or inventing a penalty.",
                official=False,
            ),
            Act(
                "Use of weapon or dangerous object",
                "weapon_assault",
                "KSA, weapon, dangerous object, assault, injury, threat, violence",
                "Weapon context can be legally important, but the exact KSA source and penalty must be verified from an official article.",
                "Guardian Eye can flag weapon context for legal review; it must not state a specific KSA penalty without a verified source.",
                "Unavailable verified official article for weapon assault",
                None,
                "unavailable://ksa/weapon-assault-official-article",
                "Reliable source unavailable in current rebuild.",
                official=False,
            ),
            Act(
                "Threats and intimidation",
                "threats",
                "KSA, threat, intimidation, coercion, fear, violence",
                "Threats may be relevant to criminal or protective review, especially in abuse or coercion contexts.",
                "If connected to abuse, the Protection from Abuse Law may be relevant; otherwise exact article-level consequences require verified official legal text.",
                "Protection from Abuse Law / unavailable general threats article",
                None,
                "https://www.mof.gov.sa/docslibrary/RegulationsInstructions/DocLib/%D9%86%D8%B8%D8%A7%D9%85%20%D8%A7%D9%84%D8%AD%D9%85%D8%A7%D9%8A%D8%A9%20%D9%85%D9%86%20%D8%A7%D9%84%D8%A5%D9%8A%D8%B0%D8%A7%D8%A1.pdf",
                "Use only for abuse/protection context unless a separate official threats article is verified.",
            ),
            Act(
                "Robbery or violent taking",
                "robbery",
                "KSA, robbery, theft, force, threat, weapon, violence",
                "Violent taking of property requires jurisdiction-specific legal classification.",
                "Reliable official article-level source was unavailable in the local source set, so Guardian Eye should only flag this for review and avoid penalty claims.",
                "Unavailable verified official article for robbery with violence",
                None,
                "unavailable://ksa/robbery-with-violence-official-article",
                "Marked unavailable instead of inventing source or punishment.",
                official=False,
            ),
            Act(
                "Sexual assault or sexual violence",
                "sexual_assault",
                "KSA, sexual assault, sexual violence, force, threat, consent",
                "Sexual violence requires specialist legal review and should not be inferred from generic violence detection.",
                "Reliable official article-level source was unavailable in the local rebuild; no specific KSA penalty is asserted.",
                "Unavailable verified official article for sexual violence",
                None,
                "unavailable://ksa/sexual-violence-official-article",
                "Marked unavailable instead of invented.",
                official=False,
            ),
        ),
    ),
    CountryDoc(
        country="Egypt",
        filename="egypt_legal_consequences.md",
        acts=(
            Act(
                "Assault and battery",
                "assault",
                "Egypt, assault, battery, beating, injury, fight, violence",
                "Egypt Penal Code references are relevant to assault and injury, but the verified local source is an unofficial translation and must be checked against Arabic official text.",
                "A verified assault or battery incident may lead to criminal review under the Egyptian Penal Code. Exact article and penalty are unavailable in the current verified source set.",
                "Egypt Penal Code No. 58 of 1937",
                None,
                "https://www.refworld.org/legal/legislation/natlegbod/1937/en/119651",
                "High-reliability legal archive, not an official Egyptian legislation site. Do not invent article numbers.",
                official=False,
            ),
            Act(
                "Wounding or grievous bodily harm",
                "grievous_bodily_harm",
                "Egypt, grievous bodily harm, wounding, injury, weapon, violence",
                "Serious injury or wounding requires article-level review in the Egyptian Penal Code and medical/legal fact finding.",
                "A verified serious injury context can make a case more serious, but this document does not assert an exact penalty without official article verification.",
                "Egypt Penal Code No. 58 of 1937",
                None,
                "https://www.refworld.org/legal/legislation/natlegbod/1937/en/119651",
                "Article-level official source unavailable in current local rebuild.",
                official=False,
            ),
            Act(
                "Use of weapon during assault",
                "weapon_assault",
                "Egypt, weapon, dangerous object, assault, injury, violence",
                "Weapon context may be relevant to charging or aggravation, but exact Egyptian article-level consequences require official verification.",
                "Guardian Eye can flag a weapon or dangerous-object context for legal review; it must not state a specific Egyptian penalty without verified source text.",
                "Egypt Penal Code No. 58 of 1937",
                None,
                "https://www.refworld.org/legal/legislation/natlegbod/1937/en/119651",
                "High-reliability archive only; exact official article unavailable.",
                official=False,
            ),
            Act(
                "Threats and intimidation",
                "threats",
                "Egypt, threat, intimidation, fear, coercion, violence",
                "Threats may be legally relevant even without completed physical injury.",
                "A verified threat incident may require Egyptian Penal Code review, but no exact article or penalty is asserted from the current source set.",
                "Egypt Penal Code No. 58 of 1937",
                None,
                "https://www.refworld.org/legal/legislation/natlegbod/1937/en/119651",
                "Exact official article unavailable; marked clearly.",
                official=False,
            ),
            Act(
                "Robbery or taking by force",
                "robbery",
                "Egypt, robbery, theft, force, threat, violence, weapon",
                "Robbery-like conduct requires distinguishing theft, force, threat, injury, and weapon facts under Egyptian law.",
                "A verified violent taking may require Penal Code review. Exact article-level consequences are unavailable in the current verified source set.",
                "Egypt Penal Code No. 58 of 1937",
                None,
                "https://www.refworld.org/legal/legislation/natlegbod/1937/en/119651",
                "Do not invent article or penalty.",
                official=False,
            ),
            Act(
                "Homicide or murder",
                "homicide",
                "Egypt, homicide, murder, death, lethal violence",
                "Homicide provisions are relevant only where death and the required legal elements are established by competent authorities.",
                "Guardian Eye should not infer homicide from generic violence detection; it can only route lethal-harm facts to homicide-specific legal review.",
                "Egypt Penal Code No. 58 of 1937",
                None,
                "https://www.refworld.org/legal/legislation/natlegbod/1937/en/119651",
                "Exact official article unavailable in current source set.",
                official=False,
            ),
            Act(
                "Sexual assault or sexual violence",
                "sexual_assault",
                "Egypt, sexual assault, sexual violence, force, threat, consent",
                "Sexual violence requires specialist legal review and cannot be inferred from ordinary fight detection.",
                "Reliable article-level official source was unavailable in the local rebuild, so no exact penalty is asserted here.",
                "Unavailable verified official article for sexual violence",
                None,
                "unavailable://egypt/sexual-violence-official-article",
                "Marked unavailable instead of invented.",
                official=False,
            ),
        ),
    ),
)


def main() -> int:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chunks: list[dict[str, object]] = []
    for doc in DOCS:
        markdown = build_markdown(doc)
        (DOCS_DIR / doc.filename).write_text(markdown, encoding="utf-8")
        for act in doc.acts:
            text = section_text(doc.country, act)
            chunks.append(chunk_for(doc.country, act, text))

    CHUNKS_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(DOCS)} curated legal document(s) to {DOCS_DIR}")
    print(f"Wrote {len(chunks)} curated legal chunk(s) to {CHUNKS_PATH}")
    return 0


def build_markdown(doc: CountryDoc) -> str:
    sections = [
        f"# {doc.country} Legal Consequences - Curated RAG Document",
        "",
        "Purpose: self-contained retrieval sections for Guardian Eye Legal RAG.",
        "Rules: no webpage navigation, no footer text, no unrelated scraped sections.",
        "",
    ]
    for act in doc.acts:
        sections.append(section_text(doc.country, act))
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def section_text(country: str, act: Act) -> str:
    article = act.article or "Unavailable in verified local source set"
    source = act.source
    return "\n".join(
        [
            f"## {country} — {act.name}",
            f"Country: {country}",
            f"Violent act: {act.name}",
            f"Keywords: {act.keywords}",
            f"Legal meaning: {act.legal_meaning}",
            f"Possible legal consequence: {act.consequence}",
            f"Relevant law/article/section: {act.law}; {article}",
            f"Source: {source}",
            (
                "Retrieval guidance: Use this section for country-specific questions about "
                f"{country} {act.name}, {act.category.replace('_', ' ')}, weapons, threats, "
                "injury, domestic context, public fighting, or violence where the keywords match. "
                "Do not use it to identify a suspect, decide guilt, infer consent, or state a "
                "penalty that is not present in the cited source line."
            ),
            f"Notes: {act.notes}",
        ]
    )


def chunk_for(country: str, act: Act, text: str) -> dict[str, object]:
    basis = f"{country}|{act.name}|{act.source}|{text}"
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]
    return {
        "chunk_id": f"curated-legal-{digest}",
        "country": country,
        "source_language": "en",
        "law_title": act.law,
        "article_number": act.article,
        "section_title": act.name,
        "violence_category": act.category,
        "source_url": act.source,
        "official_source": act.official,
        "text": text,
        "matched_keywords": keywords_from(act.keywords),
    }


def keywords_from(value: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for raw in value.split(","):
        term = re.sub(r"\s+", " ", raw).strip()
        if term and term.casefold() not in seen:
            seen.add(term.casefold())
            terms.append(term)
    return terms


if __name__ == "__main__":
    raise SystemExit(main())
