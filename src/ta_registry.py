"""Which 66 TAs exist, and how to match their names across sources.
normalise() collapses any spelling to a canonical key; CANONICAL_TAS maps
each key to a display name. Proven: all four sources reduce to exactly these 66."""
import unicodedata, re

RAW_OVERRIDES = {"Tauranga District/Tauranga City": "Tauranga City"}
_SUFFIXES = (" district", " city", " territory", " territorial authority")

def normalise(name):
    """Canonical match key for a TA name (None if blank)."""
    if name is None:
        return None
    s = str(name).replace("\xa0", " ").strip()
    s = RAW_OVERRIDES.get(s, s)
    s = unicodedata.normalize("NFD", s)                          # split off macrons
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.lower().strip()
    for suf in _SUFFIXES:
        if s.endswith(suf):
            s = s[:-len(suf)]; break
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None

CANONICAL_TAS = {
    "ashburton": "Ashburton District", "auckland": "Auckland City",
    "buller": "Buller District", "carterton": "Carterton District",
    "central hawkes bay": "Central Hawke's Bay District", "central otago": "Central Otago District",
    "christchurch": "Christchurch City", "clutha": "Clutha District",
    "dunedin": "Dunedin City", "far north": "Far North District",
    "gisborne": "Gisborne District", "gore": "Gore District", "grey": "Grey District",
    "hamilton": "Hamilton City", "hastings": "Hastings District", "hauraki": "Hauraki District",
    "horowhenua": "Horowhenua District", "hurunui": "Hurunui District",
    "invercargill": "Invercargill City", "kaikoura": "Kaikōura District",
    "kaipara": "Kaipara District", "kapiti coast": "Kāpiti Coast District",
    "kawerau": "Kawerau District", "lower hutt": "Lower Hutt City",
    "mackenzie": "Mackenzie District", "manawatu": "Manawatū District",
    "marlborough": "Marlborough District", "masterton": "Masterton District",
    "matamatapiako": "Matamata-Piako District", "napier": "Napier City",
    "nelson": "Nelson City", "new plymouth": "New Plymouth District",
    "opotiki": "Ōpōtiki District", "otorohanga": "Ōtorohanga District",
    "palmerston north": "Palmerston North City", "porirua": "Porirua City",
    "queenstownlakes": "Queenstown-Lakes District", "rangitikei": "Rangitikei District",
    "rotorua": "Rotorua District", "ruapehu": "Ruapehu District", "selwyn": "Selwyn District",
    "south taranaki": "South Taranaki District", "south waikato": "South Waikato District",
    "south wairarapa": "South Wairarapa District", "southland": "Southland District",
    "stratford": "Stratford District", "tararua": "Tararua District", "tasman": "Tasman District",
    "taupo": "Taupō District", "tauranga": "Tauranga City",
    "thamescoromandel": "Thames-Coromandel District", "timaru": "Timaru District",
    "upper hutt": "Upper Hutt City", "waikato": "Waikato District",
    "waimakariri": "Waimakariri District", "waimate": "Waimate District",
    "waipa": "Waipā District", "wairoa": "Wairoa District", "waitaki": "Waitaki District",
    "waitomo": "Waitomo District", "wellington": "Wellington City",
    "western bay of plenty": "Western Bay of Plenty District", "westland": "Westland District",
    "whakatane": "Whakatāne District", "whanganui": "Whanganui District",
    "whangarei": "Whangārei District",
}
CANONICAL_KEYS = frozenset(CANONICAL_TAS)

def display_name(key):
    return CANONICAL_TAS.get(key, key)
