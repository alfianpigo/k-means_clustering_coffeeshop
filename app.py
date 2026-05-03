# app.py
import re
import io
import unicodedata
from typing import List, Dict, Set, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_dark"


# =========================================================
# UI CONFIG
# =========================================================
st.set_page_config(
    page_title="Clustering Ulasan Coffee Shop Depok (K-Means, TF-IDF)",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 800px at 15% 10%, rgba(56,189,248,0.18), transparent 60%),
    radial-gradient(1000px 700px at 85% 25%, rgba(168,85,247,0.18), transparent 55%),
    radial-gradient(900px 700px at 50% 95%, rgba(34,197,94,0.10), transparent 55%),
    linear-gradient(180deg, rgba(2,6,23,1) 0%, rgba(15,23,42,1) 55%, rgba(2,6,23,1) 100%);
}
[data-testid="stHeader"]{ background: transparent; }

.block-container{
  padding-top: 1.0rem;
  padding-bottom: 2.2rem;
  max-width: 1280px;
}

section[data-testid="stSidebar"]{
  background:
    radial-gradient(900px 600px at 10% 20%, rgba(56,189,248,0.10), transparent 55%),
    radial-gradient(900px 650px at 90% 30%, rgba(168,85,247,0.10), transparent 55%),
    linear-gradient(180deg, rgba(2,6,23,0.90) 0%, rgba(15,23,42,0.82) 70%, rgba(2,6,23,0.90) 100%) !important;
  border-right: 1px solid rgba(148,163,184,0.16);
  backdrop-filter: blur(10px);
}

div[data-testid="stMetric"]{
  background: rgba(15,23,42,0.55);
  border: 1px solid rgba(148,163,184,0.18);
  border-radius: 16px;
  padding: 12px 14px;
  box-shadow: 0 12px 40px rgba(0,0,0,0.20);
  backdrop-filter: blur(10px);
}

div[data-testid="stTabs"] button{
  border-radius: 14px !important;
  padding: 10px 14px !important;
}

div[data-testid="stDataFrame"]{
  border: 1px solid rgba(148,163,184,0.14);
  border-radius: 14px;
  overflow: hidden;
  margin: 1.2rem 0 !important;
}

button[data-testid="collapsedControl"]{
  position: fixed !important;
  top: 14px !important;
  left: 14px !important;
  z-index: 999999 !important;
  width: 44px !important;
  height: 44px !important;
  border-radius: 14px !important;
  background: rgba(15,23,42,0.78) !important;
  border: 1px solid rgba(148,163,184,0.28) !important;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35) !important;
}
button[data-testid="collapsedControl"] svg{
  width: 22px !important;
  height: 22px !important;
  color: rgba(248,250,252,0.95) !important;
}

.hero{
  padding: 22px 18px;
  border-radius: 18px;
  background: rgba(15,23,42,0.50);
  border: 1px solid rgba(148,163,184,0.18);
  box-shadow: 0 14px 42px rgba(0,0,0,0.22);
  backdrop-filter: blur(10px);
  margin-bottom: 12px;
}
.hero-title{
  font-size: 2.0rem;
  font-weight: 900;
  letter-spacing: -0.5px;
  color: rgba(248,250,252,0.95);
}
.hero-sub{
  margin-top: 8px;
  color: rgba(148,163,184,1);
  font-size: 1.02rem;
}
.section-title{
  font-size: 1.5rem;
  font-weight: 850;
  color: rgba(226,232,240,0.98);
  margin-top: 1.5rem;
  margin-bottom: 1.0rem;
  padding-bottom: 0.7rem;
  border-bottom: 2px solid rgba(56,189,248,0.35);
}
.small-note{
  font-size: 0.95rem;
  color: rgba(148,163,184,0.95);
  margin: 0.8rem 0 1.0rem 0;
  padding: 0.8rem 1.0rem;
  border-left: 3px solid rgba(56,189,248,0.40);
  background: rgba(15,23,42,0.35);
  border-radius: 10px;
}
.card{
  background: rgba(15,23,42,0.55);
  border: 1px solid rgba(148,163,184,0.18);
  border-radius: 14px;
  padding: 1.0rem 1.1rem;
  margin: 0.7rem 0;
  box-shadow: 0 10px 32px rgba(0,0,0,0.18);
  backdrop-filter: blur(10px);
}
.card h4{
  margin: 0 0 0.5rem 0;
  color: rgba(248,250,252,0.95);
}
.mono{
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  color: rgba(226,232,240,0.95);
  white-space: pre-wrap;
  word-break: break-word;
}
.badge{
  display:inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  border: 1px solid rgba(148,163,184,0.22);
  background: rgba(2,6,23,0.45);
  color: rgba(226,232,240,0.95);
  font-size: 0.85rem;
  margin-right: 8px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

CSS_SIDEBAR = """
<style>
section[data-testid="stSidebar"]{ padding-top: 0.6rem; }
.sidebar-title{
  display:flex;
  align-items:center;
  gap:10px;
  font-weight:900;
  letter-spacing: .3px;
  font-size: 1.35rem;
  color: rgba(226,232,240,0.95);
  margin: 0.2rem 0 0.3rem 0;
}
.sidebar-divider{
  border: none;
  border-top: 1px solid rgba(148,163,184,0.16);
  margin: 0.8rem 0;
}
section[data-testid="stSidebar"] details{
  background: rgba(15,23,42,0.52);
  border: 1px solid rgba(148,163,184,0.18);
  border-radius: 16px;
  padding: 10px 10px;
  margin-bottom: 10px;
  box-shadow: 0 12px 36px rgba(0,0,0,0.22);
  backdrop-filter: blur(10px);
}
section[data-testid="stSidebar"] summary{
  font-weight: 820;
  color: rgba(248,250,252,0.95);
  padding: 2px 4px;
}
section[data-testid="stSidebar"] button[kind="primary"]{
  border-radius: 14px !important;
  padding: 0.70rem 0.9rem !important;
  font-weight: 850 !important;
}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] .stSlider{
  border-radius: 14px !important;
}
</style>
"""
st.markdown(CSS_SIDEBAR, unsafe_allow_html=True)


# =========================================================
# PREPROCESSING (VERSI LEBIH TAJAM)
# Target: final_text lebih "isi" → buang noise, expand singkatan, fix typo, buang token asing/tidak bermakna
# =========================================================
factory_stem = StemmerFactory()
STEMMER = factory_stem.create_stemmer()

factory_sw = StopWordRemoverFactory()
BASE_STOPWORDS = set(factory_sw.get_stop_words())

# stopwords gaya ulasan (lebih agresif + comprehensive - v2)
DEFAULT_CUSTOM_STOPWORDS: Set[str] = {
    # partikel / filler umum
    "nya", "aja", "nih", "deh", "dong", "kok", "sih", "yah", "ya", "y", "iya", "iy",
    "lah", "kah", "tah", "pun", "kan",
    # tawa / noise
    "wkwk", "wk", "wkwkwk", "hehe", "hihi", "haha", "xixi", "kwkw",
    # singkatan umum lebih agresif
    "yg", "yng", "dgn", "dg", "dr", "tdk", "gk", "ga", "gak", "nggak", "ngga", "klo", "klu", "krn", "karna",
    "jg", "dll", "dkk", "etc", "dst", "ref", "utk", "sm", "bs", "kb",
    # pronomina & kata fungsi yang sering tidak membantu cluster (EXPANDED)
    "aku", "saya", "gue", "gw", "lu", "kamu", "dia", "mereka", "kita", "kami", "anda", "beliau",
    "ini", "itu", "sini", "situ", "sana", "disini", "disitu", "disana", "dalelah",
    "yang", "dan", "atau", "dari", "ke", "di", "pada", "dengan", "untuk", "karena", "tapi", "tetapi",
    "akan", "telah", "sudah", "pernah", "juga", "saja", "hanya", "saat", "ketika", "lalu", "kemudian",
    # waktu/urutan lebih agresif
    "pas", "saat", "ketika", "lagi", "tadi", "kemarin", "besok", "hari", "minggu", "bulan", "tahun",
    "pagi", "siang", "malam", "sore", "subuh", "waktu",
    # intensifier umum
    "banget", "bgt", "sangat", "amat", "cukup", "lumayan", "sekali", "lebih", "kurang", "agak", "mungkin", "kayak",
    "sekali", "sekalian", "sama", "bersama",
    # kata review generik tidak membantu
    "overall", "recommended", "rekomend", "rekomen",
    # lain-lain umum
    "tuh", "sama", "beri", "kaya", "bal", "sedia", "buat", "mau", "kapan", "kesana", "musti", "mobil", "liat",
    "request", "tengah", "rumah", "jangkau", "biar", "malem", "tambah", "berbeda",
    # verb umum lemah
    "ada", "ada2", "punya", "punya2", "biasa", "kali", "soal", "hal", "bab", "poin", "point", "aspek",
    "perlu", "harus", "boleh", "bisa", "bisa2",
    # domain sering lemah
    "orang", "ramai", "sepi", "macet", "rame", "rame2", "sepi2",
    # location & generic reference
    "rumah", "rumah2", "rumahnya", "lokasi", "lokasi2", "tempat2", "lapak",
    # unclear/vague terms
    "bagini", "begitunya", "like", "such", "something", "stuff",
}

# domain stopwords (lebih tajam; kamu bisa matikan)
DOMAIN_STOPWORDS: Set[str] = {
    # terlalu umum/generic adjektif
    "enak", "mantap", "bagus", "oke", "ok", "jahat",
    # tempat & fungsi generik
    "tempat", "makan", "minum", "kopi", "coffee", "cafe", "kafe", "menu", "rasa",
    "harga", "murah", "mahal", "promo", "diskon",
    "orang", "ramai", "sepi", "rame", "rame2",
    # minuman pendamping noise
    "teh", "tea", "es", "ice", "soda", "smoothie", "air", "minuman", "susu", "yogurt",
    "juice", "jus", "kombucha", "matcha", "boba", "teh2", "susu2",
    "espresso", "latte", "cappuccino", "americano",
    "mocha", "macchiato", "flat", "cortado",
    # food-specific items (apa yang dipesan, bukan feature)
    # - makanan lokal/spesifik
    "nasgitel", "nasi", "gitel", "camilan", "camil", "snack", "snack2", "makanan",
    "ketan", "ketan2", "goreng", "gorengnya", "gorengan", "gorengan2",
    "keripik", "kripik", "kripik2", "udang", "ikan", "ikan2", "ayam", "daging", "telur",
    "roti", "bread", "pastry", "donut", "cake", "kue", "kue2",
    "pizza", "pasta", "burrito", "sandwich", "burger", "mie", "indomie", "kwetiau", "lumpia",
    # non-informative generic/unclear
    "barang", "barang2", "item", "jenis", "macam", "tipe", "solo", "sesuatu",
    "seperti", "mirip", "begini", "begitunya", "gini2",
}

# kata "kotor" / token yang sering muncul tapi tidak punya makna inti
GARBAGE_TOKENS: Set[str] = {
    # partikel filler
    "gitu", "gini", "aja", "doang", "cuma", "sih", "nih", "deh", "dong", "kok", "saja",
    "banget", "bgt", "bgtt", "overall", "bangetnya", "banyak", "banyak2",
    # tanda interjeksi/emosi pendek
    "ah", "eh", "oh", "whoa", "wow", "yay", "bah", "tuh", "loh", "nah", "nih", "enak", "mantap", "nih2",
    # verba generik/lemah
    "ada", "ada2", "biasa", "kali", "soal", "hal", "bab", "poin", "point", "aspek",
    "perlu", "punya", "punya2", "harus", "boleh", "bisa", "bisa2", "musti",
    # angka/simbol jadi huruf
    "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t", "v", "w", "x", "z",
    # singkatan lemah
    "dll", "dkk", "etc", "dst", "ref", "utk", "sm", "ya", "yg", "kb", "bs", "bb",
    # filler internetais
    "lol", "omg", "rtfm", "idk",
    # token tidak jelas/vague
    "kan", "nan", "pun", "tuh", "gak", "nggak", "ngga",
    "begini", "begini2", "seperti", "mirip", "sama2", "macam", "soal2",
    # custom non-informative
    "cantik", "cantik2", "bagus2", "oke2", "okoke", "baik2",
}

# Normalisasi singkatan / typo yang sering muncul (VERSI EXPANDED)
NORMALIZE_DICT: Dict[str, str] = {
    # intensifier - normalize ke base form
    "bgt": "banget", "bgtt": "banget", "bnget": "banget", "bgtss": "banget", "bgt2": "banget",
    "bynyknya": "banyak", "bnyknya": "banyak",
    # negasi - normalize ke bentuk standar (hapus nanti di GARBAGE_TOKENS)
    "ga": "tidak", "gak": "tidak", "gk": "tidak", "nggak": "tidak", "ngga": "tidak", "tdk": "tidak", "nga": "tidak",
    # konektor
    "tp": "tapi", "tpi": "tapi", "pdhl": "padahal", "krn": "karena", "karna": "karena", "krna": "karena",
    # kata kerja/umum
    "udh": "sudah", "udah": "sudah", "dah": "sudah", "sdh": "sudah", "udh2": "sudah",
    "trus": "terus", "trs": "terus",
    "jg": "juga", "jugaa": "juga",
    "jd": "jadi", "jdi": "jadi", "jdnya": "jadi",
    "utk": "untuk", "untk": "untuk", "utuk": "untuk",
    "dgn": "dengan", "dg": "dengan", "dgn2": "dengan",
    "dr": "dari", "drn": "dari",
    "sm": "sama", "sama2": "sama",
    "bikin": "membuat",
    "pesen": "pesan", "pesen2": "pesan", "pesn": "pesan",
    "nyari": "cari", "nyariin": "cari",
    "kmrn": "kemarin", "kmrin": "kemarin",
    "klo": "kalau", "klu": "kalau", "kalo": "kalau",
    "gmn": "bagaimana", "gmna": "bagaimana", "gmn2": "bagaimana",
    "slh": "salah", "salh": "salah",
    "byk": "banyak", "bnyk": "banyak", "bnyak": "banyak", "bny": "banyak",
    # typo - perbaiki ke kata semantik lebih jelas
    "ras": "rasa", "rass": "rasa", "rasanya": "rasa",
    "pel": "pelaya", "pela": "pelaya", # ke pelayan (auto-stem)
    # domain - normalisasi ke bentuk canonical
    "coffeeshop": "coffee", "coffe": "coffee", "coffea": "coffee", "coffeshop": "coffee",
    "photobox": "photo", "photobooth": "photo",
    "colokan": "colok", "colokannya": "colok", "colokan2": "colok",
    "parkiran": "parkir", "parkirnya": "parkir", "parkir2": "parkir",
    "musolla": "musola", "mushola": "musola", "musholla": "musola", "musallah": "musola",
    "toiletnya": "toilet", "toiletnye": "toilet", "toiletny": "toilet",
    "wc": "toilet", "wcnya": "toilet",
    # makna - clarify
    "penganan": "camilan", "penanan": "camilan",
    "nasgitelnya": "nasgitel", "nasgitinya": "nasgitel", "nasittel": "nasgitel",
    "ikannya": "ikan",
    "harganya": "harga", "harga2": "harga",
    "suasananya": "suasana", "suasana2": "suasana",
    "minum2": "minum",
    # expansion tipikal
    "kuy": "ayo", "yuk": "ayo", "yok": "ayo",
    "pantes": "pantas",
    "promo": "promosi",
    "disc": "diskon", "diskon2": "diskon",
    "pas": "tepat",
    "langganan": "pelanggan",
    "order2": "order",
    # food items → kategori (agar lebih abstrak)
    "nasgitel": "makanan", "nasitel": "makanan", "nasgitelnya": "makanan",
    "camil": "makanan", "camilan2": "makanan", "camilannya": "makanan",
    "ketan2": "makanan", "ketannya": "makanan",
    "gorengan2": "makanan", "gorengnya2": "makanan", "gorengan": "makanan", "goreng": "makanan",
    "keripik2": "makanan", "kripik2": "makanan", "keripiknya": "makanan",
    # location vague
    "rumah2": "tempat", "rumahnya2": "tempat", "rumahnya": "tempat", "rumah": "tempat",
    "lokasi2": "tempat", "lokasinya": "tempat",
    "solo": "tempat",
    # reduce false-positive features
    "cantik2": "menarik", "cantiknya": "menarik",
    "lumayan2": "cukup", "lumayannya": "cukup",
    "suasanan": "suasana", "suasana2": "suasana", "suasananya": "suasana",
    # additional vague terms
    "sesuatu": "item",
    "gitu": "begitu",
    "gini": "begini",
}

# typo correction tambahan (pattern-based) – murah tapi efektif
# contoh: "kualitass" -> "kualitas"
REPEAT_END_RE = re.compile(r"([a-z])\1+$", re.IGNORECASE)

# Token alfabet (min 2 huruf)
TOKEN_RE = re.compile(r"[a-zA-Z]{2,}")

# Suffix yang sering nempel
CLITIC_SUFFIXES = ("nya", "ku", "mu", "lah", "kah", "tah", "pun")

def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def _compress_repeated_chars(token: str) -> str:
    # "enaaakkk" -> "enaakk"
    return re.sub(r"(.)\1{2,}", r"\1\1", token)

def _strip_clitics(tok: str) -> str:
    for suf in CLITIC_SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok

def _final_typo_fix(tok: str) -> str:
    # hapus huruf yang ngulang di akhir berlebihan: "kualitassss" -> "kualitas"
    tok2 = REPEAT_END_RE.sub(r"\1", tok)
    return tok2

def clean_and_casefold(text: str) -> str:
    if pd.isna(text):
        return ""
    t = str(text)
    t = _strip_accents(t)
    t = t.lower()

    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"[@#]\w+", " ", t)
    t = re.sub(r"\d+", " ", t)                 # buang angka
    t = re.sub(r"[^a-zA-Z\s]", " ", t)         # sisakan huruf
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return TOKEN_RE.findall(text)

def normalize_tokens(tokens: List[str], norm_dict: Dict[str, str]) -> List[str]:
    """
    Normalisasi token dengan langkah-langkah ketat:
    1. Kompresi karakter berulang
    2. Lookup di normalisasi dict
    3. Strip clitics (suffix)
    4. Lookup normalisasi dict lagi
    5. Fix typo akhir
    6. Filter panjang >= 4 (LEBIH KETAT dari sebelumnya)
    7. Filter garbage tokens
    """
    out: List[str] = []
    for t in tokens:
        t2 = _compress_repeated_chars(t)
        t2 = norm_dict.get(t2, t2)
        t2 = _strip_clitics(t2)
        t2 = norm_dict.get(t2, t2)
        t2 = _final_typo_fix(t2)

        # filter ketat LEBIH KETAT:
        # - buang token panjangnya < 4 (bukan 3)
        # - buang token "garbage"
        # - buang token single-char
        if len(t2) < 4:
            continue
        if t2 in GARBAGE_TOKENS:
            continue

        out.append(t2)
    return out

def remove_stopwords(tokens: List[str], stopwords: Set[str]) -> List[str]:
    """
    Hapus stopwords + domain stopwords + token sangat pendek (>= 4 karakter)
    """
    combined_stopwords = stopwords | DOMAIN_STOPWORDS
    return [t for t in tokens if t not in combined_stopwords and len(t) >= 4]

def stem_sentence(tokens: List[str]) -> List[str]:
    """
    Stemming dengan post-filter ketat
    """
    if not tokens:
        return []
    joined = " ".join(tokens)
    stemmed = STEMMER.stem(joined)
    toks = stemmed.split()
    # post-filter setelah stemming (kadang jadi pendek) - LEBIH KETAT
    # filter: panjang >= 4, bukan garbage, gabungkan dengan DOMAIN_STOPWORDS
    combined_stopwords = GARBAGE_TOKENS | DOMAIN_STOPWORDS
    toks = [t for t in toks if len(t) >= 4 and t not in combined_stopwords]
    return toks

def pipeline_preprocess(
    df: pd.DataFrame,
    text_col: str,
    norm_dict: Dict[str, str],
    stopwords: Set[str],
    use_clean: bool = True,
    use_stop: bool = True,
    use_stem: bool = True,
) -> pd.DataFrame:
    """
    Pipeline preprocessing dengan tahapan:
    1. Casefold & cleaning
    2. Tokenizing
    3. Normalizing (ketat: min 4 char)
    4. Stopword removal (ketat: min 4 char)
    5. Stemming (ketat: min 4 char)
    6. Final cleanup (ketat: min 4 char)
    """
    out = df.copy()
    out["_raw"] = out[text_col].astype(str)

    if use_clean:
        out["cleaning_casefold"] = out["_raw"].apply(clean_and_casefold)
    else:
        out["cleaning_casefold"] = out["_raw"].astype(str).str.lower()

    out["tokenizing"] = out["cleaning_casefold"].apply(tokenize)
    out["normalizing"] = out["tokenizing"].apply(lambda toks: normalize_tokens(toks, norm_dict))

    if use_stop:
        out["stopword_removal"] = out["normalizing"].apply(lambda toks: remove_stopwords(toks, stopwords))
    else:
        out["stopword_removal"] = out["normalizing"].apply(lambda toks: [t for t in toks if len(t) >= 4])

    if use_stem:
        out["stemming"] = out["stopword_removal"].apply(stem_sentence)
    else:
        out["stemming"] = out["stopword_removal"].apply(lambda toks: toks)

    # final: rapikan lagi dengan filter ketat
    out["final_text"] = out["stemming"].apply(lambda toks: " ".join([t for t in toks if len(t) >= 4]))
    return out


# =========================================================
# CLUSTER LABELING (AUTO + FIX K=3)
# =========================================================
THEME_KEYWORDS = {
    "Pelayanan": {"layan", "pelayan", "staff", "kasir", "order", "pesan", "antri", "ramah", "cepat", "service", "sopan"},
    "Suasana & Fasilitas": {"suasana", "nyaman", "luas", "bersih", "parkir", "toilet", "musola", "wifi", "colok",
                           "indoor", "outdoor", "rame", "sejuk", "ac", "smoking", "nonsmoking", "kursi", "meja", "ruang",
                           "live", "music", "musik", "akustik", "nongki", "nongkrong", "nugas"},
    "Rasa Makan & Minum": {"rasa", "kopi", "menu", "manis", "asin", "pahit", "susu", "latte",
                           "espresso", "snack", "kue", "murah", "mahal"}
}

def top_terms_per_cluster(tfidf_matrix, labels, feature_names, topn=15):
    top = {}
    for c in sorted(np.unique(labels)):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            top[c] = []
            continue
        mean_vec = tfidf_matrix[idx].mean(axis=0)
        mean_arr = np.asarray(mean_vec).ravel()
        top_idx = mean_arr.argsort()[::-1][:topn]
        top[c] = [(feature_names[i], float(mean_arr[i])) for i in top_idx]
    return top

def top_terms_unique_per_cluster(tfidf_matrix, labels, feature_names, cluster_id, topn=20):
    global_mean = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    mask = (labels == cluster_id)
    if mask.sum() == 0:
        return []
    cluster_mean = np.asarray(tfidf_matrix[mask].mean(axis=0)).ravel()
    unique = cluster_mean - global_mean
    top_idx = np.argsort(unique)[::-1][:topn]
    return [(feature_names[i], float(unique[i])) for i in top_idx]

def assign_theme_for_cluster(top_terms):
    scores = {theme: 0.0 for theme in THEME_KEYWORDS}
    if len(top_terms) == 0:
        return "Suasana & Fasilitas", scores
    for rank, (t, _w) in enumerate(top_terms):
        rank_boost = (len(top_terms) - rank) / len(top_terms)
        for theme, kws in THEME_KEYWORDS.items():
            if t in kws:
                scores[theme] += 1.0 * rank_boost
    best = max(scores.items(), key=lambda x: x[1])[0]
    return best, scores

def remap_clusters_to_fixed_ids(cluster_to_theme):
    desired_order = ["Pelayanan", "Suasana & Fasilitas", "Rasa Makan & Minum"]
    theme_to_desired_id = {t: i for i, t in enumerate(desired_order)}

    mapping = {}
    used_desired = set()

    for theme in desired_order:
        candidates = [c for c, th in cluster_to_theme.items() if th == theme]
        if candidates:
            c = candidates[0]
            did = theme_to_desired_id[theme]
            mapping[c] = did
            used_desired.add(did)

    remaining_clusters = [c for c in cluster_to_theme.keys() if c not in mapping]
    remaining_ids = [i for i in [0, 1, 2] if i not in used_desired]
    for c, did in zip(remaining_clusters, remaining_ids):
        mapping[c] = did

    return mapping


# =========================================================
# DATA LOADER (WAJIB UPLOAD)
# =========================================================
@st.cache_data(show_spinner=False)
def load_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ CONTROL PANEL</div>', unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    with st.expander("📁 Upload Data", expanded=True):
        up = st.file_uploader("Upload file (CSV)", type=["csv"], key="uploader_csv")
        st.caption("Wajib upload CSV dulu, lalu klik **Jalankan Proses**.")

    with st.expander("🛠️ Pengaturan Preprocessing", expanded=False):
        use_clean = st.checkbox("Cleaning text & case folding", value=True, key="cb_clean")
        use_stop = st.checkbox("Stopword removal", value=True, key="cb_stop")
        use_stem = st.checkbox("Stemming (Sastrawi)", value=True, key="cb_stem")

        use_domain_stop = st.checkbox("Domain stopwords", value=True, key="cb_domain_stop")
        st.caption("Aktif: kata umum (kopi/teh/harga/tempat) makin dibuang → cluster lebih fokus.")

        st.markdown("---")
        add_stopwords_text = st.text_area(
            "Stopwords tambahan (pisahkan koma/spasi)",
            value="",
            placeholder="contoh: nasgitel, penganan, citylight, request (kalau mau dibuang)",
            height=110,
            key="ta_add_stop"
        )
        preview_rows = st.slider("Jumlah contoh hasil preprocessing", 5, 60, 12, step=1, key="sl_preview_rows")

    with st.expander("📊 Pengaturan TF-IDF", expanded=False):
        max_features = st.number_input("max_features", min_value=300, max_value=30000, value=4000, step=100, key="ni_maxfeat")
        ngram_max = st.slider("n-gram range (max)", min_value=1, max_value=2, value=2, step=1, key="sl_ngram")
        ngram = (1, ngram_max)
        min_df = st.number_input("min_df (minimal dokumen)", min_value=1, max_value=100, value=2, step=1, key="ni_mindf")
        max_df = st.slider("max_df (maks dokumen %)", min_value=0.10, max_value=1.00, value=0.90, step=0.05, key="sl_maxdf")

    with st.expander("🎯 K-Means Clustering", expanded=True):
        k_mode = st.radio("Pilih cara menentukan K", ["Manual", "Elbow"], index=1, key="radio_kmode")
        random_state = st.number_input("random_state", min_value=0, max_value=9999, value=42, step=1, key="ni_rs")
        if k_mode == "Manual":
            k_final = st.slider("Jumlah Cluster (K)", 2, 10, 3, key="sl_kfinal")
        else:
            k_final = 3
        run_btn = st.button("▶ Jalankan Proses", type="primary", use_container_width=True, key="btn_run")


# =========================================================
# LANDING (WAJIB UPLOAD)
# =========================================================
def landing_page():
    st.markdown("""
    <div class="hero">
      <div class="hero-title">Clustering Ulasan Google Maps Coffee Shop Depok</div>
      <div class="hero-sub">Preprocessing (tajam) → TF-IDF → K-Means → PCA → Wordcloud</div>
    </div>
    """, unsafe_allow_html=True)
    st.info("Upload CSV di sidebar untuk memulai.")
    st.stop()

if "run_clicked" not in st.session_state:
    st.session_state.run_clicked = False
if run_btn:
    st.session_state.run_clicked = True

if up is None:
    landing_page()
if not st.session_state.run_clicked:
    st.warning("CSV sudah dipilih. Klik **Jalankan Proses** di sidebar.")
    st.stop()


# =========================================================
# LOAD DF + SELECT TEXT COLUMN
# =========================================================
df = load_uploaded_csv(up.getvalue())

candidate_text_cols = [c for c in df.columns if "review" in c.lower() and "text" in c.lower()]
default_text_col = candidate_text_cols[0] if candidate_text_cols else df.columns[-1]

st.markdown("""
<div class="hero">
  <div class="hero-title">Clustering Ulasan Coffee Shop Depok</div>
  <div class="hero-sub">Hasil preprocessing lebih tajam: singkatan/typo/clitic dibersihkan + token pendek dibuang.</div>
</div>
""", unsafe_allow_html=True)

text_col = st.selectbox(
    "Pilih kolom teks ulasan",
    options=list(df.columns),
    index=list(df.columns).index(default_text_col),
    key="sel_text_col"
)

m1, m2, m3, m4 = st.columns(4, gap="large")
m1.metric("Sumber Data", "Upload CSV")
m2.metric("Jumlah Baris", f"{len(df):,}")
m3.metric("Jumlah Kolom", f"{df.shape[1]}")
m4.metric("Kolom Teks", str(text_col))

tabs = st.tabs([
    "Data",
    "Preprocessing",
    "TF-IDF & Elbow",
    "K-Means & PCA",
    "Wordcloud & Karakteristik",
    "Download"
])


# =========================================================
# TAB: DATA
# =========================================================
with tabs[0]:
    st.markdown('<div class="section-title">Preview Dataset</div>', unsafe_allow_html=True)
    try:
        df_unique = df.drop_duplicates(subset=[text_col]) if text_col in df.columns else df.drop_duplicates()
    except Exception:
        df_unique = df
    n_preview = min(50, len(df_unique)) if len(df_unique) > 0 else 0
    st.dataframe(df_unique.sample(n=n_preview, random_state=42).reset_index(drop=True) if n_preview > 0 else df_unique.head(0),
                 use_container_width=True)


# =========================================================
# RUN PREPROCESS (TAJAM)
# =========================================================
custom_added = set()
if add_stopwords_text.strip():
    tmp = re.split(r"[,\s]+", add_stopwords_text.strip().lower())
    custom_added = {t for t in tmp if t}

STOPWORDS = set(BASE_STOPWORDS) | set(DEFAULT_CUSTOM_STOPWORDS) | custom_added
if use_domain_stop:
    STOPWORDS |= set(DOMAIN_STOPWORDS)

@st.cache_data(show_spinner=True)
def run_preprocess_cached(
    _df_in: pd.DataFrame,
    text_col_in: str,
    _use_clean: bool,
    _use_stop: bool,
    _use_stem: bool,
    _stopwords_size: int,
) -> pd.DataFrame:
    return pipeline_preprocess(
        _df_in,
        text_col_in,
        NORMALIZE_DICT,
        STOPWORDS,
        use_clean=_use_clean,
        use_stop=_use_stop,
        use_stem=_use_stem,
    )

df_prep = run_preprocess_cached(df, text_col, use_clean, use_stop, use_stem, len(STOPWORDS))


# =========================================================
# TAB: PREPROCESSING (CARD UI)
# =========================================================
with tabs[1]:
    st.markdown('<div class="section-title">Hasil Preprocessing (Lebih Tajam)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="small-note">Filter ketat aktif: token minimal 3 huruf, singkatan/typo diperbaiki, clitic dibuang, noise ditendang.</div>',
        unsafe_allow_html=True
    )

    try:
        df_u = df_prep.drop_duplicates(subset=["_raw"])
    except Exception:
        df_u = df_prep

    n = min(preview_rows, len(df_u)) if len(df_u) else 0
    samp = df_u.sample(n=n, random_state=42).reset_index(drop=True) if n > 0 else df_u.head(0)

    def _clip_list(x, k=36):
        try:
            x = list(x)
        except Exception:
            return ""
        if len(x) > k:
            return ", ".join(x[:k]) + " ..."
        return ", ".join(x)

    for i in range(len(samp)):
        raw = str(samp.loc[i, "_raw"])
        clean = str(samp.loc[i, "cleaning_casefold"])
        final = str(samp.loc[i, "final_text"])
        toks = samp.loc[i, "tokenizing"]
        norm = samp.loc[i, "normalizing"]
        stopv = samp.loc[i, "stopword_removal"]
        stemv = samp.loc[i, "stemming"]

        st.markdown(
            f"""
            <div class="card">
              <div style="margin-bottom:10px;">
                <span class="badge">Sample #{i+1}</span>
                <span class="badge">Final tokens: {len(final.split())}</span>
              </div>
              <h4>Final Text</h4>
              <div class="mono">{final}</div>
              <details style="margin-top:10px;">
                <summary style="color: rgba(226,232,240,0.95); font-weight: 800;">Detail tahap</summary>
                <div style="height:10px;"></div>
                <div class="mono"><b>Raw:</b> {raw}</div>
                <div class="mono"><b>Cleaning:</b> {clean}</div>
                <div class="mono"><b>Tokenizing:</b> {_clip_list(toks)}</div>
                <div class="mono"><b>Normalizing:</b> {_clip_list(norm)}</div>
                <div class="mono"><b>Stopword removal:</b> {_clip_list(stopv)}</div>
                <div class="mono"><b>Stemming:</b> {_clip_list(stemv)}</div>
              </details>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# TF-IDF + ELBOW + KMEANS
# =========================================================
@st.cache_data(show_spinner=True)
def tfidf_and_models(
    _final_text_series: pd.Series,
    max_features_in: int,
    ngram_in: Tuple[int, int],
    min_df_in: int,
    max_df_in: float,
    random_state_in: int,
    k_final_in: int,
):
    vec = TfidfVectorizer(
        max_features=max_features_in,
        ngram_range=ngram_in,
        min_df=min_df_in,
        max_df=max_df_in,
        sublinear_tf=True,
    )
    X = vec.fit_transform(_final_text_series.values)

    ks = list(range(2, 9))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state_in, n_init=20)
        km.fit(X)
        inertias.append(float(km.inertia_))

    km_final = KMeans(n_clusters=k_final_in, random_state=random_state_in, n_init=20)
    labels = km_final.fit_predict(X)

    return vec, X, ks, inertias, km_final, labels

vectorizer, X_tfidf, ks, inertias, kmeans_model, labels = tfidf_and_models(
    df_prep["final_text"], int(max_features), ngram, int(min_df), float(max_df), int(random_state), int(k_final)
)

df_clustered = df_prep.copy()
df_clustered["cluster_raw"] = labels

@st.cache_data(show_spinner=False)
def compute_pca(_X):
    svd = TruncatedSVD(n_components=2, random_state=42)
    X2 = svd.fit_transform(_X)
    return X2, getattr(svd, "explained_variance_ratio_", np.array([0.0, 0.0]))

X2, var_ratio = compute_pca(X_tfidf)
df_clustered["pca1"] = X2[:, 0]
df_clustered["pca2"] = X2[:, 1]

feature_names = vectorizer.get_feature_names_out()

top_terms_unique = {}
for c in sorted(np.unique(labels)):
    top_terms_unique[c] = top_terms_unique_per_cluster(
        X_tfidf, df_clustered["cluster_raw"].values, feature_names, c, topn=25
    )

cluster_to_theme = {}
for c in sorted(top_terms_unique.keys()):
    theme, _scores = assign_theme_for_cluster(top_terms_unique[c])
    cluster_to_theme[c] = theme

remap = remap_clusters_to_fixed_ids(cluster_to_theme)
df_clustered["cluster"] = df_clustered["cluster_raw"].map(remap).astype(int)

inv_remap = {new: old for old, new in remap.items()}
mean_top_terms = top_terms_per_cluster(X_tfidf, df_clustered["cluster_raw"].values, feature_names, topn=25)

fixed_mean_top_terms = {}
fixed_unique_top_terms = {}
for new_c in [0, 1, 2]:
    old_c = inv_remap.get(new_c, new_c)
    fixed_mean_top_terms[new_c] = mean_top_terms.get(old_c, [])
    fixed_unique_top_terms[new_c] = top_terms_unique.get(old_c, [])


# =========================================================
# TAB: TF-IDF & ELBOW
# =========================================================
with tabs[2]:
    st.markdown('<div class="section-title">TF-IDF & Elbow</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        mean_vec = np.asarray(X_tfidf.mean(axis=0)).ravel()
        top_n = 20
        top_idx = mean_vec.argsort()[::-1][:top_n]
        df_top = pd.DataFrame({"term": feature_names[top_idx], "mean_tfidf": mean_vec[top_idx]}).sort_values("mean_tfidf", ascending=True)
        fig_top = px.bar(df_top, x="mean_tfidf", y="term", orientation="h", title="Top Terms Global (Mean TF-IDF)")
        fig_top.update_layout(height=520, margin=dict(l=10, r=10, t=55, b=10), showlegend=False)
        st.plotly_chart(fig_top, use_container_width=True)

    with c2:
        df_elbow = pd.DataFrame({"K": ks, "Inertia": inertias})
        fig_elbow = px.line(df_elbow, x="K", y="Inertia", markers=True, title="Elbow Method")
        fig_elbow.add_vline(x=3, line_width=2, line_dash="dash", annotation_text="K=3", annotation_position="top left")
        fig_elbow.update_layout(height=520, margin=dict(l=10, r=10, t=55, b=10), xaxis=dict(dtick=1))
        st.plotly_chart(fig_elbow, use_container_width=True)


# =========================================================
# TAB: KMEANS & PCA
# =========================================================
with tabs[3]:
    st.markdown('<div class="section-title">K-Means & PCA (2D)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="small-note">Variansi: PC1={var_ratio[0]:.2%}, PC2={var_ratio[1]:.2%}</div>', unsafe_allow_html=True)

    ctrl1, ctrl2 = st.columns([1, 2], gap="large")
    with ctrl1:
        cluster_filter = st.multiselect("Filter Cluster", options=[0, 1, 2], default=[0, 1, 2], key="ms_pca_cluster")
    with ctrl2:
        q = st.text_input("Cari kata di ulasan (opsional)", value="", placeholder="contoh: parkir, wifi, ramah", key="tx_pca_q")

    view_df = df_clustered[df_clustered["cluster"].isin(cluster_filter)].copy()
    if q.strip():
        view_df = view_df[view_df["_raw"].astype(str).str.contains(q.strip(), case=False, na=False)].copy()

    hover_cols = [c for c in ["place_name", "rating_place", "rating_review", "_raw"] if c in view_df.columns]
    fig = px.scatter(view_df, x="pca1", y="pca2", color=view_df["cluster"].astype(str), hover_data=hover_cols,
                     title="PCA 2D - Sebaran Ulasan per Cluster", labels={"color": "Cluster"})
    fig.update_traces(marker=dict(size=8, opacity=0.82))
    fig.update_layout(height=680, legend_title_text="Cluster")
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TAB: WORDCLOUD & KARAKTERISTIK
# =========================================================
def top_terms_df(term_list, topn=18):
    df_t = pd.DataFrame(term_list[:topn], columns=["term", "score"]).sort_values("score", ascending=True)
    return df_t

with tabs[4]:
    st.markdown('<div class="section-title">Wordcloud & Karakteristik</div>', unsafe_allow_html=True)

    pick = st.radio(
        "Pilih Cluster",
        options=[0, 1, 2],
        horizontal=True,
        format_func=lambda c: f"Cluster {c} — {['Pelayanan','Suasana & Fasilitas','Rasa Makan & Minum'][c]}",
        key="radio_cluster_wordcloud"
    )

    left, right = st.columns([1, 1], gap="large")

    with left:
        mode_terms = st.radio("Mode term", options=["Unik (disarankan)", "Mean TF-IDF (standar)"], horizontal=True, index=0)
        if mode_terms.startswith("Unik"):
            df_terms = top_terms_df(fixed_unique_top_terms[pick], topn=18)
            xlab = "Skor Unik"
        else:
            df_terms = top_terms_df(fixed_mean_top_terms[pick], topn=18)
            xlab = "Mean TF-IDF"
        fig_terms = px.bar(df_terms, x="score", y="term", orientation="h", title="Term Dominan di Cluster",
                           labels={"score": xlab, "term": "Term"})
        fig_terms.update_layout(height=560, margin=dict(l=10, r=10, t=55, b=10), showlegend=False)
        st.plotly_chart(fig_terms, use_container_width=True)

    with right:
        texts = df_clustered.loc[df_clustered["cluster"] == pick, "final_text"].astype(str)
        joined = " ".join(texts.tolist()).strip()
        if not joined:
            st.info("Tidak ada data pada cluster ini.")
        else:
            wc = WordCloud(width=1200, height=720, background_color="white", collocations=False, max_words=280,
                           prefer_horizontal=0.92, repeat=False, min_font_size=6).generate(joined)
            fig_wc, ax = plt.subplots(figsize=(9.2, 5.8))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc, use_container_width=True)


# =========================================================
# TAB: DOWNLOAD
# =========================================================
with tabs[5]:
    st.markdown('<div class="section-title">Download Hasil</div>', unsafe_allow_html=True)
    out_cols = []
    for col in ["place_name", "place_id", "rating_place", "rating_review", "date", "user", "address"]:
        if col in df_clustered.columns:
            out_cols.append(col)
    out_cols += ["_raw", "cleaning_casefold", "final_text", "cluster"]

    out_df = df_clustered[out_cols].copy().rename(columns={"_raw": "review_text_raw"})
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV (hasil preprocessing + cluster)",
        data=csv_bytes,
        file_name="hasil_preprocessing_cluster.csv",
        mime="text/csv"
    )