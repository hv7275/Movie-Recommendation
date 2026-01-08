import requests
import streamlit as st

# =============================
# CONFIG
# =============================
API_BASE = "https://movie-rec-466x.onrender.com"  # or "http://127.0.0.1:8000"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# =============================
# DARK THEME STYLES
# =============================
st.markdown("""
<style>

/* ---- Hide Streamlit Chrome ---- */
[data-testid="stStatusWidget"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
.stDeployButton {
    display: none !important;
}

/* ---- App Background ---- */
.stApp {
    background: radial-gradient(circle at top, #0f172a 0%, #020617 65%);
    color: #e5e7eb;
}

/* ---- Main Container ---- */
.block-container {
    max-width: 1500px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

/* ---- Typography ---- */
h1, h2, h3, h4 {
    color: #f9fafb;
}

.small-muted {
    color: #9ca3af;
    font-size: 0.9rem;
}

/* ---- Inputs ---- */
input, textarea {
    background-color: #020617 !important;
    color: #f9fafb !important;
}

/* ---- Movie Card ---- */
.movie-card {
    background: linear-gradient(180deg, #020617, #020617cc);
    border-radius: 16px;
    padding: 10px;
    transition: all 0.25s ease;
}

.movie-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.6);
}

/* ---- Poster ---- */
.movie-poster img {
    border-radius: 12px;
}

/* ---- Title ---- */
.movie-title {
    margin-top: 8px;
    font-size: 0.9rem;
    font-weight: 600;
    line-height: 1.2rem;
    height: 2.4rem;
    overflow: hidden;
    color: #f3f4f6;
    text-align: center;
}

/* ---- Buttons ---- */
.stButton > button {
    width: 100%;
    border-radius: 10px;
    background: #2563eb;
    color: white;
    border: none;
    margin-top: 6px;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #1d4ed8;
}

/* ---- Detail Card ---- */
.detail-card {
    border-radius: 18px;
    padding: 18px;
    background: linear-gradient(180deg, #020617, #020617cc);
    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
}

/* ---- Inline Suggestions ---- */
.suggestion-btn button {
    text-align: left !important;
    background: #020617 !important;
    border: 1px solid #1f2933 !important;
}

</style>
""", unsafe_allow_html=True)

# =============================
# STATE + ROUTING
# =============================
if "view" not in st.session_state:
    st.session_state.view = "home"
if "selected_tmdb_id" not in st.session_state:
    st.session_state.selected_tmdb_id = None

qp_view = st.query_params.get("view")
qp_id = st.query_params.get("id")

if qp_view in ("home", "details"):
    st.session_state.view = qp_view

if qp_id:
    try:
        st.session_state.selected_tmdb_id = int(qp_id)
        st.session_state.view = "details"
    except:
        pass


def goto_home():
    st.session_state.view = "home"
    st.query_params["view"] = "home"
    if "id" in st.query_params:
        del st.query_params["id"]
    st.rerun()


def goto_details(tmdb_id: int):
    st.session_state.view = "details"
    st.session_state.selected_tmdb_id = int(tmdb_id)
    st.query_params["view"] = "details"
    st.query_params["id"] = str(int(tmdb_id))
    st.rerun()


# =============================
# API
# =============================
@st.cache_data(ttl=30)
def api_get_json(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=25)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}"
        return r.json(), None
    except Exception as e:
        return None, str(e)


# =============================
# UI COMPONENTS
# =============================
def poster_grid(cards, cols=6, key_prefix="grid"):
    if not cards:
        st.info("No movies to show.")
        return

    rows = (len(cards) + cols - 1) // cols
    idx = 0

    for r in range(rows):
        colset = st.columns(cols, gap="medium")

        for c in range(cols):
            if idx >= len(cards):
                break

            m = cards[idx]
            idx += 1

            tmdb_id = m.get("tmdb_id")
            title = m.get("title", "Untitled")
            poster = m.get("poster_url")

            with colset[c]:
                st.markdown("<div class='movie-card'>", unsafe_allow_html=True)

                if poster:
                    st.markdown("<div class='movie-poster'>", unsafe_allow_html=True)
                    st.image(poster, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.write("🖼️ No poster")

                if st.button("Open", key=f"{key_prefix}_{r}_{c}_{tmdb_id}"):
                    if tmdb_id:
                        goto_details(tmdb_id)

                st.markdown(
                    f"<div class='movie-title'>{title}</div>",
                    unsafe_allow_html=True,
                )

                st.markdown("</div>", unsafe_allow_html=True)


def to_cards_from_tfidf_items(tfidf_items):
    cards = []
    for x in tfidf_items or []:
        tmdb = x.get("tmdb") or {}
        if tmdb.get("tmdb_id"):
            cards.append({
                "tmdb_id": tmdb["tmdb_id"],
                "title": tmdb.get("title") or x.get("title") or "Untitled",
                "poster_url": tmdb.get("poster_url"),
            })
    return cards


def parse_tmdb_search_to_cards(data, keyword: str, limit: int = 24):
    keyword_l = keyword.strip().lower()

    if isinstance(data, dict) and "results" in data:
        raw = data.get("results") or []
        raw_items = []
        for m in raw:
            if not m.get("title") or not m.get("id"):
                continue
            raw_items.append({
                "tmdb_id": int(m["id"]),
                "title": m["title"],
                "poster_url": f"{TMDB_IMG}{m['poster_path']}" if m.get("poster_path") else None,
                "release_date": m.get("release_date", ""),
            })

    else:
        return [], []

    matched = [x for x in raw_items if keyword_l in x["title"].lower()]
    final_list = matched if matched else raw_items

    suggestions = []
    for x in final_list[:8]:
        year = (x.get("release_date") or "")[:4]
        label = f"{x['title']} ({year})" if year else x["title"]
        suggestions.append((label, x["tmdb_id"]))

    cards = [{"tmdb_id": x["tmdb_id"], "title": x["title"], "poster_url": x["poster_url"]}
             for x in final_list[:limit]]

    return suggestions, cards


# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.markdown("## 🎬 Browse")

    if st.button("🏠 Home", use_container_width=True):
        goto_home()

    st.markdown("---")
    st.markdown("### 🔥 Home Feed")

    home_category = st.selectbox(
        "Category",
        ["trending", "popular", "top_rated", "now_playing", "upcoming"],
        index=0,
    )

    grid_cols = st.slider("Grid Density", 4, 8, 6)


# =============================
# HEADER
# =============================
st.title("🎬 Movie Recommender")
st.markdown(
    "<div class='small-muted'>Type → click suggestion → open details</div>",
    unsafe_allow_html=True,
)
st.divider()


# =============================
# HOME VIEW
# =============================
if st.session_state.view == "home":

    typed = st.text_input(
        "Search by movie title",
        placeholder="Avengers, Batman, Interstellar...",
        key="search_box"
    )

    if typed.strip() and len(typed) >= 2:
        data, err = api_get_json("/tmdb/search", params={"query": typed})

        if not err and data:
            suggestions, cards = parse_tmdb_search_to_cards(data, typed)

            # ---- INLINE SUGGESTIONS ----
            if suggestions:
                st.markdown("**Suggestions**")
                for label, tmdb_id in suggestions:
                    if st.button(label, key=f"sugg_{tmdb_id}", use_container_width=True):
                        goto_details(tmdb_id)

            st.markdown("### Results")
            poster_grid(cards, cols=grid_cols, key_prefix="search")

        else:
            st.error("Search failed.")

        st.stop()

    st.markdown(f"### 🔥 {home_category.replace('_',' ').title()}")

    home_cards, err = api_get_json("/home", params={"category": home_category, "limit": 24})

    if err or not home_cards:
        st.error("Failed to load home feed.")
    else:
        poster_grid(home_cards, cols=grid_cols, key_prefix="home")


# =============================
# DETAILS VIEW
# =============================
elif st.session_state.view == "details":

    tmdb_id = st.session_state.selected_tmdb_id

    if not tmdb_id:
        st.warning("No movie selected.")
        if st.button("← Back"):
            goto_home()
        st.stop()

    a, b = st.columns([3, 1])
    with a:
        st.markdown("### 🎥 Movie Details")
    with b:
        if st.button("← Back to Home"):
            goto_home()

    data, err = api_get_json(f"/movie/id/{tmdb_id}")

    if err or not data:
        st.error("Could not load movie details.")
        st.stop()

    left, right = st.columns([1, 2.5], gap="large")

    with left:
        st.markdown("<div class='detail-card'>", unsafe_allow_html=True)
        if data.get("poster_url"):
            st.image(data["poster_url"], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='detail-card'>", unsafe_allow_html=True)
        st.markdown(f"## {data.get('title','')}")
        st.markdown(
            f"<div class='small-muted'>Release: {data.get('release_date','-')}</div>",
            unsafe_allow_html=True
        )
        genres = ", ".join([g["name"] for g in data.get("genres", [])]) or "-"
        st.markdown(
            f"<div class='small-muted'>Genres: {genres}</div>",
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.markdown("### Overview")
        st.write(data.get("overview") or "No overview available.")
        st.markdown("</div>", unsafe_allow_html=True)

    if data.get("backdrop_url"):
        st.markdown("### Backdrop")
        st.image(data["backdrop_url"], use_container_width=True)

    st.divider()
    st.markdown("### 🎯 Recommendations")

    title = data.get("title", "").strip()

    if title:
        bundle, err2 = api_get_json("/movie/search", params={
            "query": title,
            "tfidf_top_n": 12,
            "genre_limit": 12
        })

        if not err2 and bundle:
            st.markdown("#### 🔍 Similar Movies")
            poster_grid(
                to_cards_from_tfidf_items(bundle.get("tfidf_recommendations")),
                cols=grid_cols,
                key_prefix="tfidf",
            )

            st.markdown("#### 🎭 Genre Based")
            poster_grid(
                bundle.get("genre_recommendations", []),
                cols=grid_cols,
                key_prefix="genre",
            )
        else:
            st.info("No recommendations available.")