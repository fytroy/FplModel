"""Download data from the official Fantasy Premier League API and the vaastav GitHub archive.

This is the only file that talks to the internet.
"""
import gzip
import time

import requests

BASE_URL = "https://fantasy.premierleague.com/api"
NTFY_URL = "https://ntfy.sh"
TOKEN_URL = "https://account.premierleague.com/as/token"
FPL_CLIENT_ID = "bfcbaf69-aade-4c1b-8f00-c1cb8a193030"     # the FPL website's public login client
VAASTAV_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
# FPL sometimes rejects requests that don't look like they come from a browser.
HEADERS = {"User-Agent": "Mozilla/5.0 (fpl-model personal project)"}

# One shared session keeps the connection open between requests, instead of
# doing a fresh secure handshake (~1s) for every one of the ~700 players.
_session = requests.Session()
_session.headers.update(HEADERS)


def get_json(url: str, retries: int = 3, timeout: int = 30):
    """Fetch a URL and return its JSON, retrying a few times on failure."""
    return _with_retries(lambda: _get_once(url, timeout).json(), retries)


def get_bytes(url: str, retries: int = 3, timeout: int = 60) -> bytes:
    """Fetch a URL and return the raw file contents, retrying a few times on failure."""
    return _with_retries(lambda: _get_once(url, timeout).content, retries)


def _with_retries(fetch, retries: int):
    for attempt in range(1, retries):
        try:
            return fetch()
        except requests.RequestException as error:
            wait = 2 * attempt
            print(f"  Request failed ({error}); retrying in {wait}s...", flush=True)
            time.sleep(wait)
    return fetch()  # last try: let any error stop the script


def _get_once(url: str, timeout: int) -> requests.Response:
    response = _session.get(url, timeout=timeout)
    response.raise_for_status()
    return response


def fetch_bootstrap() -> dict:
    """All players, teams, gameweeks and positions in one big response."""
    return get_json(f"{BASE_URL}/bootstrap-static/")


def fetch_fixtures() -> list:
    """Every fixture of the season."""
    return get_json(f"{BASE_URL}/fixtures/")


def fetch_player_history(player_id: int) -> dict:
    """One player's match-by-match history this season, plus upcoming fixtures."""
    return get_json(f"{BASE_URL}/element-summary/{player_id}/")


def send_ntfy(topic: str, title: str, message: str, priority: int = 3, tags=()) -> None:
    """Push a phone notification to an ntfy topic (priority 1-5; 3 is normal)."""
    def post():
        response = _session.post(NTFY_URL, json={"topic": topic, "title": title, "message": message,
                                                 "priority": priority, "tags": list(tags)}, timeout=30)
        response.raise_for_status()
    _with_retries(post, 3)


def refresh_access_token(refresh_token: str) -> dict:
    """Swap a refresh token for an access token. FPL replaces the refresh token on every
    use, so this is never retried: a lost reply would burn the only valid one."""
    response = requests.post(TOKEN_URL, timeout=30, headers={"Accept": "application/json", **HEADERS},
                             data={"grant_type": "refresh_token", "refresh_token": refresh_token,
                                   "client_id": FPL_CLIENT_ID})
    response.raise_for_status()
    return response.json()


def _account_headers(access_token: str, page: str) -> dict:
    return {"X-API-Authorization": f"Bearer {access_token}", "Accept": "application/json",
            "Origin": "https://fantasy.premierleague.com",
            "Referer": f"https://fantasy.premierleague.com/{page}"}


def fetch_my_team(team_id: int, access_token: str) -> dict:
    """Your squad with real selling prices, bank, free transfers and chips (login needed)."""
    response = _session.get(f"{BASE_URL}/my-team/{team_id}/", timeout=30,
                            headers=_account_headers(access_token, "my-team"))
    response.raise_for_status()
    return response.json()


def post_transfers(payload: dict, access_token: str) -> None:
    """Make transfers. Changes your real team, so it is never retried automatically."""
    response = _session.post(f"{BASE_URL}/transfers/", json=payload, timeout=30,
                             headers=_account_headers(access_token, "transfers"))
    response.raise_for_status()


def post_picks(team_id: int, payload: dict, access_token: str) -> None:
    """Save the starting XI, bench order, captain and vice-captain."""
    response = _session.post(f"{BASE_URL}/my-team/{team_id}/", json=payload, timeout=30,
                             headers=_account_headers(access_token, "my-team"))
    response.raise_for_status()


def fetch_vaastav_csv(season: str, path: str) -> bytes:
    """One CSV from the vaastav archive, e.g. ('2025-26', 'gws/merged_gw.csv')."""
    return get_bytes(f"{VAASTAV_URL}/{season}/{path}")


GITHUB_API = "https://api.github.com/repos/vaastav/Fantasy-Premier-League"
GITHUB_RAW = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League"
WAYBACK = "https://web.archive.org"


def fetch_entry(team_id: int) -> dict:
    """A manager's team: name, current gameweek, bank. Public; no login."""
    return get_json(f"{BASE_URL}/entry/{team_id}/")


def fetch_entry_history(team_id: int) -> dict:
    """Gameweek-by-gameweek points, bank, transfers and hits, chips played, past seasons."""
    return get_json(f"{BASE_URL}/entry/{team_id}/history/")


def fetch_entry_transfers(team_id: int) -> list:
    """Every transfer this season with buy (element_in_cost) and sale (element_out_cost) prices."""
    return get_json(f"{BASE_URL}/entry/{team_id}/transfers/")


def fetch_entry_picks(team_id: int, gw: int) -> dict:
    """The 15 picks, captain and chip for one gameweek whose deadline has passed."""
    return get_json(f"{BASE_URL}/entry/{team_id}/event/{gw}/picks/")


def fetch_event_live(gw: int) -> dict:
    """Live match stats and points for all players in gameweek gw."""
    return get_json(f"{BASE_URL}/event/{gw}/live/")


def fetch_vaastav_commits(path: str) -> list[tuple[str, str]]:
    """(sha, ISO commit date) of every vaastav commit touching `path`, newest first."""
    commits = get_json(f"{GITHUB_API}/commits?path={path}&per_page=100")
    return [(c["sha"], c["commit"]["committer"]["date"]) for c in commits]


def fetch_vaastav_file_at(sha: str, path: str) -> bytes:
    """A vaastav file as it was at one commit, e.g. path 'data/2025-26/fixtures.csv'."""
    return get_bytes(f"{GITHUB_RAW}/{sha}/{path}")


def fetch_wayback_index(url: str, start: str, end: str) -> list[str]:
    """Internet Archive timestamps (YYYYmmddHHMMSS) of good copies of `url`, one per day."""
    rows = get_json(f"{WAYBACK}/cdx/search/cdx?url={url}&from={start}&to={end}"
                    "&output=json&filter=statuscode:200&collapse=timestamp:8")
    return [r[1] for r in rows[1:]]


def fetch_wayback(url: str, timestamp: str) -> bytes:
    """The archived copy of `url` at `timestamp`, exactly as originally served."""
    return maybe_gunzip(get_bytes(f"{WAYBACK}/web/{timestamp}id_/{url}"))


def maybe_gunzip(data: bytes) -> bytes:
    """The archive sometimes returns the original gzip-compressed body."""
    return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
