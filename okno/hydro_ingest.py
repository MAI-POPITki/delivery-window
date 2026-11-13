"""Read-only collectors for public Roshydromet products.

Meteo-DV requests are the requests of its public guest interface (user guide,
2024). This is not a supported API contract: schema errors leave the last good
snapshot intact. River-reach bulletins never become point measurements.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urljoin, urlparse, urlunparse
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import requests

from . import official_hydro as H

BASE = "https://meteo-dv.ru/hydro_dfo/"
HOSTS = {"meteo-dv.ru", "ykuthydromet.ru", "sevmeteo.ru", "www.sevmeteo.ru",
         "rcpod.ru", "www.rcpod.ru", "www.meteorf.gov.ru", "meteorf.gov.ru",
         "meteoinfo.ru", "mpr.meteoinfo.ru"}
FEEDS = ["https://www.meteorf.gov.ru/press/news/rss/", "https://www.meteorf.gov.ru/press/polovod2026/", "https://www.rcpod.ru/news/",
         "https://www.sevmeteo.ru/weather/gidro/"]
YAKUTIA = "https://ykuthydromet.ru/reki/"
MONTHS = {name: i+1 for i, name in enumerate(("январ", "феврал", "март", "апрел", "мая", "июн",
                                            "июл", "август", "сентябр", "октябр", "ноябр", "декабр"))}
REGION_PATTERNS = {"yakutia": r"якут|\bсаха\b", "yanao": r"ямало|янао",
                   "krasnoyarsk": r"краснояр|среднесибир", "irkutsk": r"иркут",
                   "arkhangelsk": r"архангел", "magadan": r"магадан|колымское угмс",
                   "chukotka": r"чукот"}
# Verified official starting documents; discovery above also follows new issues.
SEEDS = [
    "https://www.meteorf.gov.ru/press/news/42860/",
    "https://www.meteorf.gov.ru/press/news/43823/",
    "https://www.meteorf.gov.ru/press/polovod2026/45193/",
    "https://www.meteorf.gov.ru/special/press/news/39623/",
    "https://www.rcpod.ru/news/1930/", "https://www.rcpod.ru/news/2245/",
    "https://www.rcpod.ru/news/2265/", "https://www.rcpod.ru/news/2269/",
    "https://www.rcpod.ru/news/2277/",
    "https://www.meteorf.gov.ru/press/news/45140/",
    "https://www.meteorf.gov.ru/press/polovod2026/43226/",
    "https://www.sevmeteo.ru/weather/gidro/19018/",
    "https://meteoinfo.ru/novosti/22720-gidrologicheskij-obzor-10-noyabrya-2026-g",
]


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _get(url, directory, *, params=None):
    """No credentials, browser automation, paid data, or cross-site redirects."""
    for _ in range(4):
        parts = urlparse(url)
        if parts.hostname not in HOSTS or parts.scheme not in ("http", "https") or parts.username or parts.password:
            raise ValueError("Unexpected official-source URL")
        # Roshydromet's own RSS still publishes http links. Use the HTTPS
        # equivalent on the same verified host instead of discarding updates.
        url = urlunparse(parts._replace(scheme="https", fragment=""))
        r = requests.get(url, params=params, timeout=(6, 18), allow_redirects=False,
                         headers={"User-Agent": "OknoZavoza/1.0 (public hydrological observations; research)"})
        params = None
        if r.status_code in (301, 302, 303, 307, 308):
            url = urljoin(url, r.headers["Location"])
            continue
        r.raise_for_status()
        if not r.content or len(r.content) > 12_000_000:
            raise ValueError("Empty or oversized official-source response")
        digest = _digest(r.content)
        folder = Path(directory) / "raw"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{digest}.gz"
        if not path.exists():
            path.write_bytes(gzip.compress(r.content, mtime=0))
        # Most official pages use UTF-8; older UGMS pages also specify cp1251.
        if not r.encoding or r.encoding.lower() == "iso-8859-1":
            r.encoding = r.apparent_encoding
        return r, digest
    raise ValueError("Too many official-source redirects")


def parse_station_list(data, fetched_at, source_url, digest, points=None):
    if not isinstance(data, list) or any(not isinstance(r, dict) for r in data):
        raise ValueError("Meteo-DV station-list schema changed")
    points = H.registry() if points is None else points
    out, matched = [], []
    for row in data:
        point = H.match_station(row, points)
        if point is None:
            continue
        matched.append((point, row))
        distance = H.distance_km(point, row)
        note = f"Координаты источника отличаются от каталога на {distance:g} км." if distance and distance > 5 else ""
        if not row.get("last_l_dt"):
            continue
        value = H.observation(point, "water_level_cm", row.get("level_now"), row["last_l_dt"],
                              fetched_at, source_url, digest, note=note)
        if value:
            out.append(value)
    return out, matched


def parse_graph(data, point, fetched_at, source_url, digest):
    if not isinstance(data, dict) or not isinstance(data.get("levels"), list):
        raise ValueError("Meteo-DV graph schema changed")
    expected = (H.normalise(point["river"]), H.normalise(point["site"]))
    name = H.normalise(data.get("name", ""))
    if not all(part in name for part in expected):
        raise ValueError("Graph belongs to a different post")
    result = []
    # prev_y is drawn at THIS year's x-coordinates by the original website.
    # It must never be imported as new observations. Likewise, 'agk' is a
    # different instrument and cannot be silently mixed with the post series.
    for pair in data["levels"]:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Malformed graph point")
        row = H.observation(point, "water_level_cm", pair[1], pair[0], fetched_at,
                            source_url, digest)
        if row:
            result.append(row)
    return result


def parse_modal(data, point, fetched_at, source_url, digest):
    if not isinstance(data, dict):
        raise ValueError("Meteo-DV post-card schema changed")
    item = {**data, "code": point["station_code"]}
    if H.match_station(item, {point["point_id"]: point}) is None:
        return []
    result = []
    if data.get("level_last_date"):
        row = H.observation(point, "water_level_cm", data.get("level_last"), data["level_last_date"],
                            fetched_at, source_url, digest)
        if row:
            result.append(row)
    # 'level_current_date' and 'date_utc' are the requested display period,
    # not the date of the last ice measurement. Preserve an undated value for
    # inspection, but never feed it into a current forecast or training set.
    for metric, field, date_field in (("ice_cm", "ice_level_last", "ice_last_date"),
                                      ("snow_on_ice_cm", "snow_on_ice_last", "snow_on_ice_last_date")):
        observed = data.get(date_field)
        row = H.observation(point, metric, data.get(field), observed, fetched_at, source_url, digest,
                            quality="measured" if observed else "undated",
                            note="Дата замера не указана источником; значение не используется как текущее." if not observed else "")
        if row:
            result.append(row)
    return result


def _russian_date(text):
    match = re.search(r"\b(\d{1,2})\s+([а-яё]+)\s+(20\d{2})\b", text.lower())
    if match:
        day, month, year = match.groups()
        for name, number in MONTHS.items():
            if month.startswith(name):
                return datetime(int(year), number, int(day), 23, 59, 59, tzinfo=timezone.utc)
    match = re.search(r"\b(\d{2})[./](\d{2})[./](20\d{2})\b", text)
    if match:
        day, month, year = map(int, match.groups())
        return datetime(year, month, day, 23, 59, 59, tzinfo=timezone.utc)
    return None


def _regions(title, text):
    found = [k for k, pattern in REGION_PATTERNS.items() if re.search(pattern, title, re.I)]
    return found or [k for k, pattern in REGION_PATTERNS.items() if re.search(pattern, text, re.I)]


def _point_mentions(text, region, points):
    norm = H.normalise(text)
    return [p for p, v in points.items() if v["region"] == region and v["site"]
            and H.normalise(v["site"]) in norm and H.normalise(v["river"]) in norm]


def bulletin(title, text, published_at, url, digest, *, regions=None, points=None):
    if not published_at or not river_product(title):
        return []
    points = H.registry() if points is None else points
    regions = _regions(title, text) if regions is None else regions
    # Save the factual product, not navigation, surrounding advertisements or
    # instructions on the website. Prose remains prose; no range is averaged.
    text = re.sub(r"\s+", " ", text).strip()
    result = []
    for key in regions:
        if key not in H.REGIONS:
            continue
        identity = f"{url}|{key}|{published_at}|{text}"
        result.append(dict(id=_digest(identity.encode())[:24], region=key, title=title,
                           text=text, published_at=H.stamp(published_at).isoformat(),
                           source_url=url, raw_sha256=digest, scope="river_reach",
                           point_ids=_point_mentions(text, key, points)))
    return result


def river_product(title):
    if re.search(r"(?:мор[еюя]|бухт[аеуы])\b", title, re.I):
        return False
    return not re.search(r"ледов\w* обстанов", title, re.I) or bool(re.search(r"рек|реч|озер|водоем", title, re.I))


def parse_yakutia(html, digest, *, points=None):
    soup = BeautifulSoup(html, "html.parser")
    posts = soup.select(".post[data-year][data-month][data-day]")
    if not posts:
        raise ValueError("Yakutia bulletin calendar schema changed")
    result = []
    for post in posts:
        body = post.select_one(".content")
        if body is None:
            continue
        year, month, day = (int(post[f"data-{key}"]) for key in ("year", "month", "day"))
        # data-month is 1-based on the official page (verified against dated text).
        published = datetime(year, month, day, 23, 59, 59, tzinfo=timezone.utc)
        result.extend(bulletin(f"Обзор водных объектов Якутии на {day:02}.{month:02}.{year}",
                               body.get_text(" ", strip=True), published, YAKUTIA,
                               digest, regions=["yakutia"], points=points))
    return result


def parse_article(html, url, digest, *, published_at=None, points=None):
    soup = BeautifulSoup(html, "html.parser")
    for node in soup.select("script,style,nav,header,footer,aside,.related,.news-list"):
        node.decompose()
    heading = soup.find("h1")
    if heading is None:
        raise ValueError("Official article has no title")
    title = heading.get_text(" ", strip=True)
    container = soup.select_one(".news-detail,.detail-text,.news_text,.news-text,.inside.layout")
    if container is None:
        container = heading.parent
        while container.parent and len(container.get_text(" ", strip=True)) < len(title)+150:
            container = container.parent
    text = container.get_text(" ", strip=True)
    # Ignore the page-wide navigation before the article and the related links.
    if title in text:
        text = text.split(title, 1)[1]
    for end in ("См. также:", "Рубрикатор", "Навигация и возможности"):
        text = text.split(end, 1)[0]
    if published_at is None:
        node = soup.select_one('meta[property="article:published_time"],time[datetime]')
        if node:
            published_at = node.get("content", node.get("datetime"))
        published_at = published_at or _russian_date(text[:250])
    return bulletin(title, text, published_at, url, digest, points=points)


def discover(body, base, *, limit=30):
    if "<rss" in body[:200] or "<feed" in body[:200]:
        root = ET.fromstring(body)
        result = []
        for item in root.findall(".//item"):
            title = item.findtext("title", "")
            if re.search(r"гидролог|ледов|толщин.*льда", title, re.I):
                published = item.findtext("pubDate")
                result.append((item.findtext("link", ""), parsedate_to_datetime(published) if published else None))
        return result[:limit]
    soup = BeautifulSoup(body, "html.parser")
    result = []
    for node in soup.select("a[href]"):
        url = urljoin(base, node["href"])
        label = node.get_text(" ", strip=True)
        if urlparse(url).hostname not in HOSTS:
            continue
        if re.search(r"гидролог|гидрометеорологических условий", label, re.I) and re.search(r"/\d+/?$", url):
            result.append((url, None))
    return list(dict.fromkeys(result))[:limit]


def merge_records(old, new):
    indexed = {}
    for row in old+new:
        row = dict(row)
        if "metric" in row:
            row["id"] = H.observation_id(row["source"], row["point_id"], row["metric"],
                                          row.get("observed_at"), row["value"], row["quality"])
        prior = indexed.get(row["id"])
        if prior:
            row = {**row, **{k: min(prior[k], row.get(k, prior[k])) for k in ("first_seen_at", "available_at") if k in prior}}
            if prior.get("note") and not row.get("note"):
                row["note"] = prior["note"]
        indexed[row["id"]] = row
    return sorted(indexed.values(), key=lambda r: r["id"])


@contextmanager
def update_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".update.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        # Recover only after the maximum update runtime, including a crashed
        # process. An active second worker leaves the shared snapshot alone.
        if datetime.now(timezone.utc).timestamp()-path.stat().st_mtime > 1800:
            path.unlink(missing_ok=True)
        yield False
        return
    try:
        os.close(fd)
        yield True
    finally:
        path.unlink(missing_ok=True)


def _write_snapshot(directory, payload):
    fd, name = tempfile.mkstemp(prefix="snapshot-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, directory / "snapshot.json")
    finally:
        Path(name).unlink(missing_ok=True)


def update(*, directory=None, bootstrap=False, history_pages=1, now=None):
    """Fetch each source independently; failures never erase stored observations."""
    directory = H.DIRECTORY if directory is None else Path(directory)
    now = datetime.now(timezone.utc) if now is None else H.stamp(now)
    with update_lock(directory) as locked:
        if not locked:
            return {"status": "already_running"}
        previous = H.load_snapshot(directory / "snapshot.json")
        points = H.registry()
        measured, bulletins, statuses, documents = [], [], [], []

        def get(url, params=None):
            response, digest = _get(url, directory, params=params)
            documents.append(dict(url=response.url, raw_sha256=digest, fetched_at=now.isoformat()))
            return response, digest

        dates = [now.date().isoformat()]
        if bootstrap:
            dates += ["2025-02-01", "2026-03-01"]
        matched = {}
        try:
            for day in dict.fromkeys(dates):
                if day > now.date().isoformat():
                    continue
                response, digest = get(BASE+"cgi/get_info.php", dict(regions="all", type=2,
                    date_time_type="Last", date_loc=day+" 00:00", time_offset=0))
                rows, matches = parse_station_list(response.json(), now, response.url, digest, points)
                measured.extend(rows)
                matched.update({p["point_id"]: p for p, _ in matches})
            statuses.append(dict(source="meteo_dv", status="ok", checked_at=now.isoformat(),
                                 url=BASE, matched_posts=len(matched)))
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            statuses.append(dict(source="meteo_dv", status="error", url=BASE, checked_at=now.isoformat(), error=str(exc)[:250]))
        if bootstrap:
            matched.update({p: v for p, v in points.items() if v["station_code"] is not None
                            and v["region"] in ("yakutia", "magadan", "chukotka")})

        def post_data(point):
            rows, errors = [], []
            for endpoint, parser in (("get_modal_info.php", parse_modal), ("get_hydro_graph.php", parse_graph)):
                try:
                    response, digest = get(BASE+"cgi/"+endpoint, dict(station_code=point["station_code"],
                        date_loc=now.date().isoformat()+" 00:00", time_offset=0))
                    data = response.json()
                    if not data:
                        continue
                    rows.extend(parser(data, point, now, response.url, digest))
                except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                    errors.append(f"{point['point_id']}: {endpoint}: {str(exc)[:120]}")
            return rows, errors
        with ThreadPoolExecutor(max_workers=4) as pool:
            for rows, errors in pool.map(post_data, matched.values()):
                measured.extend(rows)
                if errors:
                    statuses.append(dict(source="meteo_dv_card", status="error", error="; ".join(errors)))
        try:
            response, digest = get(YAKUTIA)
            bulletins.extend(parse_yakutia(response.text, digest, points=points))
            statuses.append(dict(source="yakutia_bulletins", status="ok", url=YAKUTIA, checked_at=now.isoformat()))
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            statuses.append(dict(source="yakutia_bulletins", status="error", url=YAKUTIA, error=str(exc)[:250]))

        articles = [(url, None) for url in SEEDS] if bootstrap else []
        feeds = list(FEEDS)
        if history_pages > 1:
            for page in range(2, min(30, history_pages)+1):
                feeds.extend([f"https://www.rcpod.ru/news/?PAGEN_1={page}",
                              f"https://www.sevmeteo.ru/weather/gidro/?PAGEN_1={page}",
                              f"https://www.meteorf.gov.ru/press/polovod2026/?PAGEN_1={page}"])
        for url in feeds:
            try:
                response, digest = get(url)
                articles.extend(discover(response.text, url))
                statuses.append(dict(source=urlparse(url).hostname, status="ok", url=url, checked_at=now.isoformat()))
            except (requests.RequestException, ValueError, ET.ParseError) as exc:
                statuses.append(dict(source=urlparse(url).hostname, status="error", url=url, error=str(exc)[:250]))

        def article(item):
            url, published = item
            try:
                response, digest = get(url)
                return parse_article(response.text, url, digest, published_at=published, points=points), None
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                return [], dict(source=urlparse(url).hostname, status="error", url=url, error=str(exc)[:250])
        # Preserve the explicit RSS publication timestamp if duplicate links occur.
        unique = dict(articles)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for rows, error in pool.map(article, unique.items()):
                bulletins.extend(r for r in rows if H.stamp(r["published_at"]).date() <= now.date())
                if error:
                    statuses.append(error)
        successful = bool(measured or bulletins)
        from .hydro_training import extract_dated_ice
        # Preserve exact dated values embedded in a bulletin as measurements;
        # the rest of its text remains river-reach context. This also backfills
        # verified old bulletins when the source is temporarily unreachable.
        for report in merge_records(previous.get("bulletins", []), bulletins):
            measured.extend(extract_dated_ice(report, now, points=points))
        payload = dict(schema_version=1, checked_at=now.isoformat(),
                       last_success_at=now.isoformat() if successful else previous.get("last_success_at"),
                       observations=merge_records(previous.get("observations", []), measured),
                       bulletins=merge_records([r for r in previous.get("bulletins", [])
                           if river_product(r["title"])], bulletins),
                       documents=merge_documents(previous.get("documents", []), documents),
                       sources=statuses,
                       policies=dict(measurements="Exact station index AND river/site match; UTC dates; original units.",
                                     undated_ice="Display only; source does not publish measurement date.",
                                     bulletins="River-reach context, never synthetic point targets.",
                                     availability="First seen by this project; no retroactive forecast inputs.",
                                     severnoe="Source attribution required; non-commercial research use."))
        _write_snapshot(directory, payload)
        return dict(status="ok" if successful else "unavailable", observations=len(payload["observations"]),
                    bulletins=len(payload["bulletins"]), errors=sum(s["status"] == "error" for s in statuses))


def merge_documents(old, new):
    return sorted({(r["url"], r["raw_sha256"]): r for r in old+new}.values(), key=lambda r: (r["url"], r["raw_sha256"]))
