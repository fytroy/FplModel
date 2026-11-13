"""Offline tests for fpl/api.py helpers (no network)."""
import gzip

from fpl import api


def test_maybe_gunzip_unpacks_gzip_and_leaves_plain_bytes():
    assert api.maybe_gunzip(gzip.compress(b'{"a": 1}')) == b'{"a": 1}'
    assert api.maybe_gunzip(b'{"a": 1}') == b'{"a": 1}'


def test_send_ntfy_posts_json_to_the_topic(monkeypatch):
    sent = []

    class FakeResponse:
        def raise_for_status(self):
            pass

    class FakeSession:
        def post(self, url, json, timeout):
            sent.append((url, json))
            return FakeResponse()

    monkeypatch.setattr(api, "_session", FakeSession())
    api.send_ntfy("my-topic", "FPL GW8 advice", "Captain: Haaland → Salah", priority=4, tags=["soccer"])
    assert sent == [("https://ntfy.sh", {"topic": "my-topic", "title": "FPL GW8 advice",
                                         "message": "Captain: Haaland → Salah", "priority": 4,
                                         "tags": ["soccer"]})]
