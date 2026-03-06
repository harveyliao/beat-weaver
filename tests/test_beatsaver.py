import io
import zipfile

import requests

from beat_weaver.sources.beatsaver import BeatSaverClient


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, headers: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}
        self.url = "https://api.beatsaver.com/search/text/0?sortOrder=Rating"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} error for url: {self.url}",
                response=self,
            )

    def json(self) -> dict:
        return self._payload


def test_search_maps_retries_429_then_succeeds(monkeypatch):
    client = BeatSaverClient()
    sleeps: list[float] = []
    responses = [
        _FakeResponse(429, headers={"Retry-After": "1"}),
        _FakeResponse(
            200,
            payload={
                "docs": [
                    {
                        "id": "abc",
                        "automapper": False,
                        "stats": {"score": 0.9, "upvotes": 10},
                        "versions": [{"hash": "hash1", "downloadURL": "/download/hash1"}],
                    }
                ],
                "info": {"pages": 1},
            },
        ),
    ]

    def fake_get(url, params=None):
        return responses.pop(0)

    monkeypatch.setattr(client.session, "get", fake_get)
    monkeypatch.setattr("beat_weaver.sources.beatsaver.time.sleep", sleeps.append)

    docs = list(client.search_maps())

    assert len(docs) == 1
    assert docs[0]["id"] == "abc"
    assert sleeps == [1.0]


def test_search_maps_raises_after_max_429_retries(monkeypatch):
    client = BeatSaverClient()
    sleeps: list[float] = []

    def fake_get(url, params=None):
        return _FakeResponse(429)

    monkeypatch.setattr(client.session, "get", fake_get)
    monkeypatch.setattr("beat_weaver.sources.beatsaver.time.sleep", sleeps.append)

    try:
        next(client.search_maps(max_pages=1))
        raise AssertionError("expected HTTPError")
    except requests.HTTPError as exc:
        assert exc.response is not None
        assert exc.response.status_code == 429

    assert sleeps == [5.0, 10.0, 20.0, 40.0]


def test_download_map_retries_429_then_succeeds(monkeypatch, tmp_path):
    client = BeatSaverClient()
    sleeps: list[float] = []
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("Info.dat", "{}")
    zip_bytes = zip_buffer.getvalue()
    responses = [
        _FakeResponse(429, headers={"Retry-After": "2"}),
        _FakeResponse(200),
    ]
    responses[1].content = zip_bytes

    def fake_get(url, headers=None):
        return responses.pop(0)

    monkeypatch.setattr("beat_weaver.sources.beatsaver.requests.get", fake_get)
    monkeypatch.setattr("beat_weaver.sources.beatsaver.time.sleep", sleeps.append)

    map_info = {
        "id": "abc",
        "versions": [{"hash": "hash1", "downloadURL": "/download/hash1"}],
    }
    result = client.download_map(map_info, tmp_path)

    assert result == tmp_path / "hash1"
    assert result.exists()
    assert (result / "_beatsaver_meta.json").exists()
    assert sleeps == [2.0]
