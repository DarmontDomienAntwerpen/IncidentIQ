import pytest

from app import clean_tx, get_vid_id


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=7OH5FEWWM_k",
    "https://www.youtube.com/watch?v=7OH5FEWWM_k&t=42s",
    "https://youtu.be/7OH5FEWWM_k",
    "https://youtu.be/7OH5FEWWM_k?si=abc",
])
def test_get_vid_id_supports_common_url_formats(url):
    assert get_vid_id(url) == "7OH5FEWWM_k"


def test_get_vid_id_rejects_non_youtube_url():
    with pytest.raises(ValueError):
        get_vid_id("https://example.com/video")


def test_clean_tx_removes_sound_annotations():
    assert clean_tx("[Music] fire on floor 7 [Applause]") == "fire on floor 7"


def test_clean_tx_drops_empty_timestamps_and_collapses_whitespace():
    assert clean_tx("[00:01]  [00:02] water   supply") == "[00:02] water supply"


def test_first_ts_reads_first_timestamp_and_sorts_unstamped_last():
    from app import first_ts
    assert first_ts("smoke [05:32] means [05:34]") == 332
    assert first_ts("no timestamp") > first_ts("[59:59]")


def test_plain_text_strips_markdown_emphasis_and_headers():
    from app import plain_text
    assert plain_text("**SCENARIO BRIEF**\n## Title\n- Fire: **5th** floor") == "SCENARIO BRIEF\nTitle\n- Fire: 5th floor"
