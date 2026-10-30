"""Tests de la configuración de la cocina (docs/env_design.md, secciones 3.1 y 3.8)."""

import numpy as np

from envs.kitchen_config import (
    DEFAULT_RANGES, RUSH_LENGTH, KitchenParams, classify, sample_kitchen,
)


def test_classify_thresholds():
    assert classify(0.99) == "raw"
    assert classify(1.0) == "ok"
    assert classify(1.29) == "ok"
    assert classify(1.3) == "burnt"


def _random_kitchens(n=200):
    rng = np.random.default_rng(0)
    return [sample_kitchen(rng, 200, "random") for _ in range(n)]


def test_random_kitchen_respects_ranges():
    for k in _random_kitchens():
        assert k.n_stations in (2, 3)
        assert 0.06 <= k.arrival_prob <= 0.16
        assert 0.05 <= k.sigma_obs <= 0.25
        assert len(k.rushes) <= 2
        for start, end in k.rushes:
            assert end - start == RUSH_LENGTH
            assert 0 <= start and end <= 200


def test_random_kitchen_covers_the_ranges():
    kitchens = _random_kitchens()
    assert {k.n_stations for k in kitchens} == {2, 3}
    assert {len(k.rushes) for k in kitchens} == {0, 1, 2}


def test_rushes_never_overlap():
    for k in _random_kitchens():
        if len(k.rushes) == 2:
            (s0, e0), (s1, e1) = k.rushes
            assert e0 <= s1 or e1 <= s0


def test_presets_are_fixed():
    a = sample_kitchen(np.random.default_rng(1), 200, "hard")
    b = sample_kitchen(np.random.default_rng(2), 200, "hard")
    for k in (a, b):
        assert k.n_stations == 2
        assert k.arrival_prob == 0.16
        assert k.sigma_obs == 0.25
        assert len(k.rushes) == 2


def test_dict_spec_overrides_medium():
    k = sample_kitchen(np.random.default_rng(0), 200, {"sigma_obs": 0.1})
    assert k.sigma_obs == 0.1
    assert k.n_stations == 2 and k.arrival_prob == 0.10 and len(k.rushes) == 1


def test_same_rng_consumption_for_any_spec():
    draws = []
    for spec in ("random", "easy", {"sigma_obs": 0.1}):
        rng = np.random.default_rng(7)
        sample_kitchen(rng, 200, spec)
        draws.append(rng.random())
    assert draws[0] == draws[1] == draws[2]


def test_arrival_prob_during_rush():
    k = KitchenParams(2, 0.5, 0.1, ((10, 30),))
    assert k.arrival_prob_at(5) == 0.5
    assert k.arrival_prob_at(10) == 1.0
    assert k.arrival_prob_at(30) == 0.5


def test_default_ranges_match_design():
    assert DEFAULT_RANGES["n_stations"] == (2, 3)
    assert DEFAULT_RANGES["n_rushes"] == (0, 2)
