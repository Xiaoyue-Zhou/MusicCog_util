import pytest
from dissonance_tool import *


class TestDissonanceCurve(object):
    def test_dissonance_curve_sample1(self):
        assert round(dissonance_curve(1), 5) == pytest.approx(0.03966)

    def test_dissonance_curve_sample2(self):
        assert round(dissonance_curve(1.3), 5) == pytest.approx(0.00608)


class TestGetFrequency(object):
    def test_on_c4(self):
        assert get_frequency(60) == pytest.approx(261.6)

    def test_on_a4(self):
        assert get_frequency(69) == pytest.approx(440.0)


class TestCalcDissonance(object):
    def test_on_c_major(self):
        assert round(calc_dissonance([60, 64, 67], n_harmonics=11), 3) == pytest.approx(0.122)

    def test_on_c_minor(self):
        assert round(calc_dissonance([60, 63, 67], n_harmonics=5), 3) == pytest.approx(0.092)

    def test_on_c_diminish(self):
        assert round(calc_dissonance([60, 63, 66], n_harmonics=11), 3) == pytest.approx(0.202)


@pytest.fixture()
def melody_sample():
    pitches = [60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60]
    durations = [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2]
    yield pitches, durations


def test_key_estimate(melody_sample):
    pitches, durations = melody_sample
    estimate_table = key_estimate(pitches, durations)
    tonic = estimate_table['tonic'][0]
    mode = estimate_table['mode'][0]
    assert tonic == 0
    assert mode == 'Major'
