# Import packages
import pandas as pd
import numpy as np
from itertools import combinations
import pretty_midi


# define dissonance-related functions
def dissonance_curve(bandwidth_distance):
    """
    :param bandwidth_distance: a numeric value indicating bandwidth
    :return: value of dissonance
    """
    return (4 * bandwidth_distance * np.exp(1 - 4 * bandwidth_distance)) ** 2


def get_frequency(midi_number):
    """
    Calculate frequency of sound wave according to midi number
    """
    frequency = 440 * 2 ** ((int(midi_number) - 69) / 12)
    return round(frequency, 1)


def read_midi(filename):
    """
    Read files in midi format into separate notes
    """
    midifile = pretty_midi.PrettyMIDI(filename)
    notes = []
    for instru in midifile.instruments:
        notes = notes + instru.notes
    return notes


def expand_into_harmonics(chord_midi, n_harmonics=11):
    """
    expand tones in a given chord into its implied harmonics
    :param chord_midi: a list of midi number
    :param n_harmonics: a numeric number indicating wanted number of harmonics
    :return: a spectrum
    """

    chord_freq = [get_frequency(midi_num) for midi_num in chord_midi]

    # initialize freq list and amplitude list for the spectrum
    freq_list = []
    amplitude_list = []

    for freq in chord_freq:
        freq_list += [(x + 1) * freq for x in range(n_harmonics)]
        amplitude_list += [1 / (x + 1) for x in range(n_harmonics)]

    spectrum = pd.DataFrame({'freq_hz': freq_list, 'amplitude': amplitude_list})
    spectrum = spectrum.sort_values('freq_hz', ignore_index=True)
    return spectrum


def calc_dissonance_spec(spectrum):
    # get all the possible combinations
    comb = combinations(range(len(spectrum)), 2)
    # initialize dissonance score
    dissonance_score = 0
    # for each combination, calculate critical_bandwidth, per_critical_bandwidth
    for i, j in comb:
        freq_i, freq_j = spectrum['freq_hz'].iloc[i], spectrum['freq_hz'].iloc[j]
        amp_i, amp_j = spectrum['amplitude'].iloc[i], spectrum['amplitude'].iloc[j]

        avg_freq = (freq_i + freq_j) / 2
        bandwidth = 1.72 * (avg_freq ** 0.65)
        per_bandwidth = abs(freq_i - freq_j) / bandwidth

        dissonance_score += amp_i * amp_j * dissonance_curve(per_bandwidth)

    # calculate normalizing factor
    amp = spectrum['amplitude']
    squared_amp = amp ** 2
    normalize_factor = squared_amp.sum()

    # calculate dissonant score
    dissonance_score_normalized = dissonance_score / normalize_factor

    return dissonance_score_normalized


def calc_dissonance(chord, n_harmonics=11):
    """
    Calculate dissonance for a chord
    :param chord: a list of pitch in midi format
    :param n_harmonics: wanted number of harmonics used
    :return: dissonance score
    """
    spectrum = expand_into_harmonics(chord, n_harmonics)
    dissonance_score_normalized = calc_dissonance_spec(spectrum)
    return dissonance_score_normalized


def salami_slice(filename):
    notes = read_midi(filename)

    pitches = [note.pitch for note in notes]
    starts = [note.start for note in notes]
    ends = [note.end for note in notes]

    # prepare for slice
    slice_start_times = set(starts)
    num_slices = len(slice_start_times)
    salami_box = []

    for i, slice_start_time in enumerate(slice_start_times):
        pitches_curr_slice = [pitch for (pitch, start, end) in zip(pitches, starts, ends)
                              if (start == slice_start_time) or (start < slice_start_time <= end)]
        salami_box.append(pitches_curr_slice)

    return salami_box, slice_start_times


def calc_dissonance_salami_slice(filename):
    salami_box, slice_start_times = salami_slice(filename)
    num_slices = len(salami_box)
    dissonance_time_course = np.zeros((1, num_slices))

    for i, curr_slice in enumerate(salami_box):
        score_dissonance = calc_dissonance(curr_slice)
        dissonance_time_course[0, i] = score_dissonance

    # sort arrays according to start time
    dissonance_table = pd.DataFrame(
        {'dissonance_score': dissonance_time_course[0], 'start_time': list(slice_start_times)})
    dissonance_table = dissonance_table.sort_values('start_time')
    return dissonance_table


def avg_dissonance_table(dissonance_table, time_step, chunk_time):
    chunk_start_time = 0
    chunk_end_time = chunk_start_time + chunk_time
    avg_dissonance = []

    while chunk_end_time <= dissonance_table['start_time'].max():
        chunk = dissonance_table[
            (dissonance_table['start_time'] >= chunk_start_time) & (dissonance_table['start_time'] < chunk_end_time)]
        avg_dissonance.append(chunk['dissonance_score'].mean())

        chunk_start_time += time_step
        chunk_end_time = chunk_start_time + chunk_time
    return np.array(avg_dissonance)


# define key-finding related functions
def origin_kk_profiles():
    kk_profiles = {'Major': np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
                   'Minor': np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])}

    return kk_profiles


def key_estimate(pitches, durations):
    profile = [0 for _ in range(12)]

    for pitch, duration in zip(pitches, durations):
        pitch_class = pitch % 12
        profile[pitch_class] += duration
    results = []

    for tonic in range(12):
        for mode in ["Major", "Minor"]:
            score = get_fit(profile, tonic=tonic, mode=mode)
            results.append({
                "score": score,
                "tonic": tonic,
                "mode": mode,
            })
    results.sort(key=lambda x: - x["score"])
    return pd.DataFrame.from_records(results)


def get_fit(profile, tonic, mode):
    reference_profile = get_kk_profile(mode=mode, tonic=tonic)
    score = get_correlation(profile, reference_profile)
    return score


def get_correlation(x, y):
    return np.corrcoef(x, y)[0, 1]


# function: deriving the kk profile to specified mode and tonic.
def get_kk_profile(mode, tonic):
    # return kk_profiles[mode][(np.array(range(12)) - tonic) % 12]
    kk_profiles = origin_kk_profiles()
    mode_profile = kk_profiles[mode]
    abs_pitch_class_idx = np.array(range(12))
    relative_pitch_class_idx = (abs_pitch_class_idx - int(tonic)) % 12
    return mode_profile[relative_pitch_class_idx]


# define a function to analyse pitch chunk-by-chunk
# input type is midi
def key_estimate_from_midi(filename):
    notes = read_midi(filename)

    pitches = [note.pitch for note in notes]
    durations = [note.end - note.start for note in notes]
    return key_estimate(pitches, durations)


def key_estimate_rolling_from_midi(filename, chunk_time, time_step):
    notes = read_midi(filename)

    pitches = [note.pitch for note in notes]
    durations = [note.end - note.start for note in notes]
    starts = [note.start for note in notes]
    ends = [note.end for note in notes]

    # set pitch and duration as dataframe and cut notes into chunk
    key_time_course = []
    best_key_time_course = []
    tonalness = []

    chunk_start_time = 0
    chunk_end_time = chunk_start_time + chunk_time
    chunk_num = 1
    while chunk_end_time <= max(ends):
        chunk_pitches = [
            pitch for (pitch, start, end) in zip(pitches, starts, ends)
            if (start >= chunk_start_time) & (end < chunk_end_time)
        ]
        chunk_durations = [
            duration for (duration, start, end) in zip(durations, starts, ends)
            if (start >= chunk_start_time) & (end < chunk_end_time)
        ]
        key_possible = key_estimate(chunk_pitches, chunk_durations)
        key_possible['chunk_num'] = chunk_num

        key_time_course.append(key_possible)
        best_key_time_course.append(key_possible[['tonic', 'mode']].iloc[[0]])
        tonalness.append(key_possible[['score']].iloc[0])

        # update chunk information
        chunk_start_time += time_step
        chunk_end_time = chunk_start_time + chunk_time
        chunk_num += 1

    key_time_course = pd.concat(key_time_course, ignore_index=True, axis=0)
    best_key_time_course = pd.concat(best_key_time_course, ignore_index=True, axis=0)
    tonalness = pd.concat(tonalness, ignore_index=True, axis=0)

    return key_time_course, best_key_time_course, tonalness

