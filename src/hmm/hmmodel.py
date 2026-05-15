import numpy as np


# hidden affective states
states = [
    "calm",
    "tense",
    "lazy",
    "angry",
    "artistic"
]


# initial state probabilities
start_probability = {

    "calm": 0.3,
    "tense": 0.2,
    "lazy": 0.1,
    "angry": 0.1,
    "artistic": 0.3
}


# transition probability matrix
transition_probability = {

    "calm": {
        "calm": 0.5,
        "tense": 0.1,
        "lazy": 0.1,
        "angry": 0.05,
        "artistic": 0.25
    },

    "tense": {
        "calm": 0.2,
        "tense": 0.4,
        "lazy": 0.1,
        "angry": 0.2,
        "artistic": 0.1
    },

    "lazy": {
        "calm": 0.2,
        "tense": 0.1,
        "lazy": 0.5,
        "angry": 0.05,
        "artistic": 0.15
    },

    "angry": {
        "calm": 0.1,
        "tense": 0.3,
        "lazy": 0.05,
        "angry": 0.45,
        "artistic": 0.1
    },

    "artistic": {
        "calm": 0.3,
        "tense": 0.1,
        "lazy": 0.1,
        "angry": 0.05,
        "artistic": 0.45
    }
}