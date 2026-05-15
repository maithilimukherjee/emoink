from hmm.hmmodel import (
    states,
    start_probability,
    transition_probability
)


def viterbi(observations):
    """
    Perform Viterbi decoding.

    Parameters:
        observations (list):
            List of normalized probability dictionaries.

    Returns:
        tuple:
            (most likely state sequence,
             final probabilities)
    """

    # dynamic programming table
    V = [{}]

    # path tracker
    path = {}

    # -------------------------
    # INITIALIZATION
    # -------------------------

    for state in states:

        V[0][state] = (
            start_probability[state]
            * observations[0].get(state, 0)
        )

        path[state] = [state]

    # -------------------------
    # RECURSION
    # -------------------------

    for t in range(1, len(observations)):

        V.append({})

        new_path = {}

        for current_state in states:

            # compute maximum transition
            max_prob, best_prev_state = max(

                (
                    V[t - 1][prev_state]
                    * transition_probability[prev_state][current_state]
                    * observations[t].get(current_state, 0),

                    prev_state
                )

                for prev_state in states
            )

            V[t][current_state] = max_prob

            new_path[current_state] = (
                path[best_prev_state]
                + [current_state]
            )

        path = new_path

    # -------------------------
    # TERMINATION
    # -------------------------

    max_final_prob, best_final_state = max(

        (V[-1][state], state)

        for state in states
    )

    best_path = path[best_final_state]

    return best_path, max_final_prob