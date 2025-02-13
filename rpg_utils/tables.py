from rpg_utils.distributions import (
    generate_last_room_probabilities,
    Roll,
    LastRoomProbability,
)

from tabulate import tabulate

roll = Roll(2, 20, 10)
probs = generate_last_room_probabilities(roll)


def generate_table(probabilities: list[LastRoomProbability]) -> str:
    """
    Generate a table of probabilities.

    :param probabilities: list of last room probabilities with different splits.
    :return: latex table
    """
    rows = []
    first_lrp = probabilities[0]

    for num_rooms in range(first_lrp.max_rooms + 1):

        row = [num_rooms]

        for num_splits, lrp in enumerate(probabilities):
            if num_rooms < lrp.min_rooms:
                row.append(0)
            elif num_rooms > lrp.max_rooms:
                row.append(100)
            else:
                pct = lrp[num_rooms] * 100
                row.append(pct)

        if all(pct == 0 for pct in row[1:]):
            # in no conditions it ends here
            continue

        rows.append(row)

    headers = [
        "# Rooms",
        "No Splits",
        "One Split",
        "Two Splits",
        "Three Splits",
        "Four Splits",
        "Five Splits",
        "Six Splits",
    ][: len(probabilities) + 1]
    table = tabulate(
        rows,
        headers=headers,
        tablefmt="latex_booktabs",
    )

    return table
