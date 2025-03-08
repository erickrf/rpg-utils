import marimo

__generated_with = "0.11.13"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Action Tracks

        Experiments and statistics for an approach to action economy in RPG games.

        Key ideas:

        - Each different action has a different cost in action points
            - e.g., 4 for an attack, 2 for a parry, 3 for picking up a weapon
        - A common number of action points (AP) per character is around 5-8, allowing more granular progression in gaining AP and a less abrupt difference in the power level of characters with different APs.
        - There is a centralized action track (AT) that keeps track of whose turn it is.

        This is how the action track works:

        - When a character does an action, their tracker advances in the AT a number equal to the action cost.
        - The character in the lowest position is the next to act.
        - When going past their max AP on the track, the character tracker reverts to the beginning.
        - A character cannot perform an action if it would place their tracker after the "current turn" marker.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    import rich
    from importlib import reload

    from rpg_utils import action_track as at
    reload(at)
    return at, mo, reload, rich


@app.cell
def _(track):
    track.draw()
    return


@app.cell
def _():
    def do_action_turns(track, num_turns: int = 20):
        action_counter = {'red': 0, 'green': 0, 'blue': 0}
        undefended_counter = {'red': 0, 'green': 0, 'blue': 0}

        for i in range(num_turns):
            next_to_act = track.get_next_to_act()
            if next_to_act in ['blue', 'green']:
                defender = 'red'
            else:
                defender = 'green'

            track.perform_action(next_to_act, 4)
            action_counter[next_to_act] += 1

            if track.can_act(defender, 2):
                track.perform_action(defender, 2)
            else:
                undefended_counter[defender] += 1

        return action_counter, undefended_counter
    return (do_action_turns,)


@app.cell
def _(mo):
    slider_ap_blue = mo.ui.slider(5, 10, show_value=True, label='Blue AP')
    slider_ap_green = mo.ui.slider(5, 10, show_value=True, label='Green AP')
    slider_ap_red = mo.ui.slider(5, 10, show_value=True, label='Red AP')

    mo.vstack([slider_ap_blue, slider_ap_green, slider_ap_red])
    return slider_ap_blue, slider_ap_green, slider_ap_red


@app.cell
def _(at, do_action_turns, slider_ap_blue, slider_ap_green, slider_ap_red):
    track = at.ActionTrack()

    t1 = at.CharacterTracker(slider_ap_blue.value, 'blue')
    t2 = at.CharacterTracker(slider_ap_green.value, 'green')
    t3 = at.CharacterTracker(slider_ap_red.value, 'red')

    for t in [t1, t2, t3]:
        track.add_tracker(t)

    do_action_turns(track, 100)
    return t, t1, t2, t3, track


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
