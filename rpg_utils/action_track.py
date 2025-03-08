"""
Functionality for an action tracker in RPG games
"""

from dataclasses import dataclass


class CharacterTracker:
    """
    Tracker for a character in the action track.
    """

    def __init__(self, action_points: int = 7, name: str = ""):
        self.action_points = action_points
        self.name = name


@dataclass()
class InvalidAction(Exception):
    current_position: int
    tracker_position: int
    action_cost: int


class ActionTrack:

    def __init__(self, max_num_positions: int = 12):
        self.max_num_positions = max_num_positions
        self.current_position = 0
        self.track = [[] for _ in range(max_num_positions)]
        self.character_trackers: dict[str, CharacterTracker] = {}
        self.character_positions: dict[str, int] = {}

    def reset(self, delete_chars: bool = False):
        """
        Reset the track to the starting position.
        """
        self.current_position = 0

        if delete_chars:
            self.track = [[] for _ in range(self.max_num_positions)]
            self.character_trackers: dict[str, CharacterTracker] = {}
            self.character_positions: dict[str, int] = {}
        else:
            first_position = list(self.character_trackers.keys())
            self.track = [first_position] + [
                [] for _ in range(self.max_num_positions - 1)
            ]
            self.character_positions = {c: 0 for c in self.character_positions}

    def add_tracker(self, tracker: CharacterTracker, position: int = 0) -> None:
        """
        Add a new character tracker and return its code.

        :param tracker: the tracker to add
        :param position: the position to place the tracker
        """
        name = tracker.name
        self.character_trackers[name] = tracker
        self.track[position].append(name)
        self.character_positions[name] = 0

    def get_next_to_act(self):
        current_position_list = self.track[self.current_position]

        if len(current_position_list):
            return current_position_list[0]
        else:
            return None

    def remove_character(self, name: str):
        position = self.character_positions[name]
        self.track[position].remove(name)

        del self.character_trackers[name]
        del self.character_positions[name]

    def maybe_advance_central_tracker(self):
        """
        Advance the central tracker if necessary.
        """
        if len(self.character_positions) == 0:
            return

        while len(self.track[self.current_position]) == 0:
            self.current_position += 1
            self.current_position %= self.max_num_positions

    def can_act(self, character_name: str, cost: int) -> bool:
        """
        Return whether the given character can perform an action with the given cost.
        """
        tracker = self.character_trackers[character_name]
        char_position = self.character_positions[character_name]
        action_points = tracker.action_points

        virtual_position = char_position + cost
        resulting_position = virtual_position % action_points

        # this happens if the character has already looped around the track
        char_is_looped = self.current_position > char_position

        # if the character goes past their action point limit, they loop around the track
        char_will_loop = virtual_position > action_points

        if char_will_loop and char_is_looped:
            # this would be another loop in the track before everyone else catches up!
            return False

        if char_is_looped and virtual_position > self.current_position:
            # same, this overtakes other characters before they act
            return False

        if char_will_loop and resulting_position > self.current_position:
            # this would make the character loop and go after the central tracker
            return False

        return True

    def perform_action(self, character_name: str, cost: int):
        tracker = self.character_trackers[character_name]
        char_position = self.character_positions[character_name]
        action_points = tracker.action_points

        # is it valid?
        virtual_position = char_position + cost
        resulting_position = virtual_position % action_points

        if not self.can_act(character_name, cost):
            raise InvalidAction(self.current_position, char_position, cost)

        self.track[char_position].remove(character_name)
        self.track[resulting_position].append(character_name)
        self.character_positions[character_name] = resulting_position

        # if the character who acted was at the acting position, advance the
        # central tracker
        self.maybe_advance_central_tracker()

    def draw(self):
        for i in range(self.max_num_positions):
            char_list = self.track[i]

            arrow = "-->" if self.current_position == i else "   "
            # contents = "" if len(char_list) else "[]"

            char_name_list = []
            for char in char_list:
                ap = self.character_trackers[char].action_points
                char_name_list.append(f"{char} ({ap} AP)")

            contents = ", ".join(char_name_list)
            line = f"{arrow} {i + 1:2d}: [{contents}]"
            print(line)
