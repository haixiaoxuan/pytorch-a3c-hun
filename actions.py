
"""
    A: 跳
    B: 开火
"""


ACTIONS = [
    ["NOOP"],
    ["B"],
    ["A", "B"],
    ["down"],
    ["down", "B"],
    ["down", "A", "B"],

    ["up"],
    ["up", "B"],
    ["up", "A", "B"],

    ["right"],
    ["right", "B"],
    ["right", "A", "B"],

    ["right", "up"],
    ["right", "B", "up"],
    ["right", "A", "B", "up"],

    ["right", "down"],
    ["right", "B", "down"],
    ["right", "A", "B", "down"],
]

ACTIONS_MASK = {}


def gen_action_mask():
    for idx, ele in enumerate(ACTIONS):
        if "A" in ele or "B" in ele:
            for i in range(1, 3):
                if "A" not in ACTIONS[idx - i] and "B" not in ACTIONS[idx - i]:
                    ACTIONS_MASK[idx] = idx - i


gen_action_mask()