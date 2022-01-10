
"""
    A: 跳
    B: 开火
"""


ACTIONS = [
    ['NOOP'],

    ['right', 'B'],
    ['right', 'A', 'B'],

    ['B'],
    ['down', 'B'],

    ['right', 'A', 'B', 'up'],
    ['right', 'A', 'B', 'down'],

    ['right', 'B', 'up'],
    ['right', 'B', 'down'],
]


[
    ['NOOP'],
    # ['right'],
    # ['right', 'A'],
    ['right', 'B'],
    # ['right', 'A', 'up'],
    ['right', 'B', 'up'],
    ['right', 'A', 'B', 'up'],
    # ['A'],
    ['B'],
    ['A', 'B'],

    # ['left'],
    # ['left', 'A'],
    # ['left', 'B'],
    # ['left', 'A', 'up'],
    # ['left', 'B', 'up'],
    # ['left', 'A', 'B', 'up'],

    # ['down', 'A'],
    ['down', 'B'],
    ['down', 'A', 'B'],
    ['right', 'down', 'B'],

    # ['up', 'A'],
    ['up', 'B'],
    ['up', 'A', 'B'],
    ['right', 'up', 'B']
]
