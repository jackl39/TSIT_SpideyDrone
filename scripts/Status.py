#!/usr/bin/env python3

class Status:
    # Class attributes that define possible status values
    SAFE = "Safe"
    UNSAFE = "Unsafe"
    GOAL = "Goal"
    UNKNOWN = "Unknown"

    def __init__(self, state=UNKNOWN):
        self.state = state

    def set_safe(self):
        self.state = Status.SAFE

    def set_unsafe(self):
        self.state = Status.UNSAFE

    def set_goal(self):
        self.state = Status.GOAL

    def is_safe(self):
        return self.state == Status.SAFE

    def is_unsafe(self):
        return self.state == Status.UNSAFE
    
    def is_goal(self):
        return self.state == Status.GOAL
