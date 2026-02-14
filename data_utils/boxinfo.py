
## file ( volleyball_tracking_annotation )
## ex: 0 244 430 318 568 3636 0 0 1 standing 
## ex: 5 868 464 992 614 3653 0 1 1 moving 
## ex: 9 880 358 961 518 3651 0 1 1 setting

class BoxInfo:
    def __init__(self, line):
        words = line.split()
        self.category = words.pop()
        words = [int(string) for string in words]
        self.player_ID = words[0]
        del words[0]

        x1, y1, x2, y2, frame_ID, lost, grouping, generated = words
        self.box = x1, y1, x2, y2
        self.frame_ID = frame_ID
        self.lost = lost
        self.grouping = grouping
        self.generated = generated
