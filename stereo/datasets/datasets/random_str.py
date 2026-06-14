import numpy as np


class random_str(str):
    def __init__(self, choices=None,rating=None, seed = 123):
        self.choices = choices
        self.rating = rating if isinstance(rating, list) else [1/len(self.choices) for i in range(len(self.choices))]
        self.seed = seed
        self._check()
        np.random.RandomState(self.seed)
        pass

    def _check(self):
        assert len(self.choices) == len(self.rating), "Length of choices and rating must be the same"
        if sum(self.rating) != 1:
            self.rating = [i/sum(self.rating) for i in self.rating]
        for choice in self.choices:
            assert isinstance(choice, str), "All choices must be string"

    def __str__(self):
        return self._get_random_str()

    def _get_random_str(self):
        return np.random.choice(self.choices, p=self.rating)

if __name__ == "__main__":
    choices = ['ADE20K', 'diode', 'diw', 'mapillary', 'mscoco', 'sceneflow', 'kitti2015']
    rating = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4]
    random_str = random_str(choices, rating)
    maps = {
        "ADE20K": 0,
        "diode": 0,
        "diw": 0,
        "mapillary": 0,
        "mscoco": 0,
        "sceneflow": 0,
        "kitti2015": 0,
    }
    for i in range(10000):
        maps[random_str._get_random_str()] += 1

    for key in maps:
        print(f"{key}: {maps[key] / 10000}")


