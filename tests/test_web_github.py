import unittest
from unittest import mock

from src.web import github


class GitHubStarsTests(unittest.TestCase):
    def setUp(self):
        github._cache.update(stars=None, expires=0.0)

    def test_stars_are_cached_and_kept_after_a_failure(self):
        with mock.patch.object(github, "fetch_stars", return_value=12) as fetch:
            self.assertEqual(github.stars(now=0), 12)
            self.assertEqual(github.stars(now=100), 12)
            self.assertEqual(fetch.call_count, 1)
        with mock.patch.object(github, "fetch_stars", return_value=None) as fetch:
            self.assertEqual(github.stars(now=github.SUCCESS_TTL + 1), 12)
            self.assertEqual(github.stars(now=github.SUCCESS_TTL + 2), 12)
            self.assertEqual(fetch.call_count, 1)

    def test_compact_count(self):
        self.assertEqual([github.compact(n) for n in (0, 999, 1000, 1250, 12000)], ["0", "999", "1 mil", "1,2 mil", "12 mil"])
