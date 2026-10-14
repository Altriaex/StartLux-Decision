import unittest

from startlux_decide.cli import raise_open_file_limit

try:
    import resource
except ImportError:          # Windows
    resource = None


@unittest.skipIf(resource is None, "no RLIMIT_NOFILE on this platform")
class OpenFileLimitTest(unittest.TestCase):
    def setUp(self):
        self.saved = resource.getrlimit(resource.RLIMIT_NOFILE)

    def tearDown(self):
        resource.setrlimit(resource.RLIMIT_NOFILE, self.saved)

    def test_raises_a_low_soft_limit(self):
        soft, hard = self.saved
        if soft < 256:
            self.skipTest("soft limit already below 256")
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard))
        before, after = raise_open_file_limit(1024)
        self.assertEqual(before, 256)
        self.assertGreater(after, 256)
        self.assertEqual(resource.getrlimit(resource.RLIMIT_NOFILE)[0], after)

    def test_never_lowers_the_limit(self):
        soft = self.saved[0]
        before, after = raise_open_file_limit(16)
        self.assertEqual(before, after)
        self.assertEqual(resource.getrlimit(resource.RLIMIT_NOFILE)[0], soft)


if __name__ == "__main__":
    unittest.main()
