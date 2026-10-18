import unittest

from tools.validate_fig3_source_data import validate


class Figure3SourceTests(unittest.TestCase):
    def test_published_tables_and_models(self):
        validate()


if __name__ == "__main__":
    unittest.main()
