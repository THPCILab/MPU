import unittest
import hashlib
from pathlib import Path
import tempfile

from tools.validate_fig3_source_data import validate, matches_source_hash


class Figure3SourceTests(unittest.TestCase):
    def test_csv_hash_allows_only_line_ending_conversion(self):
        expected = hashlib.sha256(b"value\r\n1\r\n").hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "values.csv"
            path.write_bytes(b"value\n1\n")
            self.assertTrue(matches_source_hash(path, expected))
            path.write_bytes(b"value\n2\n")
            self.assertFalse(matches_source_hash(path, expected))

    def test_published_tables_and_models(self):
        validate()


if __name__ == "__main__":
    unittest.main()
