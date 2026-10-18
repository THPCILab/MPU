from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTests(unittest.TestCase):
    def test_documentation_links_resolve(self):
        checked = 0
        for document in [ROOT / "README.md", ROOT / "paper_data/README.md",
                         *sorted((ROOT / "docs").glob("*.md"))]:
            content = document.read_text(encoding="utf-8-sig")
            for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", content):
                checked += 1
                target = target.strip("<>").split("#", 1)[0]
                if not target or re.match(r"^[a-zA-Z]+:", target):
                    continue
                self.assertTrue((document.parent / target).exists(),
                                str(document.relative_to(ROOT)) + ": " + target)
        self.assertGreater(checked, 10)

    def test_generated_outputs_do_not_target_paper_data(self):
        for script in ("run_main_10pct_sweep_reuse_warmup.sh", "run_replicate_20pct_sweep.sh"):
            text = (ROOT / "scripts" / script).read_text(encoding="utf-8")
            self.assertNotIn("--out-dir paper_data", text)
            self.assertIn("--out-dir results/generated/", text)
        self.assertIn("results/generated/", (ROOT / ".gitignore").read_text(encoding="utf-8"))

    def test_repository_scope_is_explicit(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("**Figure 5**", text)
        self.assertIn("**Figures 2–4**", text)
        self.assertIn("](results/MPU.png)", text)
        self.assertIn("](results/main_figure.png)", text)
        self.assertIn("## Training workflow", text)
        for filename in ("requirements.txt", "LICENSE", "CITATION.cff",
                         "docs/installation.md", "docs/reproduction.md", "docs/custom_data.md"):
            self.assertTrue((ROOT / filename).is_file())


if __name__ == "__main__":
    unittest.main()
