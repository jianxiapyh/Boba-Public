import unittest

from benchmarks.run_batched_full_runtime_case import build_parser, validate_args


class RuntimeVisualizationCliTests(unittest.TestCase):
    def parse(self, *extra):
        return build_parser().parse_args(
            ["--case_name", "test_case", "--batch_size", "2", *extra]
        )

    def test_display_batch_grid_defaults_off(self):
        self.assertFalse(self.parse().display_batch_grid)

    def test_display_batch_grid_is_valid_for_batch_images(self):
        args = self.parse("--display_batch_grid")
        validate_args(args)
        self.assertTrue(args.display_batch_grid)

    def test_display_batch_grid_is_rejected_for_instance_mode(self):
        args = self.parse(
            "--render_mode",
            "instance",
            "--instance_id",
            "0",
            "--display_batch_grid",
        )
        with self.assertRaisesRegex(ValueError, "display_batch_grid"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
