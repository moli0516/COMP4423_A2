"""Migrated from test_features-2.py: CommandLineTests."""

from tests.support.features_fixtures import *

class CommandLineTests(unittest.TestCase):

    def run_main(self, arguments):
        stdout = io.StringIO()
        with mock.patch('sys.argv', ['features.py', *arguments]), contextlib.redirect_stdout(stdout):
            code = features_public.main()
        return (code, stdout.getvalue())

    def test_reports_default_dimensions_without_images(self):
        code, output = self.run_main([])
        self.assertEqual(code, 0)
        self.assertIn('total=1812', output)

    def test_lbp_flag_changes_reported_dimension(self):
        code, output = self.run_main(['--lbp'])
        self.assertEqual(code, 0)
        self.assertIn("'lbp': 10", output)
        self.assertIn('total=1822', output)

    def test_image_errors_return_one_but_other_images_still_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            good = write_image(Path(temp) / 'good.png', random_array(11, 90))
            bad = Path(temp) / 'bad.jpg'
            bad.write_bytes(b'bad')
            code, output = self.run_main([str(bad), str(good)])
        self.assertEqual(code, 1)
        self.assertIn('ERROR', output)
        self.assertIn('OK', output)
        self.assertIn('shape=(1812,)', output)
