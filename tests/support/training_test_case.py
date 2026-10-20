from tests.support.training_fixtures import *

class TrainingTestCase(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_training(self, dataset, name='run', **overrides):
        options = dict(feature_sets=('color', 'hog'), classifiers=('svm', 'random_forest'), image_size=64, hog_cell=16, grids=FAST_GRIDS, log=lambda *_: None)
        options.update(overrides)
        grids = options.pop('grids')
        log = options.pop('log')
        config = TrainingConfig(dataset_dir=dataset, models_dir=self.root / name / 'models', results_dir=self.root / name / 'results', **options)
        return ModelTrainer(config, grids=grids, log=log).run()
