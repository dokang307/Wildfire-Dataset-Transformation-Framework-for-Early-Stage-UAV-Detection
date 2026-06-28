import unittest
from app import app, detector

class TestBackend(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_health(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['status'], 'healthy')

    def test_model_loaded(self):
        self.assertIsNotNone(detector)
        self.assertEqual(detector.input_width, 1280)

if __name__ == '__main__':
    unittest.main()
