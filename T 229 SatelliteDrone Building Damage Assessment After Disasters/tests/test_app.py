import io
import unittest
from PIL import Image
from app import app
from model_inference import load_annotation
from pathlib import Path

class AppTests(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
    def test_homepage_and_health(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertEqual(self.client.get('/api/health').get_json()['application'], 'ready')
    def test_missing_annotation_is_actionable(self):
        image=io.BytesIO();Image.new('RGB',(16,16),'white').save(image,'PNG');image.seek(0)
        r=self.client.post('/api/analyze',data={'image':(image,'scene.png')},content_type='multipart/form-data')
        self.assertEqual(r.status_code,400);self.assertIn('annotation',r.get_json()['error'].lower())
    def test_florence_csv_is_polygon_annotation(self):
        path=Path(__file__).parents[1]/'T229_final_submission'/'hurricane-florence_00000537_geographic_building_damage.csv'
        if path.exists():
            data=load_annotation(path)
            self.assertGreater(len(data['features']['xy']),0)
            self.assertIn('POLYGON',data['features']['xy'][0]['wkt'])

if __name__ == '__main__': unittest.main()
