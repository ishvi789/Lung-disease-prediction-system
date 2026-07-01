from django.test import SimpleTestCase
from django.urls import reverse


class AppPagesTests(SimpleTestCase):
    def test_homepage_renders(self):
        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Lung Disease Classifier')

    def test_stylesheet_is_served(self):
        response = self.client.get('/static/css/style.css')
        self.assertEqual(response.status_code, 200)
