import os
import sys
from pathlib import Path
from django.core.wsgi import get_wsgi_application

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_DIR = Path(__file__).resolve().parent.parent
for path in [ROOT_DIR, PROJECT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_app.django_app.settings')
application = get_wsgi_application()
