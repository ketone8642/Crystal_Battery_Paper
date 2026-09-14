"""Check the installed browser assets and the configured API without training."""

from pathlib import Path
import sys
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from app.main import create_app
from app.schemas import EXAMPLE


class AssetLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.assets = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            if attrs['id'] in self.ids:
                raise ValueError('Duplicate HTML element ID: ' + attrs['id'])
            self.ids.add(attrs['id'])
        for name in ('src', 'href'):
            value = attrs.get(name, '')
            if value.startswith('/static/'):
                self.assets.append(value)


def main():
    try:
        with TestClient(create_app(ROOT)) as client:
            ready = client.get('/ready')
            if ready.status_code != 200:
                raise ValueError('The Step 7 model configuration is not ready: ' + ready.text)
            home = client.get('/')
            if home.status_code != 200:
                raise ValueError('The browser page did not load')
            parsed = AssetLinks()
            parsed.feed(home.text)
            required = {'single-form', 'batch-form', 'profile-form', 'models-table'}
            if not required.issubset(parsed.ids):
                raise ValueError('The installed page is not the Step 8 interface')
            for url in parsed.assets + ['/static/ui_utils.mjs?v=8.1']:
                response = client.get(url)
                if response.status_code != 200:
                    raise ValueError('Missing browser asset: ' + url)
                if any(ext in url for ext in ('.js', '.mjs')) and 'javascript' not in response.headers.get('content-type', ''):
                    raise ValueError('Incorrect JavaScript content type: ' + url)
            predicted = client.post('/predict', json=EXAMPLE)
            if predicted.status_code != 200:
                raise ValueError('Prediction check failed: ' + predicted.text)
            result = predicted.json()
        print('[OK] All six configured model bundles are ready')
        print('[OK] Four workspace sections and local browser assets are available')
        print('[OK] JavaScript files are served with module-compatible content types')
        print(f"[OK] FePO4 -> LiFePO4 prediction: {result['predicted_voltage_V']:.4f} V")
        print('STEP 8 UI ASSET AND API CHECK PASSED')
        print('Next: start Uvicorn and verify the interface in Chrome. This checker does not execute a browser.')
        return 0
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(f'[ERROR] {exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
