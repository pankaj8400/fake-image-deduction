from pathlib import Path
from PIL import Image

uploads = list(Path('app/uploads').glob('*.jpg'))
for p in sorted(uploads):
    im = Image.open(p)
    ex = im.getexif()
    soft = ex.get(305)
    make = ex.get(271)
    model = ex.get(272)
    if soft or make or model:
        print(f'{p.name}: Software="{soft}", Make="{make}", Model="{model}"')
