import argparse
import cv2
import os
from pathlib import Path
from tqdm import tqdm

# Setup Argument Parser
parser = argparse.ArgumentParser(description='todo: insert description')
parser.add_argument('--train', required=True, help='Path to images dir')
parser.add_argument('--result', required=True, help='Path to result dir')

# x, y, w, h = [502, 344, 61, 86]
# x, y, w, h = [627, 427, 193, 93]
# x, y, w, h = [522, 654, 90, 85]
# x, y, w, h = [704, 594, 90, 80]
# x, y, w, h = [545, 577, 81, 75]
x, y, w, h = [460, 700, 52, 80]


if __name__ == '__main__':
    # Parse arguments 
    cli = parser.parse_args()

    img_files = [x for x in Path(cli.train).glob('**/*') if x.is_file() and x.name.endswith('.jpg')]
    for img_path in tqdm(img_files):
        img = cv2.imread(str(img_path))

        # cv2.imshow('image',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        roi = img[y:y + h, x:x + w]
        cv2.imwrite(os.path.join(Path(cli.result), img_path.name), roi)

    print("Done!")


