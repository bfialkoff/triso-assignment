from pathlib import Path
from random import sample, seed
import pandas as pd

def build_annotations(subject_list):
    all_images = []
    all_masks = []
    for subject in subject_list:
        # glob _gt, as masks and files is replace with non
        masks = list(subject.glob('*_gt.mhd'))
        masks = [str(p) for p in masks]
        images = [p.replace('_gt', '') for p in masks]
        all_images += images
        all_masks += masks
    df = pd.DataFrame({'image_path': all_images, 'mask_path': all_masks})
    return df

mkdir = lambda p: p.mkdir(parents=True) if not p.exists() else None

if __name__ == '__main__':
    out_data_path = Path(__file__).joinpath('..', '..', 'data').resolve()
    data_path = Path(__file__).joinpath('..', '..', '..', 'camus_data').resolve()
    mkdir(out_data_path)
    seed(42)
    train_frac = 0.7

    train_path = data_path.joinpath('training')
    test_path = data_path.joinpath('testing')

    train_subjects = list(train_path.glob('*'))
    val_subjects = sample(train_subjects, int((1 - train_frac) * len(train_subjects)))
    train_subjects = list(set(train_subjects).difference(val_subjects))
    test_subjects = list(test_path.glob('**/*.mhd'))
    test_subjects = [str(p) for p in test_subjects if 'seq' not in p.name]

    train_df = build_annotations(train_subjects)
    val_df = build_annotations(val_subjects)
    test_df = pd.DataFrame(test_subjects, columns=['image_path'])

    train_df.to_csv(out_data_path.joinpath('train_annotations.csv').resolve(), index=False)
    val_df.to_csv(out_data_path.joinpath('val_annotations.csv').resolve(), index=False)
    test_df.to_csv(out_data_path.joinpath('test_annotations.csv').resolve(), index=False)

