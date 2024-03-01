from utils import  AugmentedDataFormatter


if __name__ == "__main__":
    formatter = AugmentedDataFormatter()
    formatter.process_raw_data(folder_path='./Datasets/Augmentation/Gemini', output_filename='gemini')