from utils import  AugmentedWikiDataFormatter


if __name__ == "__main__":
    formatter = AugmentedWikiDataFormatter()
    formatter.process_raw_data(folder_path='./Datasets/Augmentation/Gemini Wiki', output_filename='gemini_wiki')