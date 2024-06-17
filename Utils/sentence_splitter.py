import re

from py4j.java_gateway import JavaGateway

class IndonesianSentenceDetector:
    def __init__(self):
        self.akronim = [line.strip() for line in open('./resource/sentencedetector/acronym.txt', 'r')]
        self.delimiter = [line.strip() for line in open('./resource/sentencedetector/delimiter.txt', 'r')]


    def split_sentence(self, sentences):
        sentence_list = []
        sentence_buffer = ""
        sentences = sentences.replace("–", "-")
        sentences = re.sub("[^\x00-\x7F]", "", sentences)
        sentences = re.sub("\s+", " ", sentences)
        sentences = re.sub("\.\s*\.", ".", sentences)
        sentences = re.sub("\?\?+", "-", sentences)
        tokens = sentences.split(" ")

        for token in tokens:
            if not any(char in token for char in self.delimiter):
                sentence_buffer += token + " "
            elif any(char in token for char in self.delimiter):
                if token in self.akronim:
                    sentence_buffer += token + " "
                else:
                    last = token.rfind(".")
                    if last != -1:
                        token = token[:last] + "."
                    sentence_buffer += token + " "
                    sentence_list.append(sentence_buffer.strip())
                    sentence_buffer = ""

        if sentence_buffer:
            sentence_list.append(sentence_buffer + " .")

        return sentence_list

    def is_akronim(self, word):
        return word in self.akronim

if __name__ == "__main__":
    detector = IndonesianSentenceDetector()
    sentence = "Ir. Soekarno dilahirkan di Surabaya, Jawa Timur, pada 6 Juni 1901 dan meninggal di Jakarta, pada 21 Juni 1970 adalah Presiden Indonesia pertama yang menjabat pada periode 1945–1966. Ia memainkan peranan penting untuk memerdekakan bangsa Indonesia dari penjajahan Belanda. Soekarno adalah penggali Pancasila karena ia yang pertama kali mencetuskan konsep mengenai dasar negara Indonesia itu dan ia sendiri yang menamainya Pancasila. Ia adalah Proklamator Kemerdekaan Indonesia (bersama dengan Mohammad Hatta) yang terjadi pada tanggal 17 Agustus 1945."
    sentence_list = detector.split_sentence(sentence)

    java = JavaGateway.launch_gateway(classpath="Utils/InaNLP.jar")
    isd = java.jvm.IndonesianNLP.IndonesianSentenceDetector()

    sentence_list_2 = isd.splitSentence(sentence)
    
    print(sentence_list)
    print(sentence_list_2)
    # for sentence in sentence_list:
    #     print(sentence + "\n")