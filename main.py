import PySimpleGUI as sg
from model import *
import numpy as np
from operator import itemgetter
sg.theme("Reds")
model = build_model()
model.load_weights("./model_finall.hdf5")
embedding_model = get_embedding_model(
    "/mnt/wine/ai_models/sentence_transformers/sentence-transformers_paraphrase-multilingual-mpnet-base-v2/"
)


def detect(text):
    text = clean_text(text)
    embeddings = embedding_model.encode([text])
    embeddings = np.expand_dims(embeddings, axis=1)
    pred = model.predict(embeddings)
    if pred > 0.5:
        return "Hate Speech/Offensive Language", pred
    else:
        return "Clean Language", pred


def text(text):
    return sg.Text(text, font="Arial 20", justification="center")


def main_screen():

    layout = [
        [text("Hate Speech/Offensive Language Detector")],
        [sg.Multiline(size=(50, 10), font="Arial 20", key="text")],
        [sg.Button("Detect", font="Arial 20", key="Detect")],
    ]

    return sg.Window("Hate Speech/Offensive Language Detector", layout, finalize=True)


if __name__ == "__main__":

    window = main_screen()

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        if event == "Detect":
            result = detect(values["text"])[0]
            if result == "Hate Speech/Offensive Language":
                words_score = {}
                tokens = clean_text(values["text"]).split()
                for token in tokens:
                    words_score[token] = detect(token)[1]
                
                sorrted_scores=sorted(words_score.items(), key=itemgetter(1), reverse=True)
                bad_word = sorrted_scores[0][0]
                result = f"{result}\n\nMost likely because you used the word '{bad_word}'"
                sg.popup(result, font="Arial 25", title="Result")
            else:
                sg.popup(result, font="Arial 25", title="Result")
