import states
import matplotlib.pyplot as plt

from botHandler import BotHandler
from utils import TOKEN, STATES
from models import load_models, Network


def main(pytorch_model, sklearn_model):
    # Initializations
    new_offset = 0
    count = 0
    img = None
    pytorch_result, sklearn_result = 0, 0

    MaskDetectoBot = BotHandler(TOKEN)

    # previous_state is used to get back to the state before an error or help states happened
    state = STATES["INITIAL"]
    previous_state = STATES["INITIAL"]

    print('hi, now launching...')

    while True:
        all_updates = MaskDetectoBot.get_updates(new_offset)

        if len(all_updates) > 0:
            for current_update in all_updates:
                print(current_update)

                update_id = current_update['update_id']
                chat_id = current_update['message']['chat']['id']

                # Read any text before proceeding
                if "text" in current_update["message"]:
                    chat_text = current_update["message"]["text"]

                    # Enter help state
                    if chat_text.lower() == "help":
                        previous_state = state
                        state = STATES["HELP"]

                    # Enter error state
                    elif chat_text.lower() != "/start":
                        previous_state = state
                        state = STATES["ERROR"]

                # Choose the current state
                if state == STATES["INITIAL"]:
                    state = states.initial(MaskDetectoBot, chat_id)
                    new_offset = update_id + 1

                elif state == STATES["LOADING"]:

                    if "photo" in current_update["message"]:
                        file_id = current_update["message"]["photo"][0]["file_id"]

                        state, img = states.loading(MaskDetectoBot, chat_id, file_id)
                        plt.imshow(img)
                        plt.show()
                        img.save(f"treasure{count}.jpg")
                        count += 1

                    else:
                        previous_state = state
                        state = STATES["NO_IMAGE"]

                if state == STATES["PREDICT"]:
                    state, pytorch_result, sklearn_result = states.predict(img, pytorch_model, sklearn_model)

                if state == STATES["RESULTS"]:
                    state = states.results(MaskDetectoBot, chat_id, pytorch_result, sklearn_result)
                    new_offset = update_id + 1

                if state == STATES["HELP"]:
                    states.help(MaskDetectoBot, chat_id, previous_state)
                    state = previous_state
                    new_offset = update_id + 1

                elif state == STATES["ERROR"] or state == STATES["NO_IMAGE"]:
                    states.error(MaskDetectoBot, chat_id, state)
                    state = previous_state
                    new_offset = update_id + 1

if __name__ == '__main__':
    # Load the models here
    pytorch_model, sklearn_model = load_models()

    try:
        main(pytorch_model, sklearn_model)
    except KeyboardInterrupt:
        exit()