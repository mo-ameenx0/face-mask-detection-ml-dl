"""
    initial state:
        Display the welcome message at the start ===change state===> LOADING

    loading state:
        Load the img ===change state===> PREDICT ===ELSE===> return to the initial state and display the error message.

    predict state:
        run the two models to give some predictions ===change state===> RESULTS

    results state:
        print the results ===change state===> INITIAL

    GENERAL:
        Display the error message for unknown command
        Display the help message when requested
"""

import torch
import torch.nn.functional as F

from utils import *
from PIL import Image
from io import BytesIO
from imagePreprocessing import pytorch_preprocessing, sklearn_preprocessing


def initial(bot, chat_id):
    bot.send_message(chat_id, WELCOME_MESSAGE)

    return STATES["LOADING"]


def loading(bot, chat_id, file_id):
    bot.send_message(chat_id, "Please Wait......")

    img_name = bot.get_file_path(file_id)
    img = bot.load_image(img_name)
    img = Image.open(BytesIO(img))

    return STATES["PREDICT"], img


def predict(img, pytorch_model, sklearn_model):
    # Process the image for training using pytorch and sklearn
    pytorch_features = pytorch_preprocessing(img)
    sklearn_features = sklearn_preprocessing(img)

    # Generate pytorch prediction
    pytorch_output = pytorch_model(pytorch_features)
    _, pytorch_output = torch.max(F.softmax(pytorch_output, dim=1), 1)
    pytorch_output = pytorch_output.view(-1)
    pytorch_pred = "With Mask" if pytorch_output == 0 else "Without Mask"

    # Generate sklearn prediction
    sklearn_output = sklearn_model.predict(sklearn_features)
    sklearn_pred = "With Mask" if sklearn_output == 0 else "Without Mask"


    return STATES["RESULTS"], pytorch_pred, sklearn_pred


def results(bot, chat_id, pytorch_result, sklearn_result):
    message = f"Pytorch Prediction = {pytorch_result}\n" \
              f"sklearn Prediction = {sklearn_result}"
    print(message)
    bot.send_message(chat_id, message)

    return STATES["LOADING"]


def help(bot, chat_id, help):
    bot.send_message(chat_id, HELP_MESSAGES[help])


def error(bot, chat_id, error):
    bot.send_message(chat_id, ERROR_MESSAGES[error])
