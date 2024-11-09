import telebot
from telebot import types
import cv2

print("Запуск...")

TOKEN = '7194071701:AAGTn9BpCn2KzOxsGtDKUmB27nE8UExEkHQ'
bot = telebot.TeleBot(TOKEN)
user_states = {}

print("Сервер запущен")


@bot.message_handler(commands=['start'])
def welcome_message(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("Круглые")
    btn2 = types.KeyboardButton("Квадратные")
    markup.add(btn1)
    markup.add(btn2)
    bot.send_message(message.chat.id,
                     f"Здравствуйте, {message.from_user.first_name}. Я бот для подсчёта труб на фото. Пожалуйста, выберите тип труб, которых вы хотите подсчитать или введите команду /info для дополнительной информации",
                     reply_markup=markup)


@bot.message_handler(commands=['info'])
def info_message(message):
    bot.send_message(message.chat.id,
                     "Данный бот разработан для Sber AI challenge. Он использует технологию YOLO для сегментации изображений. Разработчики: \n Хватов Сергей \n Беляев Александр \n Чекуева Алима \n Латыпова Юлия \n Веселов Данила")


@bot.message_handler(func=lambda message: True)
def count_pipes(message):
    if message.text == 'Круглые':
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        bot.send_message(message.chat.id,
                         "Отправьте фото трубы. Старайтесь делать фото как можно качественней, камеру держите перпендикулярно сечению трубы. Для обрезки фото воспользуйтесь функцией обрезки.",
                         reply_markup=markup)
        user_states[message.chat.id] = {"shape": "round", "state": "nores", "image" : None}
    elif message.text == 'Квадратные':
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        bot.send_message(message.chat.id,
                         "Отправьте фото трубы. Старайтесь делать фото как можно качественней, камеру держите перпендикулярно сечению трубы. Для обрезки фото воспользуйтесь функцией обрезки.",
                         reply_markup=markup)
        user_states[message.chat.id] = {"shape": "square", "state": "nores", "image" : None}
    elif message.text=='Переснять':
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        bot.send_message(message.chat.id,
                         "Отправьте фото трубы. Старайтесь делать фото как можно качественней, камеру держите перпендикулярно сечению трубы. Для обрезки фото воспользуйтесь функцией обрезки.",
                         reply_markup=markup)
    elif message.text == 'Продолжить':
        if message.chat.id in user_states and user_states[message.chat.id]["state"]=="proceed":

            if user_states[message.chat.id]["shape"] == 'round':
                from pipe_search import sh
                save_path=user_states[message.chat.id]["image"]
                combined_image, ans = sh.start(save_path, 0, message.chat.id)
                if ans == 0:
                    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
                    btn1 = types.KeyboardButton("Круглые")
                    btn2 = types.KeyboardButton("Квадратные")
                    markup.add(btn1)
                    markup.add(btn2)
                    bot.send_message(message.chat.id, f"На этом фото нет труб", reply_markup=markup)
                else:
                    op = f"{save_path}_predict.jpg"
                    cv2.imwrite(op, combined_image)

                    downloaded_photo = open(op, 'rb')
                    bot.send_photo(message.chat.id, downloaded_photo)
                    if ans >= 5 and ans <= 20:
                        slovo = "труб"
                    elif ans % 10 == 1:
                        slovo = "труба"
                    elif ans % 10 in [2, 3, 4]:
                        slovo = "трубы"
                    else:
                        slovo = "труб"
                    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
                    btn1 = types.KeyboardButton("Круглые")
                    btn2 = types.KeyboardButton("Квадратные")
                    markup.add(btn1)
                    markup.add(btn2)
                    bot.send_message(message.chat.id, f"На этом фото {ans} {slovo}", reply_markup=markup)

            else:
                save_path=user_states[message.chat.id]["image"]
                from pipe_search import sh
                combined_image, ans = sh.start(save_path, 1, message.chat.id)
                markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
                btn1 = types.KeyboardButton("Круглые")
                btn2 = types.KeyboardButton("Квадратные")
                markup.add(btn1)
                markup.add(btn2)
                if ans == 0:
                    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
                    btn1 = types.KeyboardButton("Круглые")
                    btn2 = types.KeyboardButton("Квадратные")
                    markup.add(btn1)
                    markup.add(btn2)
                    bot.send_message(message.chat.id, f"На этом фото нет труб", reply_markup=markup)
                else:
                    op = f"{save_path}_predict.jpg"
                    cv2.imwrite(op, combined_image)

                    downloaded_photo = open(op, 'rb')
                    bot.send_photo(message.chat.id, downloaded_photo)
                    if ans >= 5 and ans <= 20:
                        slovo = "труб"
                    elif ans % 10 == 1:
                        slovo = "труба"
                    elif ans % 10 in [2, 3, 4]:
                        slovo = "трубы"
                    else:
                        slovo = "труб"
                    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
                    btn1 = types.KeyboardButton("Круглые")
                    btn2 = types.KeyboardButton("Квадратные")
                    markup.add(btn1)
                    markup.add(btn2)
                    bot.send_message(message.chat.id, f"На этом фото {ans} {slovo}", reply_markup=markup)

        else:
            bot.send_message(message.chat.id, "Я вас не понимаю. Если вы хотите подсчитать количество труб на фотографии, пожалуйста, выберите тип труб, которых вы хотите подсчитать")

    else:
        bot.send_message(message.chat.id,
                         "Я вас не понимаю. Если вы хотите подсчитать количество труб на фотографии, пожалуйста, выберите тип труб, которых вы хотите подсчитать")


@bot.message_handler(content_types=['photo'])
def process_photo(message):
    if message.chat.id not in user_states or user_states[message.chat.id]["shape"] not in ['round', 'square']:
        bot.reply_to(message, "Перед отправкой фото пожалуйста выберите тип труб, которые хотите посчитать.")
        return

    import torch
    from torchvision import transforms
    from PIL import Image

    model = torch.load(r"Model/model.pt", map_location=torch.device('cpu'))
    device=torch.device('cpu')
    model.to(device)

    def predict_image(model, image_path, device, class_names):
        image = Image.open(image_path)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image = transform(image).unsqueeze(0)
        image = image.to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)

        predicted_class = class_names[predicted.item()]
        return predicted_class

    photo = message.photo[-1]
    file_info = bot.get_file(photo.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    save_path = f'downloaded_photo{message.chat.id}.jpg'
    with open(save_path, 'wb') as new_file:
        new_file.write(downloaded_file)
    class_names = ["blur", "contrast", "crop", "dark", "normal"]

    predmodel=predict_image(model, save_path, device, class_names)
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("Переснять")
    btn2 = types.KeyboardButton("Продолжить")
    markup.add(btn1)
    markup.add(btn2)
    shape=user_states[message.chat.id]["shape"]
    if predmodel=="blur":
        bot.send_message(message.chat.id, "Вы отправили размытое фото. Рекомендуем вам переснять, иначе модель может выдавать менее точные результаты. Отправьте новое фото или нажмите кнопку \"Продолжить\", если хотите продолжить", reply_markup=markup)
        user_states[message.chat.id]={"shape": shape, "image" :save_path, "state" : "proceed"}
    elif predmodel=="contrast":
        bot.send_message(message.chat.id, "Вы отправили засвеченное фото (проблема с освещением). Рекомендуем вам переснять, иначе модель может выдавать менее точные результаты. Отправьте новое фото или нажмите кнопку \"Продолжить\", если хотите продолжить", reply_markup=markup)
        user_states[message.chat.id] = {"shape": shape, "image": save_path, "state": "proceed"}
    elif predmodel=="crop":
        bot.send_message(message.chat.id, "Вы отправили обрезанное фото. Рекомендуем вам переснять, иначе модель может выдавать менее точные результаты. Отправьте новое фото или нажмите кнопку \"Продолжить\", если хотите продолжить", reply_markup=markup)
        user_states[message.chat.id] = {"shape": shape, "image": save_path, "state": "proceed"}
    elif predmodel=="dark":
        bot.send_message(message.chat.id, "Вы отправили фото с плохим освещением. Рекомендуем вам переснять, иначе модель может выдавать менее точные результаты. Отправьте новое фото или нажмите кнопку \"Продолжить\", если хотите продолжить", reply_markup=markup)
        user_states[message.chat.id] = {"shape": shape, "image": save_path, "state": "proceed"}
    elif user_states[message.chat.id]["shape"] == 'round':
        sent_message = bot.send_message(message.chat.id,
                                        "Запускаю искуственный интеллект...")
        from pipe_search import sh
        import subprocess
        combined_image, ans = sh.start(save_path, 0, message.chat.id)
        if ans == 0:
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            btn1 = types.KeyboardButton("Круглые")
            btn2 = types.KeyboardButton("Квадратные")
            markup.add(btn1)
            markup.add(btn2)
            bot.send_message(message.chat.id, f"На этом фото нет труб", reply_markup=markup)
        elif ans==-1:
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            btn1 = types.KeyboardButton("Круглые")
            btn2 = types.KeyboardButton("Квадратные")
            markup.add(btn1)
            markup.add(btn2)
            bot.send_message(message.chat.id, f"Ошибка. Возможно вы выбрали неправильный тип труб или ваше фото сломало модель. Попробуйте еще раз", reply_markup=markup)
        else:
            op = f"{save_path}_predict.jpg"
            cv2.imwrite(op, combined_image)

            downloaded_photo = open(op, 'rb')
            bot.send_photo(message.chat.id, downloaded_photo)
            if ans >= 5 and ans <= 20:
                slovo = "труб"
            elif ans % 10 == 1:
                slovo = "труба"
            elif ans % 10 in [2, 3, 4]:
                slovo = "трубы"
            else:
                slovo = "труб"
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            btn1 = types.KeyboardButton("Круглые")
            btn2 = types.KeyboardButton("Квадратные")
            markup.add(btn1)
            markup.add(btn2)
            bot.send_message(message.chat.id, f"На этом фото {ans} {slovo}", reply_markup=markup)

    else:
        from pipe_search import sh
        import subprocess
        combined_image, ans = sh.start(save_path, 1, message.chat.id)
        if ans == 0:
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            btn1 = types.KeyboardButton("Круглые")
            btn2 = types.KeyboardButton("Квадратные")
            markup.add(btn1)
            markup.add(btn2)
            bot.send_message(message.chat.id, f"На этом фото нет труб", reply_markup=markup)
        else:
            op = f"{save_path}_predict.jpg"
            cv2.imwrite(op, combined_image)

            downloaded_photo = open(op, 'rb')
            bot.send_photo(message.chat.id, downloaded_photo)
            if ans >= 5 and ans <= 20:
                slovo = "труб"
            elif ans % 10 == 1:
                slovo = "труба"
            elif ans % 10 in [2, 3, 4]:
                slovo = "трубы"
            else:
                slovo = "труб"
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
            btn1 = types.KeyboardButton("Круглые")
            btn2 = types.KeyboardButton("Квадратные")
            markup.add(btn1)
            markup.add(btn2)
            bot.send_message(message.chat.id, f"На этом фото {ans} {slovo}", reply_markup=markup)


bot.polling()

