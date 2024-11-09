import torch
from torchvision import transforms
from PIL import Image

# Загрузка модели с указанием, что нужно использовать CPU
model = torch.load("model.pt", map_location=torch.device('cpu'))  # Загрузка модели на CPU  # Явно указываем, что мы работаем на CPU

def predict_image(model, image_path, class_names):
    image = Image.open(image_path)

    # Преобразование изображения
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = transform(image).unsqueeze(0)

    model.eval()  # Переводим модель в режим оценки
    with torch.no_grad():  # Отключаем градиенты
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    predicted_class = class_names[predicted.item()]
    print(f'Предсаазанный класс: {predicted_class}')

# Список классов
class_names = ["blur", "contrast", "crop", "dark", "normal"]
image_path = '1.jpg'  # Путь к изображению

# Вызов функции предсказания
predict_image(model, image_path, class_names)