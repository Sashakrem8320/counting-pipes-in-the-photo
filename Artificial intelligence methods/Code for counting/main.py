from pipe_search import sh
import os
import json
import pandas as pd
if __name__ == '__main__':
    print("Запуск обработки")
    with open("config.json", "r") as f:
        data = json.load(f)
    file_names = []
    test_folder = f"{data['path_to_folder_with_images_for_round']}/"
    for root, dirs, files in os.walk(test_folder):
        for file in files:
            file_names.append(os.path.join(root, file))
    results = []
    numb_img = len(file_names)
    for i in range(numb_img):
        cls = 0
        ret = sh.start(file_names[i], cls) #1 - кв            0 - круг
        results.append(
            dict(
                img_path=f"{file_names[i]}",
                annot_path=ret[1],
                pipe_class=cls,
                pipe_count=ret[0],
            )
        )

    file_names = []
    test_folder = f"{data['path_to_folder_with_images_for_square']}/"
    for root, dirs, files in os.walk(test_folder):
        for file in files:
            file_names.append(os.path.join(root, file))
    numb_img = len(file_names)
    for i in range(numb_img):
        cls = 1
        ret = sh.start(file_names[i], cls)  # 1 - кв            0 - круг
        results.append(
            dict(
                img_path=f"{file_names[i]}",
                annot_path=ret[1],
                pipe_class=cls,
                pipe_count=ret[0],
            )
        )


    result_df = pd.DataFrame(results)
    result_df.to_csv("submission.csv",sep=';', index=False)
    print("Обработка завершина, подсчёт смотри в файле submission.csv")
