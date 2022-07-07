import json
import os


def json_save(file, save_path, save_file_name):
    """

    :param file:
    :param save_path:
    :param save_file_name:
    :return:
    """
    if save_file_name.split(".")[-1] == "json":
        save_file_name = save_file_name
    else:
        save_file_name = f"{save_file_name}.json"
    path = os.path.join(save_path, save_file_name)

    with open(path, "w", encoding="utf-8") as JSON:
        json.dump(file, JSON, ensure_ascii=False, indent="\t")
    result_for_return = json.dumps(file, ensure_ascii=False, indent="\t")
    return result_for_return
