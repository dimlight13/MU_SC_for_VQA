import os
import numpy as np
import cv2
import h5py
from glob import glob
import json
from PIL import Image
from tqdm import tqdm
import os
import json
import pickle
import nltk
import tqdm
from PIL import Image
import h5py
import argparse

parser = argparse.ArgumentParser(description='Multi-modal VQA with Semantic Communication')
parser.add_argument('--raw_data_dir', type=str, default='./CLEVR_v1.0/', required=True)
args = parser.parse_args()


def resize_img(args):
    """
    Before computing semantic information, images will be resized to a commonly used resolution, 224 * 224
    S_I ∈ R^(1 * 3 * 224 * 224)
    """

    image_dir= args.raw_data_dir + 'images/'
    loop_list = ['train', 'val', 'test']

    for li in loop_list:
        path_dir = image_dir + li
        file_name_list = os.listdir(path_dir)

        for file_name in file_name_list:
            img_name = path_dir + '/' + file_name
            img = cv2.imread("{}".format(img_name)) # raw img shape = (320, 480, 3) --> (height, width, channel)
            resized = cv2.resize(img, (224, 224)) # change img size (320, 480, 3) to (224, 224, 3)
            cv2.imwrite('./resized_images/image/{}/{}'.format(li, file_name), resized)
    return

def load_image(source_path):
    if os.path.exists(source_path):
        img = Image.open(source_path)
        return np.array(img)
    else:
        print(f"Source file {source_path} does not exist.")
        return None

def preprocessing_image(args):
    question_file_path= args.raw_data_dir + 'questions/'

    loop_list = ['train', 'val']

    for li in loop_list:
        image_file_path = './input_data/image/' + li + '/'
        output_file_path = "image_" + li + '.h5'
        text_paths = glob(question_file_path + 'CLEVR_{}_questions.json'.format(li))[0]

        with open(text_paths, 'r') as file:
            data = json.load(file)
            image_filename = [datam['image_filename'] for datam in data['questions']]
            image_id = [datam['image_index'] for datam in data['questions']]

        source_paths = [image_file_path + filename for filename in image_filename]

        with h5py.File(output_file_path, 'w') as hf:
            added_image_ids = set()
            for i in tqdm(range(len(source_paths))):
                source_path = source_paths[i]
                image_data = np.array(Image.open(source_path))
                image_index = image_id[i]

                if image_index in added_image_ids:
                    continue

                hf.create_dataset(f'image_{image_index}', data=image_data, 
                                  compression="gzip", compression_opts=5)
                added_image_ids.add(image_index)

def preprocessing_que_texts(args):
    raw_data_dir = args.raw_data_dir
    root = raw_data_dir.rstrip('/')
    
    word_dic, answer_dic = process_question(root, 'train')
    process_question(root, 'val', word_dic, answer_dic)

    with open('./data/dic.pkl', 'wb') as f:
        pickle.dump({'word_dic': word_dic, 'answer_dic': answer_dic}, f)

def process_question(root, split, word_dic=None, answer_dic=None):
    if word_dic is None:
        word_dic = {}

    if answer_dic is None:
        answer_dic = {}

    with open(os.path.join(root, 'questions', 'CLEVR_{}_questions.json'.format(split))) as f:
        data = json.load(f)

    result = []
    word_index = 1
    answer_index = 0

    for question in tqdm.tqdm(data['questions']):
        words = nltk.word_tokenize(question['question'])
        question_token = []

        for word in words:
            try:
                question_token.append(word_dic[word])

            except:
                question_token.append(word_index)
                word_dic[word] = word_index
                word_index += 1

        answer_word = question['answer']

        try:
            answer = answer_dic[answer_word]

        except:
            answer = answer_index
            answer_dic[answer_word] = answer_index
            answer_index += 1

        result.append((question['image_filename'], question_token, answer, question['question_family_index']))
    with open('./data/{}.pkl'.format(split), 'wb') as f:
        pickle.dump(result, f)
    return word_dic, answer_dic

if __name__ == "__main__":
    resize_img(args)
    preprocessing_image(args)
    preprocessing_que_texts(args)