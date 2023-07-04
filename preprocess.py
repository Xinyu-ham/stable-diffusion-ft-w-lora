import os, json
from matplotlib import pyplot as plt
from torchvision import transforms

t = transforms.ToPILImage()

def get_img(path, input_dir):
    return plt.imread(os.path.join(input_dir, path))

def show_img(img):
    plt.imshow(img)
    plt.show()

def flip_image(img, angle: int):
    if angle == 90:
        return img.transpose(1, 0, 2)[:, ::-1, :]
    elif angle == 180:
        return img[::-1, ::-1, :]
    elif angle == 270:
        return img.transpose(1, 0, 2)[::-1, :, :]
    return img

def save_img(img, output_path):
    img = t(img)
    img.save(output_path)

def preprocess(img, size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    return transform(img)

def prepare_dataset(input_dir, output_dir, dim, split=0.2):
    size = (dim, dim)

    if not os.path.exists(output_dir + '/train'):
        os.makedirs(output_dir + '/train')

    if not os.path.exists(output_dir + '/test'):
        os.makedirs(output_dir + '/test')
    
    if split == False or round(i * split):
        output_root = output_dir + '/train/'
    else:
        output_root = output_dir + '/test/'
    meta_path = output_root + 'metadata.jsonl'
    if os.path.exists(meta_path):
        os.remove(meta_path)

    for i, file in enumerate([f for f in os.listdir(input_dir) if f.endswith('.JPG')]):
        img = get_img(file, input_dir)
        img = preprocess(img, size)
        
        output_path = output_root + f'img_{i}.jpg'
        caption = 'A cat named Milk Tea'
        meta = {
            'file_name': f'img_{i}.jpg',
            'text': caption
        }
        save_img(img, output_path)
        with open(meta_path, 'a') as f:
            f.write(json.dumps(meta) + '\n')

if __name__ == '__main__':
    prepare_dataset('data/raw', 'data', 500)