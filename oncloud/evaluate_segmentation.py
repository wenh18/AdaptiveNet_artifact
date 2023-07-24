import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import cv2
import matplotlib.pyplot as plt
DATA_DIR = '../datasets/CamVid'

# load repo with data if it is not exists
# if not os.path.exists(DATA_DIR):
#     print('Loading data...')
#     os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial /home/data/CamVid/')
#     print('Done!')



x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist', 'unlabelled']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)

def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


# dataset = Dataset(x_train_dir, y_train_dir, classes=['car'])

# image, mask = dataset[5] # get some sample
# visualize(
#     image=image,
#     cars_mask=mask.squeeze(),
# )

import albumentations as albu


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def getlat(subnet):
    LATS = [0.004456847872962146, [[0.009028291012739578, 0.009015668353461582, 0.008647696194844994], [0.0074704138932954745, 0.007427512604879988], [0.007465844955274605]], [[0.008359470409864313, 0.008658498757673185, 0.0083835167402155], [0.0046740344687218926, 0.004371907740201515, 0.00437155926187789], [0.004678481147604869, 0.0043814227366208765], [0.0043719374430723264]], [[0.006039405691212621, 0.005812938539549559, 0.0058129942324323305], [0.00343704621439118, 0.0031960763708502884, 0.0032035583118442965], [0.0031941252635238697, 0.0032041921498910737, 0.003390271353376853], [0.0031928925943852, 0.003197399739826614, 0.0031978365311500626], [0.003344919578119433, 0.0032466964276136627], [0.0031936176626780406]], [[0.004732835809964359, 0.0049793823675000765, 0.004713312006898398], [0.002761899960849919, 0.0027702708663346373], [0.002841451409396128]], 0.0367988079711906]

    lat = LATS[0] + LATS[-1]
    for i in range(len(subnet)):
        for j in range(len(subnet[i])):
            if subnet[i][j] != 99:
                lat += LATS[i + 1][j][subnet[i][j]]
    return lat/32  # the batch size was 32 when building the latency table

import copy
def get_all_subnet(layerlen):
    subnets = [[]]
    blockidx = 0
    while blockidx < layerlen:
        choices = [0]
        if blockidx < layerlen-1:
            choices.append(1)
        if blockidx < layerlen-2:
            choices.append(2)
        for subnetidx in range(len(subnets)):
            if len(subnets[subnetidx]) == blockidx:
                subnets[subnetidx].append(0)
                for i in range(1, len(choices)):
                    subnets.append(copy.deepcopy(subnets[subnetidx]))
                    subnets[-1][-1]=choices[i]
                    if choices[i] == 1:
                        subnets[-1].append(99)
                    elif choices[i] == 2:
                        subnets[-1] += [99, 99]
        blockidx += 1
    return subnets
def get_all_encoder_subnet():
    layer1choices = get_all_subnet(3)
    layer2choices = get_all_subnet(4)
    layer3choices = get_all_subnet(6)
    layer4choices = get_all_subnet(3)
    subnets = []
    for i in range(len(layer1choices)):
        for j in range(len(layer2choices)):
            for m in range(len(layer3choices)):
                for n in range(len(layer4choices)):
                    subnets.append([layer1choices[i], layer2choices[j], layer3choices[m], layer4choices[n]])
    return subnets
print(len(get_all_encoder_subnet()))
# print(get_all_subnet(5))

import torch
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist', 'unlabelled']
ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

# import pdb;pdb.set_trace()
model.encoder.get_multi_resnet_backbone()
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=32)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

loss = utils.losses.DiceLoss()
metrics = [
    utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
])

train_epoch = utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

print('testing subnets:')
best_model = torch.load('weights/segmentation/best_multi_class_model.pth')
# best_model = torch.load('test.pth')
# create test dataset
test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)
subnets = get_all_encoder_subnet()
# subnets.reverse()
# np.random.shuffle(subnets)
subnetidx = 0
accs, lats = [], []
for subnet in subnets:
    # if subnetidx == 600:
    #     break
    # subnetidx += 1
    logs, lat = test_epoch.run(test_dataloader, subnet=subnet, getlat=True)
    accs.append(logs['iou_score'])
    lat = getlat(subnet)
    lats.append(lat)
    print('\n\n\n=======\n', 'subnet choice:', subnet, '\n', 'latency:', lat, 'accuracy:', logs['iou_score'], '\n=======\n')
print(accs, ',', lats)
exit(0)
train_curve = []
for i in range(0, 1000):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader, main=False, stage=2)
    valid_logs = valid_epoch.run(valid_loader, main=False)
    train_curve.append(valid_logs['iou_score'])
    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')

    if i == 300:
        optimizer.param_groups[0]['lr'] = 1e-4
        print('Decrease decoder learning rate to 1e-4!')

    if i == 400:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')


# load best saved checkpoint
best_model = torch.load('/home/wenh/Desktop/segmentation_models.pytorch/best_multi_model.pth')
# create test dataset
test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

test_dataloader = DataLoader(test_dataset)
# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)
logs = test_epoch.run(test_dataloader)
print(train_curve)
# lats = [0.0006080291692884401, [[0.0011172477607599752, 0.0011172082453336282, 0.0011180653853198976], [0.0009189702247220762, 0.0009168536299725661], [0.000919947767416813]], [[0.0012086358033244946, 0.0012221365537208497, 0.0012101474672854278], [0.0006532860014409457, 0.000653071186035971, 0.0006527282770006224], [0.0006541030955924606, 0.000655250899385955], [0.0006550634000139587]], [[0.0009606255837887094, 0.0009626443181870644, 0.0009633457833057781], [0.0005442982123081093, 0.0005451882376156341, 0.0005425887060112364], [0.0005405304561865343, 0.00054124650753645, 0.0005429536269954898], [0.0005395237410293935, 0.0005463898778624741, 0.0005407993732490582], [0.0005424754638162683, 0.0005428430368425584], [0.0005426817926867254]], [[0.0008723258441759561, 0.0008712366505114732, 0.0008710348301124785], [0.0005364667321736609, 0.0005389310096341856], [0.0005399265862147721]], 0.004941137360646437]
# lats = [3.762982445908866e-06, 0.00019515873460271068, [[0.0003311398031979435, 0.00033206934393181553, 0.0003326199078586396], [0.0002732446647194257, 0.0002751130283342983], [0.0002737994719135085]], [[0.0003549325983304203, 0.0003561819753339214, 0.00036075064815058725], [0.0002361208498809441, 0.0002344688928962682, 0.00023659291336348113], [0.00023466302237335648, 0.000234458549932325], [0.00023611342416324127]], [[0.0003435500339087973, 0.0003456668938625111, 0.00034501104386682905], [0.00021235537078144554, 0.00021308070427848976, 0.00021215089834041405], [0.00021184432228096866, 0.00021202466113946596, 0.00021156877511335296], [0.0002124481922527309, 0.00021136987196059858, 0.00021240416835492128], [0.0002116761828158403, 0.00021148178813454836], [0.0002126929757327206]], [[0.0003693018924938028, 0.00036913083578243405, 0.0003744667443603244], [0.00026396967810438793, 0.0002621742456455252], [0.000263288633709357]], 0.001689274328512385]