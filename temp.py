import torch

from data_loader.data_loaders import FaceExpressionPhoenixDataLoader

batch_size = 2

full_data = FaceExpressionPhoenixDataLoader(
    r'C:\Users\chira\EasierCloud\Shared\EkmanClassifierSharedData\FePh\FePh_images',
    r'C:\Users\chira\EasierCloud\Shared\EkmanClassifierSharedData\FePh\FePh_labels.csv')

train_loader = torch.utils.data.DataLoader(dataset=full_data, batch_size=batch_size, shuffle=True)

for i, (in_images, labels) in enumerate(train_loader):
    print(in_images, labels)
    in_images = in_images.type(torch.FloatTensor)
    labels = labels.type(torch.FloatTensor)
    print(in_images.size(), labels.size())
