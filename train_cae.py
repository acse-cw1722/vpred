import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda
from CAE import VideoFrame, Encoder, Decoder

batch_size = 32

transform = Lambda(lambda x: x / 255.)

X_train = VideoFrame(split='train', transform=transform)
X_test = VideoFrame(split='test', transform=transform)

X_train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
X_test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=True)

lr = 1e-3

encoder = Encoder(code_dim=16)
decoder = Decoder(code_dim=16)

params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]
optimizer = torch.optim.Adam(
    params_to_optimize, lr=lr, weight_decay=1e-05
)
loss_fn = torch.nn.MSELoss()


def train_epoch(encoder, decoder, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    for image_batch in dataloader:
        image_batch = image_batch
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


def test_epoch(encoder, decoder, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


num_epochs = 25
diz_loss = {
    'train_loss': [],
    'val_loss': []
}

for epoch in range(num_epochs):
    train_loss = train_epoch(
        encoder, decoder, X_train_loader, loss_fn, optimizer)
    val_loss = test_epoch(encoder, decoder, X_test_loader, loss_fn)
    print(
        'EPOCH {}/{} \t train loss {} \t val loss {}'
        .format(epoch + 1, num_epochs, train_loss, val_loss))
    diz_loss['train_loss'].append(train_loss)
    diz_loss['val_loss'].append(val_loss)


def save_model(model, name='model.pt'):
    file_path = os.sep.join([
        os.path.dirname(__file__),
        name
    ])
    print(file_path)
    torch.save(model.state_dict(), file_path)


save_model(encoder, name='encoder.pt')
save_model(decoder, name='decoder.pt')
