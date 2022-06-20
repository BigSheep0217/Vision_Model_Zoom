import torch

from model import Basic_Model


if __name__ == '__main__':
    model = Basic_Model(if_export=True)
    checkpoint = torch.load('checkpoints/best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    torch.onnx.export(model, torch.randn(1, 3, 56, 224), 'checkpoints/mask.onnx', opset_version=11)
    mask_torchscript = torch.jit.trace(model, torch.randn(1, 3, 56, 224))
    mask_torchscript.save('checkpoints/mask.torchscript')


    