from pathlib import Path
import argparse
import time

import torch
from skimage import io
from torchvision import utils as vutils
import numpy as np

from preprocessing import enhance_image
from model import RRWNet, CMRRWNet
from utils import pad_images_unet, to_torch_tensors


def predict_probability(model, image_tensor, refine, tta):
    """Run flip TTA and restore every prediction to the original orientation."""
    flip_dimensions = {
        'none': [()],
        'hflip': [(), (-1,)],
        'vflip': [(), (-2,)],
        'hvflip': [(), (-1,), (-2,), (-2, -1)],
    }[tta]
    probabilities = []
    for dimensions in flip_dimensions:
        transformed = (
            torch.flip(image_tensor, dims=dimensions)
            if dimensions else image_tensor
        )
        outputs = model.refine(transformed) if refine else model(transformed)
        probability = outputs[-1]
        if not refine:
            probability = torch.sigmoid(probability)
        if dimensions:
            probability = torch.flip(probability, dims=dimensions)
        probabilities.append(probability)
    return torch.stack(probabilities).mean(dim=0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get predictions from a model')
    parser.add_argument('--weights', type=str, default='',
        help='Path to the model weights')
    parser.add_argument('--images-path', type=str, default='',
        help='Path to the images')
    parser.add_argument('--a-path', type=str, default='',
        help='Path to the masks')
    parser.add_argument('--av-path', type=str, default='',
        help='Path to the masks')
    parser.add_argument('--masks-path', type=str, default='',
        help='Path to the masks')
    parser.add_argument('--save-path', type=str, default='',
        help='Path to save the predictions')
    parser.add_argument('--in_channels', type=int, default=3,
        help='Number of input channels for the model.')
    parser.add_argument('--base_channels', type=int, default=64,
        help='Number of base channels used when the model was trained.')
    parser.add_argument('--gpu_id', type=int, default=0,
        help='GPU ID to use. ')
    parser.add_argument('--preprocess', action='store_true',
        help='Preprocess the images')
    parser.add_argument('--refine', action='store_true',
        help='Refine the predictions')
    parser.add_argument('--k', type=int, default=5,
        help='Number of iterations for the refinement module. Default=5')
    parser.add_argument(
        '--tta',
        choices=['none', 'hflip', 'vflip', 'hvflip'],
        default='none',
        help='Flip TTA; hvflip averages original, H, V, and H+V predictions',
    )
    parser.add_argument('--wandb_project', default=None)
    parser.add_argument('--wandb_entity', default=None)
    parser.add_argument(
        '--wandb_mode',
        choices=['online', 'offline', 'disabled'],
        default='online',
    )
    parser.add_argument('--wandb_run_name', default=None)
    parser.add_argument('--wandb_task', default=None)
    parser.add_argument('--wandb_tags', nargs='*', default=None)
    args = parser.parse_args()

    tracker = None
    started = time.perf_counter()
    if args.wandb_project and args.wandb_mode != 'disabled':
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                'W&B tracking was requested but wandb is not installed.'
            ) from error
        task = args.wandb_task or (
            'Task1' if args.in_channels == 3 else 'Task2'
        )
        tracker = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            job_type='inference',
            tags=tuple(dict.fromkeys([
                'GAVE2', task, 'inference', *(args.wandb_tags or []),
            ])),
            config=vars(args),
            mode=args.wandb_mode,
            save_code=True,
        )

    if args.in_channels == 3:
        model = RRWNet(iterations=args.k, input_ch=args.in_channels, base_ch=args.base_channels)
    elif args.in_channels == 5:
        model = CMRRWNet(iterations=args.k, input_ch=args.in_channels, base_ch=args.base_channels)
    else:
        raise ValueError('Unsupported number of input channels: {}'.format(args.in_channels))
    

    print(f'Loading model from {args.weights}')
    if torch.cuda.is_available():
        
        
        gpu_id = args.gpu_id
        print(f"Using GPU: {gpu_id}")
        model.eval()
        model.cuda()
        model.load_state_dict(torch.load(args.weights, map_location=f'cuda:{gpu_id}'), strict=True)
        device = torch.device("cuda")



    else:
        model.eval()
        model.cpu() 
        model.load_state_dict(torch.load(args.weights, map_location='cpu'), strict=True)
        device = torch.device("cpu")

    print(f'Creating save path {args.save_path}')
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f'Getting images, a, av, masks from {args.images_path} , {args.a_path}, {args.av_path}, {args.masks_path}')
    image_fns = sorted(Path(args.images_path).glob('*.png'))
    a_fns = sorted(Path(args.a_path).glob('*.png'))
    av_fns = sorted(Path(args.av_path).glob('*.png'))
    mask_fns = sorted(Path(args.masks_path).glob('*.png'))

    print('Processing images')
    for image_fn, mask_fn, a_fn, av_fn in zip(image_fns, mask_fns, a_fns, av_fns):
        print(f'  {image_fn.name}')
        assert Path(mask_fn).stem == Path(image_fn).stem
        assert Path(a_fn).stem == Path(image_fn).stem
        assert Path(av_fn).stem == Path(image_fn).stem 
        img = (io.imread(image_fn) / 255.0)[..., :3]
        mask = io.imread(mask_fn) * 1.0
        r_a = (io.imread(a_fn) / 255.0)[..., :1]
        r_av = (io.imread(av_fn) / 255.0)[..., :1]

        if args.in_channels == 5:
            img_5ch = np.concatenate([img, r_a, r_av], axis=2)
            imgs, paddings = pad_images_unet([img_5ch, mask])
        else: 
            imgs, paddings = pad_images_unet([img, mask])
        img = imgs[0]
        padding = paddings[0]
        mask = imgs[1]
        mask = np.stack([mask,] * 3, axis=2)
        mask_padding = paddings[1]
        tensors = to_torch_tensors([img, mask])
        image_tensor = tensors[0]
        mask_tensor = tensors[1]
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
            mask_tensor = mask_tensor.cuda()
        else:
            image_tensor = image_tensor.cpu()
            mask_tensor = mask_tensor.cpu()
        image_tensor = image_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)
        with torch.no_grad():
            last_pred = predict_probability(
                model,
                image_tensor,
                refine=args.refine,
                tta=args.tta,
            )
            last_pred[mask_tensor < 0.5] = 0
            last_pred = last_pred[:, :, padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1]]
            target_fn = save_path / Path(image_fn).name
            vutils.save_image(last_pred, target_fn)

    if tracker is not None:
        tracker.log({
            'inference/images': len(image_fns),
            'inference/runtime_seconds': time.perf_counter() - started,
        })
        tracker.summary['inference/output_directory'] = str(
            save_path.resolve()
        )
        tracker.finish()
